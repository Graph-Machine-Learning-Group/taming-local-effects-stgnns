import sys, os, glob
import warnings

import numpy as np
import torch
import einops
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from tsl import logger
from tsl.data import SpatioTemporalDataset, SpatioTemporalDataModule
from tsl.data.preprocessing import scalers
from tsl.datasets import MetrLA, PemsBay, SolarBenchmark, PvUS, Elergone
from tsl.datasets.pems_benchmarks import PeMS03, PeMS04, PeMS07, PeMS08
from tsl.experiment import Experiment, NeptuneLogger
from tsl.metrics import torch_metrics

from lib import config as lib_config
from lib.datasets import LocalGlobalGPVARDataset, AirQuality, EngRad
from lib.nn import models, EmbeddingPredictor
from lib.utils import find_devices, cfg_to_python, cfg_to_neptune
sys.path.append(os.path.abspath(os.path.join(os.path.curdir, "graph_sign_test")))
from az_analysis.stat_test import optimality_check
from az_analysis.visualization import az_score_plot, save_az_scores, AZ_COLORS


def get_model_class(model_str):
    # Basic models  #####################################################
    if model_str == 'ttg_iso':
        model = models.TimeThenGraphIsoModel
    elif model_str == 'ttg_aniso':
        model = models.TimeThenGraphAnisoModel
    elif model_str == 'tag_iso':
        model = models.TimeAndGraphIsoModel
    elif model_str == 'tag_aniso':
        model = models.TimeAndGraphAnisoModel
    # Baseline models  ##################################################
    elif model_str == 'rnn':
        model = models.RNNModel
    elif model_str == 'fcrnn':
        model = models.FCRNNModel
    elif model_str == 'local_rnn':
        model = models.LocalRNNModel
    # SOTA baseline models  #############################################
    elif model_str == 'dcrnn':
        model = models.DCRNNModel
    elif model_str == 'gwnet':
        model = models.GraphWaveNetModel
    elif model_str == 'agcrn':
        model = models.AGCRNModel
    else:
        raise NotImplementedError(f'Model "{model_str}" not available.')
    return model


def get_dataset(dataset_cfg):
    name = dataset_cfg.name
    if name == 'la':
        dataset = MetrLA(impute_zeros=True)
    elif name == 'bay':
        dataset = PemsBay()
    elif name == 'pems3':
        dataset = PeMS03()
    elif name == 'pems4':
        dataset = PeMS04()
    elif name == 'pems7':
        dataset = PeMS07()
    elif name == 'pems8':
        dataset = PeMS08()
    elif name == 'air':
        dataset = AirQuality(impute_nans=True)
        if dataset_cfg.get("test_months", False):
            dataset.test_months = dataset_cfg.test_months # (11,12,1,2,3,4)
            data_ = np.ma.array(dataset.target.to_numpy(), mask=dataset.mask[..., 0])
            for i in range(0, data_.shape[0], data_.shape[0]//200):
                print(f"{int(5 + i/data_.shape[0]*12)%12}\t{data_[i:i+50].mean():.1f}\t{data_[i:i+50].std():.1f}")
    elif name == 'engrad':
        dataset = EngRad(**dataset_cfg.hparams)
        # Add just one valid night values for MinMaxScaler
        dataset.mask[:24] = True
    elif name == 'gpvar':
        if "p_max" in dataset_cfg.hparams:
            dataset_cfg.hparams["p_max"] = 0
            dataset = LocalGlobalGPVARDataset(**dataset_cfg.hparams)
        else:
            dataset = LocalGlobalGPVARDataset(**dataset_cfg.hparams, p_max=0)
    elif name == 'lgpvar':
        dataset = LocalGlobalGPVARDataset(**dataset_cfg.hparams)
    elif name == 'lcgpvar':
        dataset = LocalGlobalGPVARDataset(**dataset_cfg.hparams)
    elif name == 'solar':
        dataset = SolarBenchmark()
    elif name == "pvwest":
        dataset = PvUS(**dataset_cfg.hparams)
        data_ = dataset.target.to_numpy()
        step_ = data_.shape[0]//48
        for i in range(0, data_.shape[0], step_):
            print(f"{int(1 + i/data_.shape[0]*12)%12}\t{data_[i:i+step_].mean():.1f}\t{data_[i:i+step_].std():.1f}")
    elif name == "elergone":
        dataset = Elergone()
    else:
        raise ValueError(f"Dataset {name} not available.")
    return dataset

def recreate_cfg(cfg_ckpt, cfg_az_analysis):
    logger.info(f"loading form {cfg_ckpt}")
    # checkpoint = os.path.abspath(args.from_checkpoint)
    # logging_path = os.path.abspath(os.path.join("/", "/".join(checkpoint.split("/")[:-1])))
    logging_path = os.path.abspath(cfg_ckpt)
    model_checkpoint = glob.glob(os.path.join(logging_path, "epoch=*"))
    assert len(model_checkpoint) == 1
    model_checkpoint = model_checkpoint[0]
    config_file = os.path.abspath(os.path.join(logging_path, "config.yaml"))
    # result_file = os.path.abspath(os.path.join(logging_path, "results_.npy"))
    
    logger.info(f"Reading config_file: {config_file}")
    # stored_cfg = tsl.Config.from_config_file(config_file)
    stored_cfg = OmegaConf.load(config_file)
    stored_cfg.ckpt = cfg_ckpt
    # stored_cfg.az_analysis = cfg_az_analysis
    # use_mask_ = cfg.get("use_mask", True)
    # cfg = stored_cfg
    stored_cfg.neptune.online = False
    # cfg.use_mask = use_mask_
    for k, v in stored_cfg.items():
        logger.info(f"{k:25s}: {v}")
    return stored_cfg, model_checkpoint

def run_traffic(cfg: DictConfig):

    if cfg.ckpt:
        cfg, model_ckpt = recreate_cfg(cfg.ckpt, cfg.az_analysis)

    ########################################
    # data module                          #
    ########################################
    dataset = get_dataset(cfg.dataset)

    covariates = dict()
    if cfg.get('add_exogenous'):
        assert not isinstance(dataset, LocalGlobalGPVARDataset)
        # encode time of the day and use it as exogenous variable
        u = [dataset.datetime_encoded('day').values]

        if isinstance(dataset, (EngRad, PvUS)):
            u.append(dataset.datetime_encoded('year').values)
        else:
            u.append(dataset.datetime_onehot('weekday').values)
        # covariates.update(u=np.concatenate(u, axis=-1))
        if 'u' in dataset.covariates:
            u.append(dataset.get_frame('u', return_pattern=False))

        if cfg.model.name == "fcrnn":
            u = np.concatenate([u_.reshape(*u_.shape[:-2], -1)
                                if u_.ndim == 3 else u_
                                for u_ in u], axis=-1)
        else:
            ndim = max(u_.ndim for u_ in u)
            u = np.concatenate([np.repeat(u_[:, None], dataset.n_nodes, 1)
                                if u_.ndim < ndim else u_
                                for u_ in u], axis=-1)
        covariates.update(u=u)

    torch_dataset = SpatioTemporalDataset(target=dataset.dataframe(),
                                          mask=dataset.mask,
                                          covariates=covariates,
                                          horizon=cfg.horizon,
                                          window=cfg.window,
                                          stride=cfg.stride)
    if cfg.get('mask_as_exog', False) and 'u' in torch_dataset:
        torch_dataset.update_input_map(u=['u', 'mask'])

    # Scale input features
    scaler_cfg = cfg.get('scaler')
    if scaler_cfg is not None:
        scale_axis = (0,) if scaler_cfg.axis == 'node' else (0, 1)
        scaler_cls = getattr(scalers, f'{scaler_cfg.method}Scaler')
        transform = dict(target=scaler_cls(axis=scale_axis))
    else:
        transform = None

    dm = SpatioTemporalDataModule(
        dataset=torch_dataset,
        scalers=transform,
        splitter=dataset.get_splitter(**cfg.dataset.splitting),
        batch_size=cfg.batch_size,
        workers=cfg.workers
    )
    dm.setup()

    # get adjacency matrix
    adj = dataset.get_connectivity(**cfg.dataset.connectivity,
                                   train_slice=dm.train_slice)
    dm.torch_dataset.set_connectivity(adj)

    # Normalize exogenous with high mean (i.e., EngRad temperature)
    if isinstance(dataset, EngRad) and 'u' in torch_dataset:
        u = torch_dataset.u
        axis = list(range(u.ndim - 1))
        u_mu = torch_dataset.u[dm.train_slice].mean(axis=axis)
        u_std = torch_dataset.u[dm.train_slice].std(axis=axis)
        um = u_mu > 3
        torch_dataset.u[..., um] = (u[..., um] - u_mu[..., um]) / u_std[..., um]

    ########################################
    # Create model                         #
    ########################################

    model_cls = get_model_class(cfg.model.name)

    d_exog = torch_dataset.input_map.u.shape[-1] if 'u' in torch_dataset else 0
    model_kwargs = dict(n_nodes=torch_dataset.n_nodes,
                        input_size=torch_dataset.n_channels,
                        exog_size=d_exog,
                        output_size=torch_dataset.n_channels,
                        weighted_graph=torch_dataset.edge_weight is not None,
                        embedding_cfg=cfg.get('embedding'),
                        horizon=torch_dataset.horizon)

    model_cls.filter_model_args_(model_kwargs)
    model_kwargs.update(cfg.model.hparams)

    ########################################
    # predictor                            #
    ########################################

    loss_fn = torch_metrics.MaskedMAE()

    log_metrics = {'mae': torch_metrics.MaskedMAE(),
                   'mse': torch_metrics.MaskedMSE(),
                   'mape': torch_metrics.MaskedMAPE(),
                   'mre': torch_metrics.MaskedMRE(),
                   'mae_at_step_1': torch_metrics.MaskedMAE(at=0),
                   f'mae_at_step_{cfg.horizon}':
                       torch_metrics.MaskedMAE(at=cfg.horizon - 1)
                   }

    if cfg.get('lr_scheduler') is not None:
        scheduler_class = getattr(torch.optim.lr_scheduler,
                                  cfg.lr_scheduler.name)
        scheduler_kwargs = dict(cfg.lr_scheduler.hparams)
    else:
        scheduler_class = scheduler_kwargs = None

    # setup predictor
    predictor = EmbeddingPredictor(
        model_class=model_cls,
        model_kwargs=model_kwargs,
        optim_class=getattr(torch.optim, cfg.optimizer.name),
        optim_kwargs=dict(cfg.optimizer.hparams),
        loss_fn=loss_fn,
        metrics=log_metrics,
        beta=cfg_to_python(cfg.regularization_weight),
        embedding_var=cfg.embedding.get('initial_var', 0.2),
        # log_embeddings_every=5 if 'plot_embeddings' in cfg.tags else None,
        scheduler_class=scheduler_class,
        scheduler_kwargs=scheduler_kwargs,
        scale_target=False if scaler_cfg is None else scaler_cfg.scale_target,
    )

    ########################################
    # logging options                      #
    ########################################

    if cfg.get("tags", False):
        tags = cfg.tags
        tags = tags.split(",") if isinstance(tags, str) else list(tags)
    else:
        tags = []
    tags = tags + [cfg.model.name, cfg.dataset.name, cfg.embedding.method]
    tags = list(set(tags))

    if cfg.neptune:
        run_args = cfg_to_neptune(cfg)
        run_args['model']['trainable_parameters'] = predictor.trainable_parameters
        exp_logger = NeptuneLogger(api_key=lib_config["neptune_token"],
                                   project_name=lib_config["neptune_project"],
                                   experiment_name=cfg.run.name,
                                   save_dir=cfg.run.dir,
                                   tags=tags,
                                   params=run_args,
                                   debug=not cfg.neptune.online,
                                #    upload_stdout=False,
                                   )
        # TODO this is for a known bug in neptunes
        # https://github.com/neptune-ai/neptune-client/issues/1702#issuecomment-2376615676
        import logging, neptune
        class _FilterCallback(logging.Filterer):
            def filter(self, record: logging.LogRecord):
                return not (
                    record.name == "neptune"
                    and record.getMessage().startswith(
                        "Error occurred during asynchronous operation processing: X-coordinates (step) must be strictly increasing for series attribute"
                    )
                )
        neptune.internal.operation_processors.async_operation_processor.logger.addFilter(
            _FilterCallback()
        )
    else:
        exp_logger = TensorBoardLogger(save_dir=cfg.run.dir, name=cfg.run.name)

    ########################################
    # training                             #
    ########################################

    early_stop_callback = EarlyStopping(
        monitor='val_mae',
        patience=cfg.patience,
        mode='min'
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.run.dir,
        save_top_k=1,
        monitor='val_mae',
        mode='min',
    )

    trainer = Trainer(max_epochs=cfg.epochs,
                      limit_train_batches=cfg.train_batches,
                      default_root_dir=cfg.run.dir,
                      logger=exp_logger,
                      accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                      devices=find_devices(1),
                      gradient_clip_val=cfg.grad_clip_val,
                      callbacks=[early_stop_callback, checkpoint_callback])

    # load_model_path = cfg.get('load_model_path')
    if cfg.get("ckpt", False):
        predictor.load_model(model_ckpt)
    else:
        trainer.fit(predictor, train_dataloaders=dm.train_dataloader(),
                    val_dataloaders=dm.val_dataloader())
        predictor.load_model(checkpoint_callback.best_model_path)

    ########################################
    # testing                              #
    ########################################

    predictor.freeze()
    # trainer.test(predictor, dataloaders=dm.test_dataloader())

    ########################################
    # residual analysis                    #
    ########################################

    a = trainer.predict(predictor, dataloaders=dm.test_dataloader(shuffle=False))
    out = {k: torch.cat([a_[k] for a_ in a]) for k in a[0].keys()}
    y_hat, y, m = out["y_hat"], out["y"], out.get("mask", None)

    for f in range(y.shape[-1]):
        for h in range(y.shape[-3]):
            for name, metr in log_metrics.items(): # [torch_metrics.MaskedMRE(), torch_metrics.MaskedMAE(), torch_metrics.MaskedMSE()]:
                print(f"{name}[{h},{f}]: {metr(y_hat[..., h, :, f], y[..., h, :, f], None if m is None else m[..., h, :, f]):.3f}", end="\t")
            print("")
    
    res = einops.rearrange(y_hat - y, "b h n f -> b n (h f) ")
    if m is not None:
        m = einops.rearrange(m, "b h n f -> b n (h f) ")
    if cfg.dataset.name in ["pvwest", "engrad"]:
        time_index = dm.torch_dataset.index[dm.test_slice.numpy()[cfg.window:dm.test_slice.numpy().shape[0]-cfg.horizon+1]]
    else:
        time_index = None

 
    print("residuals shape:", res.shape)

    ei, ew = dataset.get_connectivity(layout="edge_index")
    graph = dict(edge_index=ei, edge_weight=ew)

    log_metric_ = None if not isinstance(exp_logger, NeptuneLogger) else lambda n, v: exp_logger.log_metric(metric_name=n, metric_value=v)
    # optimality_check(residuals=res, mask=m, multivariate=cfg.az_analysis.multivariate, downsample=cfg.az_analysis.downsample, remove_median=True, 
    #                  logger_msg=logger, logger_metric=log_metric_,
    #                  **graph)
    figname = f"{cfg.dataset.name}_{cfg.model.name}_{cfg.embedding.method}"
    if cfg.az_analysis.use_mask:
        figname += "_masked"

    savepath=cfg.ckpt if cfg.ckpt else cfg.run.dir
    common_args = dict(residuals=res, mask=m, savepath=savepath, time_index=time_index, log=logger.info)

    adjusted_scores = cfg.az_analysis.get("adjusted_scores", True)
    multivariate = res.shape[2] > 1 
    if cfg.az_analysis.feat_set is not None:
        multivariate = multivariate and len(cfg.az_analysis.feat_set) > 1
        
    metric_name = "residuals/az-test-stat"
    other_cmn_args = dict(cfg=cfg, savepath=savepath, mask=m, res=res, y=y, y_hat=y_hat, dataset_name=cfg.dataset.name)

    if adjusted_scores: # if adjusted_scores, run then non adjusted scores analysis too
        OmegaConf.update(cfg, "az_analysis.adjusted_scores", False, force_add=True)
        if multivariate: # if multivariate run the univariate too
            logger.info(" ------------ base + UV ---------------------")
            az_scores_base_uni = save_az_scores(
                **common_args, **graph, **cfg.az_analysis,
                savefig=figname,
                multivariate=False,
            )
        logger.info(" ------------ base + (MV)---------------------")
        az_scores_base = save_az_scores(
            **common_args, **graph, **cfg.az_analysis,
            savefig=figname,
            multivariate=multivariate,
        )
        plot_radiation_figures(scores=az_scores_base, **other_cmn_args, suffix="base")
        log_metric_(metric_name+"[T]", az_scores_base["glob_0.0"])
        log_metric_(metric_name+"[ST]", az_scores_base["glob_0.5"])
        log_metric_(metric_name+"[S]", az_scores_base["glob_1.0"])
        OmegaConf.update(cfg, "az_analysis.adjusted_scores", adjusted_scores, force_add=True) # reset to its original value

    if adjusted_scores:
        metric_name = metric_name+"-adj"
        figname=figname + "_adjusted"
    if multivariate: # if multivariate run the univariate too
        logger.info(" ------------ (adj) + UV ---------------------")
        az_scores_uni = save_az_scores(
            **common_args, **graph, **cfg.az_analysis,
            savefig=figname,
            multivariate=False,
        )
        plot_radiation_figures(scores=az_scores_uni, **other_cmn_args, suffix="adj_uv")
    logger.info(" ------------ (adj) + (MV) ---------------------")
    az_scores = save_az_scores(
        **common_args, **graph, **cfg.az_analysis,
        savefig=figname,
        multivariate=multivariate,
    )
    plot_radiation_figures(scores=az_scores, **other_cmn_args, suffix="adj")
    log_metric_(metric_name+"[T]", az_scores["glob_0.0"])
    log_metric_(metric_name+"[ST]", az_scores["glob_0.5"])
    log_metric_(metric_name+"[S]", az_scores["glob_1.0"])


    exp_logger.finalize('success')

def plot_radiation_figures(cfg, scores, savepath, mask, res, y, y_hat, suffix, dataset_name):
    if dataset_name not in ["pvwest", "engrad"]:
        return
    T0, T1 = cfg.az_analysis.get("time_set", [0, 400])
    feat_set = [0] if "uv" in suffix else cfg.az_analysis.get("feat_set", list(range(res.shape[-1])))
    res_ = res[T0: T1, :, feat_set]
    mask_ = mask[T0: T1, :, feat_set]
    y_ = y[T0: T1, feat_set, :, 0].transpose(-1, -2)
    # night_ = torch.where((~mask_).any(-1).sum(-1) > res_.shape[1] * .95)[0].numpy()
    # night_ = torch.where((~mask_).all(-1).all(-1))[0].numpy()
    # day_mask_ = (mask_[..., 0].sum(1, keepdim=True)>.5*mask_.shape[1])
    # night_idx_ = torch.where(~day_mask_[..., 0])[0].numpy()
    day_mask_ = ~np.isnan(scores["time_0.5"])

    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    warnings = []
    for s in ["time_mae", "time_0.0", "time_0.5", "time_1.0"]:
        if not np.all(np.isnan(scores[s][~day_mask_])):
            warnings.append(s) 
            scores[s][~day_mask_] = np.nan
            print("WARNING", s)


    # --- Sign of residuals ------------------------------------------------------------------------------------------------------

    # plt.figure(figsize=[8, 3])
    # for f_ in range(res_.shape[-1]):
    #     plt.plot(scores["time_index"], torch.mean(res_[..., f_]>0, dtype=float, axis=1), label=f"{feat_set[f_]}-step ahead", color=plt.cm.Set2.colors[f_])
    # # plt.plot(torch.mean(res[:T_, :, [0,2,5]]>0, dtype=float, axis=1))       
    # plt.gca().tick_params(axis='x', labelrotation=90)
    # plt.gca().xaxis.set_major_locator(mdates.HourLocator(byhour=(0, 3, 6, 9, 12, 15, 18, 21)))
    # plt.legend()
    # plt.grid()
    # plt.tight_layout()
    # plt.savefig(os.path.join(savepath, f"tmp3_{suffix}_positive_res_ratio.pdf"))
    # plt.close()

    figsize = [9, 3.8] if dataset_name == "engrad" else [7, 3.5]
    abs_ = (res_ * mask_).abs().mean(-1) # set to zero masked value and computes MAE on feat_dim
    # f_ = 0 # len(feat_set) - 1
    # signal = y[T0: T1, :, :, 0].mean(1)
    signal_ = (y_ * mask_).abs().mean(-1) # set to zero masked value and computes MAE on feat_dim

    # abs_[torch.where((~mask_).any(-1))] = torch.nan
    # signal[torch.where((~mask_).any(-1))] = torch.nan
    abs_[~day_mask_] = torch.nan
    signal_[~day_mask_] = torch.nan
    rel_error = 100 * abs_ / signal_

    # signal_[time_mask_] = torch.nan
    fig, axs = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # --- Scores and MAE -----------------------------------------------------------------------------------------------------
    
    l1 = axs[0].plot(scores["time_index"], scores["time_1.0"], label=f"Score $c_{{1}}(t)$", color=AZ_COLORS[1.0], linestyle="dashdot", linewidth=1.)
    l0 = axs[0].plot(scores["time_index"], scores["time_0.0"], label=f"Score $c_{{0}}(t)$", color=AZ_COLORS[0.0], linestyle="dashed", linewidth=1.)
    l5 = axs[0].plot(scores["time_index"], scores["time_0.5"], label=f"Score $c_{{1/2}}(t)$", color=AZ_COLORS[0.5], linestyle="solid", linewidth=1.)

    ax0b = axs[0].twinx()
    la = ax0b.plot(scores["time_index"], scores["time_mae"], 
                    label=f"MAE", linestyle="solid", color="gray", alpha=0.8, linewidth=1.)

    # axs[0].set_ylim(top=1.0, bottom= -1)
    axs[0].set_ylabel(r"Scores $c_\lambda(t)$")
    # axs[0].tick_params(axis='y', labelcolor=AZ_COLORS[0.5])    
    axs[0].grid()    
    
    ax0b.set_ylabel(f"MAE")
    ax0b.yaxis.label.set_color("gray")
    ax0b.tick_params(axis='y', labelcolor="gray")       

    lns = l1 + l5 + l0 + la
    labs = [l.get_label() for l in lns]

    # --- Relative Error ------------------------------------------------------------------------------------------------------

    lm = axs[1].plot(scores["time_index"], signal_.nanmean(axis=-1), label=f"Target $\mathbf y$", color="darkgoldenrod", 
                    linestyle="solid", linewidth=1.)
    axs[1].fill_between(scores["time_index"], 
                        torch.nanquantile(signal_, .25, axis=-1),
                        torch.nanquantile(signal_, .75, axis=-1),
                    color="goldenrod", alpha=.5)
    ax1b = axs[1].twinx()  # instantiate a second Axes that shares the same x-axis
    lr = ax1b.plot(scores["time_index"], torch.nanmean(rel_error, axis=-1), 
                    label=f"MAPE", color="gray", linestyle="dashed", linewidth=1.)

    if dataset_name == "pvwest":
        axs[1].xaxis.set_major_locator(mdates.HourLocator(byhour=(0, 3, 6, 9, 12, 15, 18, 21)))        
        axs[1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axs[1].xaxis.get_major_locator()))     
        axs[1].tick_params(axis='x', labelrotation=90)
    else:
        axs[1].set_xticks(scores["time_index"][::24])        
        axs[1].tick_params(axis='x', labelrotation=90)   
        axs[1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axs[1].xaxis.get_major_locator()))     

    axs[1].set_ylim(bottom=0)
    axs[1].set_ylabel("Target value")
    axs[1].yaxis.label.set_color("darkgoldenrod")
    axs[1].tick_params(axis='y', labelcolor="darkgoldenrod")   
    # ax1.tick_params(axis='y', labelcolor=AZ_COLORS[0.5])    
    axs[1].grid()    
    
    # plt.plot(torch.mean(res[:T_, :, [0,2,5]]>0, dtype=float, axis=1))       
    ax1b.set_ylim(bottom=0.0, top=min([500, rel_error[day_mask_].nanmean(-1).max()]))
    # ax2.set_xticks()
    ax1b.set_ylabel(f"MAPE (\%)")
    ax1b.yaxis.label.set_color("gray")
    ax1b.tick_params(axis='y', labelcolor="gray")       


    lns = l1 + l5 + l0 + lm + lr + la
    # lns = lm + lr
    labs = [l.get_label() for l in lns]
    ax0b.legend(lns, labs, loc="lower center", ncols=3, bbox_to_anchor=(0.5, 1.))

    # plt.legend()
    # plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(savepath, f"tmp4.2b_{suffix}.pdf"))
    plt.close()



if __name__ == '__main__':
    exp = Experiment(run_fn=run_traffic, config_path='../config/static',
                     config_name='default')
    res = exp.run()
    logger.info(res)
