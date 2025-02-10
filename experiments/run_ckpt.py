import sys, os, glob

import numpy as np
import torch
import einops
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from tsl import logger
from tsl.data import SpatioTemporalDataset, SpatioTemporalDataModule
from tsl.data.preprocessing import StandardScaler
from tsl.datasets import MetrLA, PemsBay, SolarBenchmark, PvUS, Elergone
from tsl.datasets.pems_benchmarks import PeMS03, PeMS04, PeMS07, PeMS08
from tsl.experiment import Experiment, NeptuneLogger
from tsl.metrics import torch_metrics

from lib import config as lib_config
from lib.datasets import LocalGlobalGPVARDataset, AirQuality
from lib.nn import models, EmbeddingPredictor
from lib.utils import find_devices, cfg_to_python

sys.path.append(os.path.abspath(os.path.join(os.path.curdir, "graph_sign_test")))
from az_analysis.stat_test import optimality_check
from az_analysis.visualization import az_score_plot, save_az_scores


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
    stored_cfg.az_analysis = cfg_az_analysis
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
        # logger.info(f"loading form {cfg.ckpt}")
        # # checkpoint = os.path.abspath(args.from_checkpoint)
        # # logging_path = os.path.abspath(os.path.join("/", "/".join(checkpoint.split("/")[:-1])))
        # logging_path = os.path.abspath(cfg.ckpt)
        # checkpoint = glob.glob(os.path.join(logging_path, "epoch=*"))
        # assert len(checkpoint) == 1
        # checkpoint = checkpoint[0]
        # config_file = os.path.abspath(os.path.join(logging_path, "config.yaml"))
        # # result_file = os.path.abspath(os.path.join(logging_path, "results_.npy"))
        
        # logger.info(f"Reading config_file: {config_file}")
        # # stored_cfg = tsl.Config.from_config_file(config_file)
        # stored_cfg = OmegaConf.load(config_file)
        # stored_cfg.ckpt = cfg.ckpt
        # stored_cfg.az_analysis = cfg.az_analysis
        # # use_mask_ = cfg.get("use_mask", True)
        # cfg = stored_cfg
        # cfg.neptune.online = False
        # # cfg.use_mask = use_mask_
        # for k, v in cfg.items():
        #     logger.info(f"{k:25s}: {v}")

    ########################################
    # data module                          #
    ########################################
    dataset = get_dataset(cfg.dataset)

    covariates = dict()
    if cfg.get('add_exogenous'):
        assert not isinstance(dataset, LocalGlobalGPVARDataset)
        # encode time of the day and use it as exogenous variable
        day_sin_cos = dataset.datetime_encoded('day').values
        weekdays = dataset.datetime_onehot('weekday').values
        covariates.update(u=np.concatenate([day_sin_cos, weekdays], axis=-1))

    torch_dataset = SpatioTemporalDataset(target=dataset.dataframe(),
                                          mask=dataset.mask,
                                          covariates=covariates,
                                          horizon=cfg.horizon,
                                          window=cfg.window,
                                          stride=cfg.stride)
    if cfg.get('mask_as_exog', False) and 'u' in torch_dataset:
        torch_dataset.update_input_map(u=['u', 'mask'])

    scale_axis = (0,) if cfg.get('scale_axis') == 'node' else (0, 1)
    transform = {
        'target': StandardScaler(axis=scale_axis)
    }

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
                   'mre': torch_metrics.MaskedMRE(),
                   'mse': torch_metrics.MaskedMSE()}

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
        scheduler_class=scheduler_class,
        scheduler_kwargs=scheduler_kwargs,
        scale_target=cfg.scale_target,
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
        exp_logger = NeptuneLogger(api_key=lib_config["neptune_token"],
                                project_name=lib_config["neptune_project"],
                                experiment_name=cfg.run.name,
                                tags=tags,
                                params=OmegaConf.to_object(cfg),
                                debug=not cfg.neptune.online,
                                upload_stdout=False,
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
    trainer.test(predictor, dataloaders=dm.test_dataloader())

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
    if cfg.dataset.name == "pvwest":
        time_index = dm.torch_dataset.index[dm.test_slice.numpy()[cfg.window:-cfg.horizon+1]]
    else:
        time_index = None

 
    print("residuals shape:", res.shape)
    # max_ts = 2000
    # if res.shape[0] > max_ts:
    #     res = torch.cat([res[:max_ts//2], res[-max_ts//2:]], axis=0)
    #     if m is not None:
    #         m = torch.cat([m[:max_ts//2], m[-max_ts//2:]], axis=0)
    #     if time_index is not None:
    #         time_index = time_index[:max_ts//2].union(time_index[-max_ts//2:])
 
    """
    T_, N_ = 400, 10

    # plt.plot(y[:T_, 0, N_, 0], label="y")
    # plt.plot(y_hat[:T_, 0, N_, 0], label="y_hat")
    # plt.plot(res[:T_, N_, 0], label="res")

    for n_ in range(4):
        plt.plot(res[:T_, n_, 0], label=f"Node {n_}", color=plt.cm.Set1.colors[n_])
        # plt.plot(y[:T_, 0, n_, 0], label=f"Node {n_}", color=plt.cm.Set1.colors[n_], linestyle="dashed")
        # plt.plot(y[:T_, 0, n_, 0], label=f"Node {n_}", color=plt.cm.Set1.colors[n_], linestyle="dotted")
        
    plt.legend()
    plt.grid()
    plt.savefig("tmp2.pdf")
    plt.close()
    """

    ei, ew = dataset.get_connectivity(layout="edge_index")
    graph = dict(edge_index=ei, edge_weight=ew)

    logger.info(" --- all components ---------------------")
    log_metric_ = None if not isinstance(exp_logger, NeptuneLogger) else lambda n, v: exp_logger.log_metric(metric_name=n, metric_value=v)
    # optimality_check(residuals=res, mask=m, multivariate=cfg.az_analysis.multivariate, downsample=cfg.az_analysis.downsample, remove_median=True, 
    #                  logger_msg=logger, logger_metric=log_metric_,
    #                  **graph)
    figname = f"{cfg.dataset.name}_{cfg.model.name}_{cfg.embedding.method}"
    if cfg.az_analysis.use_mask:
        figname += "_masked"
    # az_score_plot(residuals=res[..., :1], mask=m[..., :1], use_mask=cfg.use_mask,
    #               multivariate=multivariate,
    #               savefig=figname,
    #               node_order="fit",
    #               **graph,
    #               plot_window=True,
    #               plot_spacetime_scores=True,
    #               time_filter=10,
    #               k_smooth_st=5,
    #               # plot_dataset=True,
    #               # node_set=list(range(60, 100)),
    #               # time_set=list(range(1550, 1900)))
    #               node_set=list(range(30, 70)),
    #               time_set=list(range(1400, 1900)))
    # az_score_plot(residuals=res[..., :1], mask=m[..., :1], #use_mask=cfg.use_mask,
    #               **graph,
    #               savefig=os.path.join(cfg.ckpt if cfg.ckpt else cfg.run.dir, figname),
    #               **cfg.az_analysis)
    adjusted_scores = cfg.az_analysis.get("adjusted_scores", True)
    if adjusted_scores:
        OmegaConf.update(cfg, "az_analysis.adjusted_scores", True, force_add=True)
        az_scores = save_az_scores(
            residuals=res, mask=m, **graph, **cfg.az_analysis,
            # use_mask=cfg.az_analysis.use_mask, multivariate=cfg.az_analysis.multivariate,
            savepath=cfg.ckpt if cfg.ckpt else cfg.run.dir, savefig=figname + "_adjusted",
            time_index=time_index,
            log=logger.info,
            # node_set=list(range(30, 70)),
            # time_set=list(range(1400, 1900)),
            # figure_parameters=dict(main_grid=dict(figsize=[4.9, 3.8]))
        )
        log_metric_("residuals/az-test-stat[T]-adj", az_scores["glob_0.0"])
        log_metric_("residuals/az-test-stat[ST]-adj", az_scores["glob_0.5"])
        log_metric_("residuals/az-test-stat[S]-adj", az_scores["glob_1.0"])

    OmegaConf.update(cfg, "az_analysis.adjusted_scores", False, force_add=True)
    az_scores = save_az_scores(
        residuals=res, mask=m, **graph, **cfg.az_analysis,
        # use_mask=cfg.az_analysis.use_mask, multivariate=cfg.az_analysis.multivariate,
        savepath=cfg.ckpt if cfg.ckpt else cfg.run.dir, savefig=figname,
        time_index=time_index,
        log=logger.info,
        # node_set=list(range(30, 70)),
        # time_set=list(range(1400, 1900)),
        # figure_parameters=dict(main_grid=dict(figsize=[4.9, 3.8]))
    )
    log_metric_("residuals/az-test-stat[T]", az_scores["glob_0.0"])
    log_metric_("residuals/az-test-stat[ST]", az_scores["glob_0.5"])
    log_metric_("residuals/az-test-stat[S]", az_scores["glob_1.0"])

    # logger.info(" --- first component --------------------")
    # optimality_check(residuals=res[..., :1], mask=m[..., :1], remove_median=True, 
    #                  logger_msg=logger, logger_metric=log_metric_,
    #                  **graph)
    
    # logger.info(" --- last component --------------------")
    # optimality_check(residuals=res[..., -1:], mask=m[..., -1:], remove_median=True, 
    #                  logger_msg=logger, logger_metric=log_metric_,
    #                  **graph)

    exp_logger.finalize('success')


if __name__ == '__main__':
    exp = Experiment(run_fn=run_traffic, config_path='../config/static',
                     config_name='default')
    res = exp.run()
    logger.info(res)
