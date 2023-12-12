import numpy as np
from tsl.datasets.air_quality import infer_mask, AirQuality as AirQuality_


class AirQuality(AirQuality_):

    def load(self, impute_nans=True):
        # load readings and stations metadata
        df, dist, eval_mask = self.load_raw()
        # compute the masks:
        mask = (~np.isnan(df.values)).astype('uint8')  # 1 if value is valid
        if eval_mask is None:
            eval_mask = infer_mask(df, infer_from=self.infer_eval_from)
        # 1 if value is ground-truth for imputation
        eval_mask = eval_mask.values.astype('uint8')
        if len(self.masked_sensors):
            eval_mask[:, self.masked_sensors] = mask[:, self.masked_sensors]
        # eventually replace nans with weekly mean by hour
        if impute_nans:
            df = df.fillna(method='ffill')
            df = df.fillna(0.)
        return df, mask, eval_mask, dist
