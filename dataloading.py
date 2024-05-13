"""Dataset for electricity consumption forecasting with neural networks."""


from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


N_INPUT_HOUR = 24


class ElectricTimeSeries(Dataset):
    """Timeseries dataset class for electricity consumption prediction."""

    def __init__(self, data: pd.DataFrame, feats: list, target: str):
        super().__init__()
        self.data = data
        self.feats = feats
        self.target = target
        self.feat_mean = torch.tensor([0] * N_INPUT_HOUR)
        self.feat_std = torch.tensor([1] * N_INPUT_HOUR)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: int) -> Union[None,
                                           Tuple[torch.Tensor, torch.Tensor]]:
        X = self.data[self.feats].iloc[i: i + N_INPUT_HOUR]
        y = self.data[self.target].iloc[i + N_INPUT_HOUR:
                                        i + N_INPUT_HOUR + 24].sum()
        if X.isna().any(axis=None) or np.isnan(y) \
                or X.shape[0] != N_INPUT_HOUR:
            return None
        # Tensors
        X = torch.from_numpy(X.values.astype('float32'))
        y = torch.tensor([y.astype('float32')])
        # Normalize
        X -= self.feat_mean.unsqueeze(0).repeat((N_INPUT_HOUR, 1))
        X /= self.feat_std.unsqueeze(0).repeat((N_INPUT_HOUR, 1))
        return X, y

    def infer_mean_and_std(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute per-column mean and std from dataframe."""
        feat_mean = torch.Tensor(self.data[self.feats].mean().values
                                 .astype('float32'))
        feat_std = torch.Tensor(self.data[self.feats].std().values
                                .astype('float32'))
        return feat_mean, feat_std

    def set_mean_and_std(self, feat_mean: torch.Tensor,
                         feat_std: torch.Tensor):
        """Set dataframe's column mean and std attributes for normalization."""
        self.feat_mean = feat_mean
        self.feat_std = feat_std


def collate_fn(batch: List[Union[None, Tuple[torch.Tensor, torch.Tensor]]]
               ) -> Union[None, List[Tuple[torch.Tensor, torch.Tensor]]]:
    """Remove None items from batch."""
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)
