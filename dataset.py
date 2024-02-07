import os

import numpy as np
import pandas as pd
import torch as th
from torch.utils.data import DataLoader, Dataset

from config import ConfigFactory
from utils import DataNormalizer
from tqdm import tqdm


class AQIDataset(Dataset):
    def __init__(self, data, time_index, stat_locations, stat_mask_ratio, k_top, window) -> None:
        super(AQIDataset, self).__init__()
        self.time_index = time_index
        self.data = data
        self.stat_num = self.data.shape[2]

        self.stat_locations = stat_locations
        self.dist_matrix = np.sum(
            (stat_locations[:, np.newaxis, :] -
             stat_locations[np.newaxis, :, :]) ** 2,
            axis=-1,
        )
        np.fill_diagonal(self.dist_matrix, np.inf)

        self.stat_mask_ratio = stat_mask_ratio
        self.k_top = k_top
        self.window = window

    @staticmethod
    def flat_window_ngb_dim(x: np.ndarray):
        assert x.ndim == 4
        b, w, ngb, f = x.shape
        new_shape = (b, w * ngb, f)
        reshaped_x = x.reshape(new_shape)
        return reshaped_x

    def __getitem__(self, index):
        sample_data = self.data[index]
        time_index = self.time_index[index]

        nan_mask = np.isnan(sample_data[:, :, -1])
        temp_dist_mtx = np.expand_dims(
            self.dist_matrix, 0).repeat(self.window + 1, 0)
        temp_dist_mtx[np.expand_dims(nan_mask, 1).repeat(
            self.stat_num, 1)] = np.inf

        used_stat_mask = ~nan_mask[0]
        mask_index = np.random.choice(
            np.arange(self.stat_num)[used_stat_mask],
            int(self.stat_mask_ratio * used_stat_mask.sum()),
            replace=False,
        )
        temp_dist_mtx[:, :, mask_index] = np.inf
        ngb_index = np.argsort(temp_dist_mtx, axis=2)[:, :, : self.k_top]

        # ngb info
        ngb_feat = sample_data[np.arange(sample_data.shape[0])[
            :, None, None], ngb_index, :]
        ngb_xy = self.stat_locations[ngb_index]
        ngb_time = np.tile(
            np.expand_dims(time_index, axis=(1, 2, 3)),
            (1, self.stat_num, self.k_top, 1),
        )

        ngb_data = np.concatenate([ngb_xy, ngb_time, ngb_feat], axis=-1).transpose(
            1, 0, 2, 3
        )[used_stat_mask]
        ngb_data = self.flat_window_ngb_dim(ngb_data)

        # mask bool
        mask_index_bool = np.zeros(self.stat_num).astype(np.bool_)
        mask_index_bool[mask_index] = True
        mask_index_bool = mask_index_bool[used_stat_mask]

        # stat
        stat_feat = sample_data[0, used_stat_mask, :]
        stat_xy = self.stat_locations[used_stat_mask]
        stat_time = np.tile(
            np.expand_dims(time_index[0], axis=(
                0, 1)), (used_stat_mask.sum(), 1)
        )
        stat_data = np.concatenate(
            [stat_xy, stat_time, stat_feat],
            axis=-1,
        )

        # data_index
        data_index = np.where(used_stat_mask)[0]

        # dist
        dist_mtx = np.take_along_axis(temp_dist_mtx, ngb_index, axis=2).transpose(
            1, 0, 2
        )[used_stat_mask]
        return [
            th.from_numpy(arr)
            for arr in (stat_data, ngb_data, mask_index_bool, data_index, dist_mtx)
        ]

    def __len__(self):
        return len(self.data)


def col_func(batch):
    keys = ("stat", "ngb", "mask_index", "data_index", "dist")
    values = (th.concat(i, dim=0) for i in zip(*batch))
    return dict(zip(keys, values))


def get_air_loader_normalizer(cfg):
    which_data = cfg.which_data
    window = cfg.window
    threshold_ratio = cfg.drop_limit

    th.multiprocessing.set_start_method("spawn")
    loc_path, data_path = {
        "tiny": ("36stat_loc_norm.csv", "36air.npy"),
        "large": ("1085stat_loc_norm.csv", "air_2018.npy"),
    }[which_data]
    stat_locations = pd.read_csv(
        os.path.join(cfg.data_folder, loc_path)).values
    raw_data = np.load(os.path.join(
        cfg.data_folder, data_path))

    if which_data == "large":
        select_stat = np.ones(raw_data.shape[1], dtype=np.bool_)
        select_stat[[671, 675]] = False
        stat_locations = stat_locations[select_stat]
        raw_data = raw_data[:, select_stat, :]
    # move pm2.5 to the last column for the sake of convenience
    label_column = {
        "pm25":0,
        "pm10":1,
        "no2":2,
        "co":3,
        "o3":4,
        "so2":5,
        "rain":6,
        "temp":7,
        "press":8
    }[cfg.label]
    label = raw_data[:, :, label_column]
    real_time = np.arange(len(raw_data))

    # delete too much nan values rows
    nan_ratio = np.isnan(label).sum(axis=1) / label.shape[1]
    select_index = nan_ratio <= threshold_ratio
    label = label[select_index, :]
    data = raw_data[select_index, :, :]
    real_time = real_time[select_index]

    # normalization
    dm = DataNormalizer()
    dm.fit(label)
    label = dm.norm(label)
    feat = []
    for i in tqdm(range(data.shape[2]),desc='Normalizing Features...'):
        # 6 air + 5 continual weather + 5 discrete weather
        if i == label_column:
            continue
        if i < 11:
            data_slice = data[:, :, i]
            feat.append(np.nan_to_num(
                dm.fit_norm(data_slice), nan=0))
        else:
            data_slice = data[:, :, i]+1
            feat.append(np.nan_to_num(
                data_slice, nan=0))
    data_norm = np.stack(feat+[label]).transpose(1, 2, 0)

    # train and test data
    length = len(data)
    train_index = np.arange(length)[:int(0.6 * length)]
    val_index = np.arange(length)[int(0.6 * length): int(0.8 * length)]
    test_index = np.arange(length)[int(0.8 * length):]

    assert window >= 0

    def generate_time_data(index, reverse=True):
        flag = -1 if reverse else 1
        data = np.stack(
            [data_norm[index + flag * i * cfg.skip_step, :, :] for i in range(window + 1)]
        ).transpose(1, 0, 2, 3)
        time_index = np.stack([real_time[index + flag * i]
                               for i in range(window + 1)]).transpose(1, 0)
        if not reverse:
            data = np.flip(data, axis=1)
            time_index = np.flip(time_index, axis=1)
        return data, time_index
    train_data, train_time_index = generate_time_data(
        train_index, reverse=False)
    val_data, val_time_index = generate_time_data(val_index)
    test_data, test_time_index = generate_time_data(test_index)

    dataloaders = [
        DataLoader(
            AQIDataset(
                data,
                time_index,
                stat_locations,
                stat_mask_ratio=cfg.mask_ratio,
                k_top=cfg.k_top,
                window=window,
            ),
            batch_size=cfg.batch_size,
            shuffle=True,
            collate_fn=col_func,
            num_workers=cfg.num_workers,
            prefetch_factor=cfg.prefetch_factor,
            pin_memory=True,
        )
        for (data, time_index) in [
            (train_data, train_time_index),
            (val_data, val_time_index),
            (test_data, test_time_index)
        ]
    ]
    print("Finish Loaders Preparing!")
    return dataloaders, dm


if __name__ == "__main__":
    args, msg = ConfigFactory.build()
    print(msg)
    (train_loader, val_loader, test_loader), _ = get_air_loader_normalizer(args)
    batch_data = next(iter(train_loader))
    print(
        pd.DataFrame.from_dict(
            {k: str(tuple(v.shape)) for k, v in batch_data.items()},
            orient="index",
            columns=["shape"],
        )
    )
    print("Finish!")
