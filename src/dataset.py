import numpy as np
import torch
from torch.utils.data import Dataset


class HandwritingDataset(Dataset):
    def __init__(self, npz_path, max_seq_len=12000, normalize=True, cache_prepared=True):
        data = np.load(npz_path, allow_pickle=True)
        self.points = data["points"]
        self.text_indices = data["text_indices"]
        self.max_seq_len = max_seq_len
        self.normalize = normalize
        self.cache_prepared = cache_prepared

        if self.points.dtype == np.object_:
            self.points = [np.asarray(p, dtype=np.float32) for p in self.points]
            self.text_indices = [np.asarray(t, dtype=np.int64).reshape(-1) for t in self.text_indices]
        else:
            self.points = [np.asarray(self.points, dtype=np.float32)]
            self.text_indices = [np.asarray(self.text_indices, dtype=np.int64).reshape(-1)]

        if len(self.points) != len(self.text_indices):
            raise ValueError("points and text_indices must contain the same number of samples")

        if self.normalize:
            all_dxdy = []
            for pts in self.points:
                if len(pts) == 0:
                    continue
                xy = pts[:, :2]
                dx = np.diff(xy[:, 0], prepend=xy[0, 0])
                dy = np.diff(xy[:, 1], prepend=xy[0, 1])
                all_dxdy.append(np.stack([dx, dy], axis=1))

            if all_dxdy:
                all_dxdy = np.concatenate(all_dxdy, axis=0).astype(np.float32)
                self.dxdy_mean = all_dxdy.mean(axis=0, keepdims=True)
                self.dxdy_std = all_dxdy.std(axis=0, keepdims=True) + 1e-6
            else:
                self.dxdy_mean = np.zeros((1, 2), dtype=np.float32)
                self.dxdy_std = np.ones((1, 2), dtype=np.float32)
        else:
            self.dxdy_mean = np.zeros((1, 2), dtype=np.float32)
            self.dxdy_std = np.ones((1, 2), dtype=np.float32)

        self.prepared_samples = None
        if self.cache_prepared:
            self.prepared_samples = [self._prepare_sample(idx) for idx in range(len(self.points))]

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        if self.prepared_samples is not None:
            return self.prepared_samples[idx]

        return self._prepare_sample(idx)

    def _prepare_sample(self, idx):
        pts = self.points[idx].copy()
        if pts.ndim != 2 or pts.shape[1] < 3:
            raise ValueError(f"sample {idx} must have shape (T, 3)")

        if pts.shape[0] > self.max_seq_len:
            pts = pts[: self.max_seq_len]

        xy = pts[:, :2].astype(np.float32)
        dx = np.diff(xy[:, 0], prepend=xy[0, 0])
        dy = np.diff(xy[:, 1], prepend=xy[0, 1])
        dxy = np.stack([dx, dy], axis=1).astype(np.float32)

        pen = pts[:, 2].astype(np.int64)
        e = (pen == 1).astype(np.float32)
        e[-1] = 1.0

        if self.normalize:
            dxy = (dxy - self.dxdy_mean) / self.dxdy_std

        text_arr = self.text_indices[idx].copy().reshape(-1)

        return {
            "dxy": torch.tensor(dxy, dtype=torch.float32),
            "e": torch.tensor(e, dtype=torch.float32).unsqueeze(1),
            "text": torch.tensor(text_arr, dtype=torch.long),
            "length": len(pts),
            "text_lengths": len(text_arr),
        }
