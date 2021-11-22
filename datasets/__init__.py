import numpy as np
from torch.utils.data import Dataset

class simpleData(Dataset):
    def __init__(self, df, idx, x_col, y_col):
        super().__init__()
        self.idx = idx

        self.feature   = df[x_col].values.astype(np.float32).reshape(-1, 80, len(x_col))
        self.pressure  = df[y_col].values.astype(np.float32).reshape(-1,80)
        self.u_out     = df['u_out'].values.astype(np.float32).reshape(-1,80)
        # self.breath_id = df['breath_id'].values[::80]

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, index):
        i = self.idx[index]
        u_out_i= self.u_out[i]
        pressure_i = self.pressure[i]
        feature_i = self.feature[i]	

        return feature_i, u_out_i, pressure_i
