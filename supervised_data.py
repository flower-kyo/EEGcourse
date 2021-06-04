from torch.utils.data import Dataset
import numpy as np
import os
import torch
import math

def get_train_valid_data(data_dir, n_fold, fold_idx):
    files = os.listdir(data_dir)
    n_file_pre_fold = math.ceil(len(files) / n_fold)
    fold_st_idx = n_file_pre_fold * fold_idx
    valid_files = []
    for _fold_file_idx in range(fold_st_idx, fold_st_idx + n_file_pre_fold):
        if _fold_file_idx <= len(files) - 1:
            valid_files.append(files[_fold_file_idx])
    print("fold {}, valid set file names {}".format(fold_idx, valid_files))
    train_files = list(set(files) - set(valid_files))
    return train_files, valid_files


class SupDataset(Dataset):
    def __init__(self, data_dir, file_list, data_name):
        super(SupDataset, self).__init__()
        data = []
        label = []
        self.data = None
        self.label = None
        for file in file_list:
            file_path = os.path.join(data_dir, file)
            npz_data =np.load(file_path)

            x = npz_data['x']
            y = npz_data['y']
            x = np.expand_dims(x, 1)


            data.append(x)
            label.append(y)

        data = np.concatenate(data, axis=0)
        label = np.concatenate(label, axis=0)
        self.data = torch.from_numpy(data).float()
        self.label = torch.from_numpy(label).long()

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return self.data.shape[0]

