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

            fpz_cz = npz_data['eeg_fpz_cz']
            pz_oz = npz_data['eeg_pz_oz']
            annotation = npz_data['annotation']
            fpz_cz = np.expand_dims(fpz_cz, 1)
            pz_oz = np.expand_dims(pz_oz, 1)
            signals = np.concatenate([fpz_cz, pz_oz], axis=1)

            data.append(signals)
            label.append(annotation)

        data = np.concatenate(data, axis=0)
        label = np.concatenate(label, axis=0)
        self.data = torch.from_numpy(data).float()
        self.label = torch.from_numpy(label).long()

    def __getitem__(self, item):
        return self.data[item], self.label[item]

    def __len__(self):
        return self.data.shape[0]
