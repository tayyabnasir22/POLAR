from torch.utils.data import Dataset

class DataProcessorBase(Dataset):
    # Must return the rgb, lr, gt, min, max
    # Get scale, base directory, agumentation, 

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass
