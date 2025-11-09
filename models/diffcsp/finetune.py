import torch
from torch.utils.data import Dataset


class DiffCSPDataset(Dataset):
    def __init__(self, data_list, rewards=None):
        super().__init__()
        self.data_list = data_list
        if rewards is not None:
            rewards = torch.tensor(rewards, dtype=torch.float)
            for i, data in enumerate(self.data_list):
                data.reward = rewards[i].unsqueeze(dim=0)

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]
