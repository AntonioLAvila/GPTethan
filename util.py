import torch
from torch.utils.data import Dataset

class ChatDataset(Dataset):
    def __init__(self, tokenized_msgs):
        self.data = tokenized_msgs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        msg = self.data[idx]

        src = msg[:-1]  # Input (exclude <eos>)
        tgt = msg[1:]   # Target (exclude <sos>)

        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)