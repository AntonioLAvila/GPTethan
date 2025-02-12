import torch
from torch.utils.data import Dataset

class Tokenizer:
    def __init__(self):
        self.word_to_id = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        self.id_to_word = {v: k for k, v in self.word_to_id.items()}
    
    def fit(self, messages: list[str]):
        for msg in messages:
            for word in msg:
                if word not in self.word_to_id:
                    i = len(self.word_to_id)
                    self.id_to_word[i] = word
                    self.word_to_id[word] = i
    
    def encode(self, message: str):
        return [self.word_to_id[word] for word in message]
    
    def decode(self, tokens: str):
        return [self.id_to_word[wid] for wid in tokens]
    

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