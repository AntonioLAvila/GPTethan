from torch.utils.data import Dataset
import torch
import re
import bidict
from tokenizers import Tokenizer
from tokenizers.models import BPE


class MessageData():
    def __init__(self, sender_id: int, channel_id: int, text: str, timestamp: int):
        self.sender_id = sender_id
        self.channel_id = channel_id
        self.text = text
        self.timestamp = timestamp


class ChatDataset(Dataset):
    def __init__(self, data_dir: str, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        # read file into memory
        with open(data_dir, 'r') as f:
            self.file = f.readlines()

    def __len__(self):
        return len(self.file) / 3

    def __getitem__(self, idx):
        line_idx = idx*3

        prompt = self.file[line_idx]
        response = self.file[line_idx+1]

        encoded_prompt = self.tokenizer.encode(prompt)
        encoded_response = self.tokenizer.encode(response)

        return torch.tensor(encoded_prompt), torch.tensor(encoded_response)
    

class Tokenizer:
    def __init__(self):
        self.bidict_word_id = bidict()
        self.bidict_word_id["<pad>"] = 0
        self.bidict_word_id["<sos>"] = 1
        self.bidict_word_id["<eos>"] = 2

        self.max_seq = -1
    
    def add_token(self, word):
        if word not in self.bidict_word_id:
            wid = len(self.bidict_word_id)
            self.bidict_word_id[word] = wid
    
    def encode(self, message: str):
        ret = [1]

        # replace links
        tokens = re.sub(r'https?://\S+', lambda match: self.bidict_word_id[match.group(0)], message)

        for i in range(self.max_seq):
            if i < len(tokens):
                ret.append(tokens[i])
            else:
                ret.append(0)
        ret.append(2)
        return ret
    
    def decode(self, tokens: list[int]):
        return [self.bidict_word_id.inv[wid] for wid in tokens]
    
    def add_tokens_from_message(self, msg: str, user_map: dict):
        length = 0

        # handle links
        links = re.findall(r'https?://\S+', msg)
        for link in links:
            self.add_token(link)
            length += 1

        msg = re.sub(r'https?://\S+', '', msg) # remove them

        msg = msg.lower()

        if re.search(r'\d{18}', msg): # replace user ids with their names
            msg = re.sub(r'\b(' + '|'.join(map(re.escape, user_map.keys())) + r')\b', lambda m: user_map[m.group(0)], msg)

        stripped = re.sub(r'[^a-zA-Z\s1-9]+', '', msg) # strip output

        tokens = stripped.split()

        for i in tokens:
            self.add_token(i)
            length += 1

        if self.max_seq < length+2:
            self.max_seq = length+2


def build_tokenizer(data):
    tokenizer = Tokenizer()
    with open(data, "r") as f:
        for prompt, response, _ in zip(f, f, f):
            tokenizer.add_tokens_from_message(prompt)
            tokenizer.add_tokens_from_message(response)
    return tokenizer

