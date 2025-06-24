from torch.utils.data import Dataset
import torch
from tokenizers import Tokenizer
from tokenizers.trainers import BpeTrainer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.normalizers import Lowercase, NFD, StripAccents, Sequence
from utils.constants import max_len


class MessageData():
    def __init__(self, sender_id: int, channel_id: int, text: str, timestamp: int):
        self.sender_id = sender_id
        self.channel_id = channel_id
        self.text = text
        self.timestamp = timestamp


class ChatDataset(Dataset):
    def __init__(self, data_dir: str, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.samples = load_dataset(data_dir)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        prompt, response = self.samples[idx]
        full_input = f"{prompt} [EOS] {response}"
        encoded = self.tokenizer.encode(full_input)

        ids = encoded.ids[:max_len]
        pad_len = max_len - len(ids)
        input_ids = ids + [self.tokenizer.token_to_id("[PAD]")] * pad_len

        x = torch.tensor(input_ids[:-1])  # model input
        y = torch.tensor(input_ids[1:])  # target (next token prediction)
        return x, y
    

def load_dataset(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines()]
    samples = [(lines[i], lines[i+1]) for i in range(0, len(lines), 3)]
    return samples


def build_tokeizer(data_dir):
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"])
    samples = load_dataset(data_dir)
    texts = [prompt for prompt, _ in samples] + [response for _, response in samples]

    tokenizer.train_from_iterator(texts, trainer=trainer)

    tokenizer.post_processor = TemplateProcessing(
        single="[BOS] $A [EOS]",
        pair="[BOS] $A [EOS] $B:1 [EOS]:1",
        special_tokens=[("[BOS]", tokenizer.token_to_id("[BOS]")),
                        ("[EOS]", tokenizer.token_to_id("[EOS]"))]
    )

    tokenizer.save("tokenizer")
    return tokenizer


def build_dataset_and_tokenizer(data_dir):
    tokenizer = build_tokeizer(data_dir)
    dataset = ChatDataset(data_dir, tokenizer)
    return dataset, tokenizer


# if __name__ == "__main__":
#     from constants import data_dir
#     with open(data_dir, 'r') as f:
#             for i, line in enumerate(f):
#                 if i == 0:
#                     if line[-1] == '\n':
#                         print('newline')


# class Tokenizer:
#     def __init__(self):
#         self.bidict_word_id = bidict()
#         self.bidict_word_id["<pad>"] = 0
#         self.bidict_word_id["<sos>"] = 1
#         self.bidict_word_id["<eos>"] = 2

#         self.max_seq = -1
    
#     def add_token(self, word):
#         if word not in self.bidict_word_id:
#             wid = len(self.bidict_word_id)
#             self.bidict_word_id[word] = wid
    
#     def encode(self, message: str):
#         ret = [1]

#         # replace links
#         tokens = re.sub(r'https?://\S+', lambda match: self.bidict_word_id[match.group(0)], message)

#         for i in range(self.max_seq):
#             if i < len(tokens):
#                 ret.append(tokens[i])
#             else:
#                 ret.append(0)
#         ret.append(2)
#         return ret
    
#     def decode(self, tokens: list[int]):
#         return [self.bidict_word_id.inv[wid] for wid in tokens]
    
#     def add_tokens_from_message(self, msg: str, user_map: dict):
#         length = 0

#         # handle links
#         links = re.findall(r'https?://\S+', msg)
#         for link in links:
#             self.add_token(link)
#             length += 1

#         msg = re.sub(r'https?://\S+', '', msg) # remove them

#         msg = msg.lower()

#         if re.search(r'\d{18}', msg): # replace user ids with their names
#             msg = re.sub(r'\b(' + '|'.join(map(re.escape, user_map.keys())) + r')\b', lambda m: user_map[m.group(0)], msg)

#         stripped = re.sub(r'[^a-zA-Z\s1-9]+', '', msg) # strip output

#         tokens = stripped.split()

#         for i in tokens:
#             self.add_token(i)
#             length += 1

#         if self.max_seq < length+2:
#             self.max_seq = length+2


# def build_tokenizer(data):
#     tokenizer = Tokenizer()
#     with open(data, "r") as f:
#         for prompt, response, _ in zip(f, f, f):
#             tokenizer.add_tokens_from_message(prompt)
#             tokenizer.add_tokens_from_message(response)
#     return tokenizer
