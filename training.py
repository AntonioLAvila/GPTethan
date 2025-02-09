import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from GPTethan import Transformer
import random
import pickle

# Load data
with open("ethan_msgs_rep.pkl", "rb") as f:
    ethan_msgs_rep = pickle.load(f)
with open("id_to_word.pkl", "rb") as f:
    id_to_word = pickle.load(f)
with open("word_to_id.pkl", "rb") as f:
    word_to_id = pickle.load(f)

# Even length
if len(ethan_msgs_rep) % 2 != 0:
    ethan_msgs_rep = ethan_msgs_rep[:-1]

# Shuffle data
indices = list(range(0, len(ethan_msgs_rep)))
random.shuffle(indices)
train_data = [ethan_msgs_rep[i] for i in indices]

# Split data
src_data = train_data[:len(train_data)//2] # torch.randint(1, src_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)
tgt_data = train_data[len(train_data)//2:] # torch.randint(1, tgt_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)

src_vocab_size = len(id_to_word) # 5000
tgt_vocab_size = len(id_to_word) # 5000
d_model = 512 # 512
num_heads = 8 # 8
num_layers = 6
d_ff = 1024 # 2048
max_seq_length = len(ethan_msgs_rep[0])# 100
dropout = 0.1
batch_size = 8

print(f"{src_vocab_size=}, {max_seq_length=}, {len(ethan_msgs_rep)=}")

class TextDataset(Dataset):
    def __init__(self, src, tgt):
        self.src = src
        self.tgt = tgt

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        src_tensor = torch.tensor(self.src[idx], dtype=torch.long)
        tgt_tensor = torch.tensor(self.tgt[idx], dtype=torch.long)
        return src_tensor, tgt_tensor

train_dataset = TextDataset(src_data, tgt_data)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout).to('cuda')

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

transformer.train()
print("Starting train loop")

for epoch in range(100):
    total_loss = 0
    for src_batch, tgt_batch in train_loader:
        src_batch = src_batch.to('cuda')
        tgt_batch = tgt_batch.to('cuda')

        optimizer.zero_grad()
        output = transformer(src_batch, tgt_batch[:, :-1])
        loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_batch[:, 1:].contiguous().view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")
    torch.cuda.empty_cache()
    # optimizer.zero_grad()
    # output = transformer(src_data, tgt_data[:, :-1])
    # loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))
    # loss.backward()
    # optimizer.step()
    # print(f"Epoch: {epoch+1}, Loss: {loss.item()}")



transformer.eval()


# we dont do that here
# val_src_data = torch.tensor(val_data[:len(val_data)//2]) # torch.randint(1, src_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)
# val_tgt_data = torch.tensor(val_data[len(val_data)//2:]) # torch.randint(1, tgt_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)

# with torch.no_grad():

#     val_output = transformer(val_src_data, val_tgt_data[:, :-1])
#     print(val_output)
#     val_loss = criterion(val_output.contiguous().view(-1, tgt_vocab_size), val_tgt_data[:, 1:].contiguous().view(-1))
#     print(f"Validation Loss: {val_loss.item()}")
