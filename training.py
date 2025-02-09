import torch
import torch.nn as nn
import torch.optim as optim
from GPTethan import Transformer
import random
import pickle

with open("ethan_msgs_rep.pkl", "rb") as f:
    ethan_msgs_rep = pickle.load(f)
with open("ethan_msgs.pkl", "rb") as f:
    ethan_msgs = pickle.load(f)
with open("id_to_word.pkl", "rb") as f:
    id_to_word = pickle.load(f)
with open("word_to_id.pkl", "rb") as f:
    word_to_id = pickle.load(f)

if len(ethan_msgs_rep) % 2 != 0: # make sure even
    ethan_msgs_rep = ethan_msgs_rep[:-1]

src_vocab_size = len(id_to_word) # 5000
tgt_vocab_size = len(id_to_word) # 5000
d_model = 1024 # 512
num_heads = 8
num_layers = 6
d_ff = 4096 # 2048
max_seq_length = len(ethan_msgs[0])# 100
dropout = 0.1

print(f"{src_vocab_size=}, {max_seq_length=}")

indices = list(range(0, len(ethan_msgs_rep)))
random.shuffle(indices)

train_data = [ethan_msgs_rep[i] for i in indices]

src_data = torch.tensor(train_data[:len(train_data)//2]).to('cuda') # torch.randint(1, src_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)
tgt_data = torch.tensor(train_data[len(train_data)//2:]).to('cuda') # torch.randint(1, tgt_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)


transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout).to('cuda')

print(f"{src_data.shape=}, {tgt_data.shape=}")

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

transformer.train()

print("Starting train loop")

for epoch in range(100):
    optimizer.zero_grad()
    output = transformer(src_data, tgt_data[:, :-1])
    loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")



transformer.eval()


# we dont do that here
# val_src_data = torch.tensor(val_data[:len(val_data)//2]) # torch.randint(1, src_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)
# val_tgt_data = torch.tensor(val_data[len(val_data)//2:]) # torch.randint(1, tgt_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)

# with torch.no_grad():

#     val_output = transformer(val_src_data, val_tgt_data[:, :-1])
#     print(val_output)
#     val_loss = criterion(val_output.contiguous().view(-1, tgt_vocab_size), val_tgt_data[:, 1:].contiguous().view(-1))
#     print(f"Validation Loss: {val_loss.item()}")
