import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from GPTethan import Transformer
from util import ChatDataset
import pickle

# Load data
with open("ethan_msgs_rep.pkl", "rb") as f:
    ethan_msgs_rep = pickle.load(f)
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

src_vocab_size = tgt_vocab_size = len(tokenizer.id_to_word) # 5000
d_model = 1024
num_heads = 8
num_layers = 6
d_ff = 2048
max_seq_length = len(ethan_msgs_rep[0]) # 100
dropout = 0.1
batch_size = 16
num_epochs = 100

print(f"{src_vocab_size=}, {max_seq_length=}, {len(ethan_msgs_rep)=}")

# Create model
transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout).to('cuda')

# Create dataset and dataloader
dataset = ChatDataset(ethan_msgs_rep)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

pad_token_id = tokenizer.word_to_id['<pad>']

criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
optimizer = optim.Adam(transformer.parameters(), lr=5e-4, betas=(0.9, 0.98), eps=1e-9)

transformer.train()

# Train
print("Starting training loop")
for epoch in range(num_epochs):
    transformer.train()
    total_loss = 0

    for src, tgt in dataloader:
        src, tgt = src.to('cuda'), tgt.to('cuda')

        optimizer.zero_grad()

        output = transformer(src, tgt[:, :-1])  # Shifted target input
        loss = criterion(output.view(-1, output.shape[-1]), tgt[:, 1:].reshape(-1))  # Ignore first token

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

# Save model
torch.save(transformer.state_dict(), "model.pth")

