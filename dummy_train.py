from tqdm import tqdm
import torch
from model.gpt import GPT
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# === Synthetic dataset for next-token prediction ===
class RandomTokenDataset(Dataset):
    def __init__(self, num_samples, seq_len, vocab_size):
        self.data = torch.randint(0, vocab_size, (num_samples, seq_len + 1))

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        sample = self.data[idx]
        x = sample[:-1]  # Input sequence
        y = sample[1:]   # Target sequence (next tokens)
        return x, y


# === Train function ===
def train_gpt():
    # Hyperparameters
    vocab_size = 30000
    d_model = 510
    d_ff = 4*510
    n_layers = 6
    n_heads = 6
    dropout = 0.1
    seq_len = 32
    batch_size = 8
    num_epochs = 10000
    num_samples = 1000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset & Loader
    dataset = RandomTokenDataset(num_samples, seq_len, vocab_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model, loss, optimizer
    model = GPT(vocab_size, d_model, d_ff, n_layers, n_heads, dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    # Training loop
    model.train()
    for epoch in tqdm(range(num_epochs)):
        total_loss = 0
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)  # (B, T, vocab)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")

    print("Training complete.")

if __name__ == "__main__":
    train_gpt()
