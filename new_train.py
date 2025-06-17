import torch
from model.gpt import GPT
from utils.util import build_dataset_and_tokenizer
from utils.constants import data_dir
from torch.utils.data import DataLoader
from tqdm import tqdm

def train():
    dataset, tokenizer = build_dataset_and_tokenizer(data_dir)

    batch_size = 8
    epochs = 10000
    lr = 3e-4
    d_model = 768
    d_ff = 4 * d_model
    n_layer = 6
    n_head = 6
    dropout = 0.1

    # Pad/truncate sequences in batch with collate_fn
    pad_id = tokenizer.token_to_id("[PAD]")
    def collate_fn(batch):
        inputs, targets, masks = zip(*batch)
        inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=pad_id)
        targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=pad_id)
        masks = torch.nn.utils.rnn.pad_sequence(masks, batch_first=True, padding_value=0)
        return inputs, targets, masks


    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    model = GPT(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=d_model,
        d_ff=d_ff,
        n_layers=n_layer,
        n_heads=n_head,
        dropout=dropout
    ).cuda()

    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id("[PAD]"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for epoch in tqdm(range(epochs)):
        total_loss = 0
        for x, y, mask in train_loader:
            x, y, mask = x.cuda(), y.cuda(), mask.cuda()

            optimizer.zero_grad()
            logits = model(x)  # (B, T, vocab_size)

            # Flatten logits and targets
            logits_flat = logits.view(-1, logits.size(-1))
            y_flat = y.view(-1)
            mask_flat = mask.view(-1)

            loss_all = criterion(logits_flat, y_flat)  # loss for all tokens
            loss = (loss_all * mask_flat).sum() / mask_flat.sum()  # mask out prompt tokens

            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    train()