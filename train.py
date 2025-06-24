import torch
from model.gpt import GPT
from utils.util import build_dataset_and_tokenizer
from utils.constants import data_dir, model_out_path
from torch.utils.data import DataLoader
from tqdm import tqdm
from clearml import Task

def train():
    dataset, tokenizer = build_dataset_and_tokenizer(data_dir)
    print(f"Vocab size: {tokenizer.get_vocab_size()}")

    batch_size = 8
    epochs = 10000
    lr = 1e-5

    # Pad/truncate sequences in batch with collate_fn
    def collate_fn(batch):
        inputs, targets, masks = zip(*batch)
        inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=tokenizer.token_to_id("[PAD]"))
        targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=tokenizer.token_to_id("[PAD]"))
        masks = torch.nn.utils.rnn.pad_sequence(masks, batch_first=True, padding_value=0)
        return inputs, targets, masks

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # cml_task = Task.init(
    #     project_name="GPTethan",
    #     task_name="training",
    #     task_type=Task.TaskTypes.training
    # )

    model = GPT(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=510,
        d_ff=4*510,
        n_layers=6,
        n_heads=6,
        dropout=0.1
    ).cuda()

    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id("[PAD]"), reduction='none')
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for epoch in tqdm(range(epochs)):
        total_loss = 0
        for x, y, mask in train_loader:
            x, y, mask = x.cuda(), y.cuda(), mask.cuda()

            optimizer.zero_grad()
            logits = model(x)

            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print("NaN or Inf in logits!")
                break

            logits_flat = logits.view(-1, logits.size(-1))
            y_flat = y.view(-1)
            mask_flat = mask.view(-1)

            loss_all = criterion(logits_flat, y_flat)  # loss for all tokens
            loss = (loss_all * mask_flat).sum() / mask_flat.sum()  # mask out prompt tokens

            if torch.isnan(loss_all).any() or torch.isinf(loss_all).any():
                print("NaN or Inf in loss_all!")
                break

            total_loss += loss.item()
            # cml_task.get_logger().report_scalar(
            #     title='loss',
            #     series='training',
            #     value=loss.item(),
            #     iteration=epoch
            # )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), model_out_path)


if __name__ == "__main__":
    train()