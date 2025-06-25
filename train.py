import torch
from model.gpt import GPT
from utils.util import build_dataset_and_tokenizer, build_dataset, load_tokenizer
from utils.constants import data_dir, model_path, batch_size, lr, d_model, d_ff, n_layers, n_heads, dropout, max_len, tokenizer_path
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
from tqdm import tqdm
from clearml import Task
import atexit

def main(arg):
    try:
        tokenizer = load_tokenizer(tokenizer_path)
        dataset = build_dataset(data_dir, tokenizer)
    except:
        dataset, tokenizer = build_dataset_and_tokenizer(data_dir, tokenizer_path)
    loader = DataLoader(dataset, batch_size=batch_size)

    model = GPT(
        vocab_size=tokenizer.get_vocab_size(), # TODO assumes you used the same tokenizer/data
        d_model=d_model,
        d_ff=d_ff,
        n_layers=n_layers,
        n_heads=n_heads,
        dropout=dropout
    ).cuda()

    if arg == 'train':
        train(model, loader, tokenizer)
    else:
        model.load_state_dict(model_path)
        prompt = input("Say something: ")
        inference(model, tokenizer, prompt)
    

def train(model: torch.nn.Module, loader: DataLoader, tokenizer: Tokenizer):
    cml_task = Task.init(project_name="GPTethan", task_name="training", task_type=Task.TaskTypes.training)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id("[PAD]"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    atexit.register(lambda: torch.save(model.state_dict(), model_path))

    model.train()
    for epoch in tqdm(range(10000)):
        total_loss = 0
        for xb, yb in loader:
            xb, yb = xb.cuda(), yb.cuda()
            logits = model(xb)
            loss = criterion(logits.view(-1, logits.size(-1)), yb.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            cml_task.get_logger().report_scalar(
                title='loss',
                series='train',
                value=loss.item(),
                iteration=epoch
            )
        print(f"Epoch {epoch+1} Loss: {total_loss / len(loader):.4f}")

    # torch.save(model.state_dict(), model_path)


def inference(model: torch.nn.Module, tokenizer: Tokenizer, prompt: str):
    model.eval()
    input_ids = tokenizer.encode(prompt).ids
    input_ids = [tokenizer.token_to_id("[BOS]")] + input_ids + [tokenizer.token_to_id("[EOS]")]

    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(next(model.parameters()).device)

    for _ in range(max_len):
        logits = model(input_ids)
        next_id = logits[0, -1].argmax().item()
        input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device=input_ids.device)], dim=1)
        if next_id == tokenizer.token_to_id("[EOS]"):
            break

    tokens = tokenizer.decode(input_ids[0].tolist())
    return tokens

if __name__ == "__main__":
    main('train')