import torch
import torch.nn as nn
import torch.optim as optim
from GPTethan import Transformer
import db_manip as db

ethan_msgs = db.get_user_msgs_unlimited()
id_to_word, word_to_id = db.vocab_mapping(ethan_msgs)

src_vocab_size = len(id_to_word) # 5000
tgt_vocab_size = len(id_to_word) # 5000
d_model = 1024 # 512
num_heads = 8
num_layers = 6
d_ff = 4096 # 2048
max_seq_length = max([len(i) for i in ethan_msgs]) # 100
dropout = 0.1


def msg_to_rep(msg):
    '''
    return a list of one hot word encodings
    '''
    rep = []
    for word in msg:
        word_rep = [0]*src_vocab_size
        word_rep[word_to_id[word]] = 1
        rep.append(word_rep)
    return torch.tensor(rep)

def rep_to_msg(msg):
    '''
    return list of words from one hot encodings
    '''
    return [id_to_word[torch.argmax(encoding)] for encoding in msg]


transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)

# Generate random sample data
src_data = torch.randint(1, src_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)
tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

transformer.train()

for epoch in range(100):
    optimizer.zero_grad()
    output = transformer(src_data, tgt_data[:, :-1])
    loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")



transformer.eval()

# Generate random sample validation data
val_src_data = torch.randint(1, src_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)
val_tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)

with torch.no_grad():

    val_output = transformer(val_src_data, val_tgt_data[:, :-1])
    val_loss = criterion(val_output.contiguous().view(-1, tgt_vocab_size), val_tgt_data[:, 1:].contiguous().view(-1))
    print(f"Validation Loss: {val_loss.item()}")