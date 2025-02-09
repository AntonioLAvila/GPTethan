import torch
import torch.nn as nn
import torch.optim as optim
from GPTethan import Transformer
import db_manip as db
import random
import pickle


# with open("ethan_msgs_rep.pkl", "rb") as f:
#     ethan_msgs_rep = pickle.load(f)
# with open("ethan_msgs.pkl", "rb") as f:
#     ethan_msgs = pickle.load(f)
# with open("id_to_word.pkl", "rb") as f:
#     id_to_word = pickle.load(f)
# with open("word_to_id.pkl", "rb") as f:
#     word_to_id = pickle.load(f)

# TODO: artificially extend shorter messages to max length

ethan_msgs = db.get_user_msgs('whetan', 20)

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
    return rep

def rep_to_msg(msg):
    '''
    return list of words from one hot encodings
    '''
    return [id_to_word[torch.argmax(encoding)] for encoding in msg]

ethan_msgs_rep = [msg_to_rep(msg) for msg in ethan_msgs]

with open("ethan_msgs_rep.pkl", "wb") as f:
    pickle.dump(ethan_msgs_rep, f)
with open("ethan_msgs.pkl", "wb") as f:
    pickle.dump(ethan_msgs, f)
with open("id_to_word.pkl", "wb") as f:
    pickle.dump(id_to_word, f)
with open("word_to_id.pkl", "wb") as f:
    pickle.dump(word_to_id, f)

print("Done with messages")

indices = list(range(0, len(ethan_msgs_rep)))
random.shuffle(indices)

src_data = [ethan_msgs_rep[i] for i in indices[:len(indices)//3]]
tgt_data = [ethan_msgs_rep[i] for i in indices[len(indices)//3:2*len(indices)//3]]
val_data = [ethan_msgs_rep[i] for i in indices[2*len(indices)//3:]]

print(type(src_data[0][0][0]))

print("Done assigning")


transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)

src_data = torch.tensor(src_data) # torch.randint(1, src_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)
tgt_data = torch.tensor(tgt_data) # torch.randint(1, tgt_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)

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

# val_src_data = torch.tensor(val_data[:len(val_data)//2]) # torch.randint(1, src_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)
# val_tgt_data = torch.tensor(val_data[len(val_data)//2:]) # torch.randint(1, tgt_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)

# with torch.no_grad():

#     val_output = transformer(val_src_data, val_tgt_data[:, :-1])
#     val_loss = criterion(val_output.contiguous().view(-1, tgt_vocab_size), val_tgt_data[:, 1:].contiguous().view(-1))
#     print(f"Validation Loss: {val_loss.item()}")