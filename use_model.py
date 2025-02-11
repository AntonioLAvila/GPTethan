from GPTethan import Transformer
import pickle
import torch

with open("ethan_msgs_rep.pkl", "rb") as f:
    ethan_msgs_rep = pickle.load(f)
with open("id_to_word.pkl", "rb") as f:
    id_to_word = pickle.load(f)
with open("word_to_id.pkl", "rb") as f:
    word_to_id = pickle.load(f)

src_vocab_size = len(id_to_word) # 5000
tgt_vocab_size = len(id_to_word) # 5000
d_model = 1024
num_heads = 8
num_layers = 6
d_ff = 2048
max_seq_length = len(ethan_msgs_rep[0])# 100
dropout = 0.1

transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout).to('cuda')

transformer.load_state_dict(torch.load("model.pth"))

transformer.eval()

test_input = "what time is it"
test_input_rep = torch.tensor([word_to_id[word] for word in test_input.split()]).to('cuda')

response = transformer
print(response)

