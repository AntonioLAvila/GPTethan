DB = "/home/antonio/dht/archive.dht"
ethan_id = 791492026986135592
ethan_name = 'whetan'
staleness = 60*60*24*5 # 5 days

data_dir = "data.txt"

d_model = 900
d_ff = 4*d_model
n_layers = 8
n_heads = 6
dropout = 0.1

max_len = 1024
batch_size = 8
lr = 1e-5

model_path = "ethan.pt"
tokenizer_path = "tokenizer.json"
