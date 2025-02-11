from GPTethan import Transformer
import pickle
import torch

with open("ethan_msgs_rep.pkl", "rb") as f:
    ethan_msgs_rep = pickle.load(f)
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

src_vocab_size = tgt_vocab_size = len(tokenizer.id_to_word)
d_model = 1024
num_heads = 8
num_layers = 6
d_ff = 2048
max_seq_length = len(ethan_msgs_rep[0])
dropout = 0.1
batch_size = 16
num_epochs = 250

print(f"{src_vocab_size=}, {max_seq_length=}, {len(ethan_msgs_rep)=}")

# Create model
transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout).to('cuda')
transformer.load_state_dict(torch.load("model.pth"))

def generate_response(input_text: str, max_response_length=max_seq_length):
    transformer.eval()
    
    # Tokenize input and add special tokens
    input_tokens = ["<sos>"] + input_text.split() + ["<eos>"]
    input_tensor = torch.tensor([tokenizer.encode(input_tokens)], dtype=torch.long).to("cuda")

    # Start response with <sos>
    response_tokens = ["<sos>"]
    
    for _ in range(max_response_length):
        response_tensor = torch.tensor([tokenizer.encode(response_tokens)], dtype=torch.long).to("cuda")
        
        with torch.no_grad():
            output = transformer(input_tensor, response_tensor)  # Generate next token probabilities
            next_token_id = torch.argmax(output[:, -1, :]).item()  # Get highest probability token
            
        next_token = tokenizer.id_to_word[next_token_id]
        
        if next_token == "<eos>":
            break  # Stop if end token is reached
        
        response_tokens.append(next_token)
    
    return tokenizer.decode(response_tokens[1:])  # Exclude <sos> from output

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    ethan_response = generate_response(user_input)
    print(f"Ethan: {ethan_response}")