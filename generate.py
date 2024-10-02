import torch
from train_decoder import BiagramLanguageModel
import tiktoken

# hyperparameters
# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 700000
eval_interval = 200
device = 'mps'
eval_iters = 20
n_embd = 768
n_head = 12
n_layer = 8
dropout = 0

# Recreate the model architecture
model = BiagramLanguageModel(n_layer, n_head, n_embd)

# Load the saved state dict and map it to the appropriate device
model.load_state_dict(torch.load('bigram_language_model_oneshard_iter_92500.pth', map_location=torch.device(device)))

# Move the model to the selected device
m = model.to(device)

# Set the model to evaluation mode (for inference)
m.eval()

# Tokenization
tokenizer = tiktoken.get_encoding("gpt2")
sentence = "Hello, As a Language Model"
tokens = tokenizer.encode(sentence)

# Create a tensor for the context and move it to the correct device
context = torch.tensor([tokens], dtype=torch.long, device=device)

# Generate tokens and decode the output
print(tokenizer.decode(m.generate(context, max_new_tokens=200)[0].tolist()))
