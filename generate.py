import torch
from train import BiagramLanguageModel
import tiktoken
# Recreate the model architecture
# hyperparameters
batch_size = 16 # how many independent sequences will we process in parallel?
block_size = 32 # what is the maximum context length for predictions?
max_iters = 7500
eval_interval = 200
learning_rate = 1e-3
device = torch.device('mps')
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.1

model = BiagramLanguageModel(n_layer, n_head, n_embd)  # Use the same architecture parameters as before

# Load the saved state dict
model.load_state_dict(torch.load('bigram_language_model.pth'))

# Move the model to the device (e.g., MPS, CUDA, or CPU)
m = model.to(device)

# Set the model to evaluation mode (for inference)
m.eval()
tokenizer = tiktoken.get_encoding("gpt2")

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(tokenizer.decode(m.generate(context, max_new_tokens=2000)[0].tolist()))
