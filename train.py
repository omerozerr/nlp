import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import Counter
import numpy as np
import os

# hyperparameters
batch_size = 16 # how many independent sequences will we process in parallel?
block_size = 128 # what is the maximum context length for predictions?
max_iters = 50000
eval_interval = 100
learning_rate = 1e-3
device = 'mps'
eval_iters = 200
n_embd = 128
n_head = 8
n_layer = 4
dropout = 0.2
# ------------

#torch.manual_seed(1337)

with(open ("input.txt", "r", encoding='utf-8')) as f:
    text = f.read()

tokenizer = tiktoken.get_encoding("gpt2")

# Tokenize the text
tokens = tokenizer.encode(text)

# Count the occurrence of each token
token_counts = Counter(tokens)

# Get the number of unique tokens
unique_tokens = 50257

data = torch.tensor(tokens, dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    if split == 'train':
        data = train_data
    else:
        data = val_data
    # sample a random block of size `block_size + 1`
    start = torch.randint(data.size(0) - block_size - 1, (batch_size,))
    # batch of size `batch_size` x `block_size`
    batch = torch.stack([data[s:s+block_size] for s in start])
    # batch of size `batch_size` x `block_size`
    labels = torch.stack([data[s+1:s+block_size+1] for s in start])
    return batch, labels

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


class DataLoaderLite:
    def __init__(self, B, T, split):
        self.B = B
        self.T = T
        assert split in {'train', 'val'}

        # get the shard filenames
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T
        return x, y

train_loader = DataLoaderLite(B=batch_size, T=block_size, split="train")
val_loader = DataLoaderLite(B=batch_size, T=block_size, split="val")

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()  # Use 'm' instead of 'model'
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            if split == 'train':
                X, Y = train_loader.next_batch()
            else:
                X, Y = val_loader.next_batch()
            X, Y = X.to(device), Y.to(device)
            logits, loss = model(X, Y)  # Use 'm' instead of 'model'
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()  # Use 'm' instead of 'model'
    return out


class Head(nn.Module):
    """ one head of self-attention """
    def __init__(self, head_size):
        super().__init__()
        self.k = nn.Linear(n_embd, head_size, bias=False)
        self.q= nn.Linear(n_embd, head_size, bias=False)
        self.v = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size, device=device)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.k(x)  # (B,T,C)
        q = self.q(x) # (B,T,C) 
        v = self.v(x) # (B,T,C)

        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out
    
class MultipleHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    
class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        #######
        head_size = n_embd // n_head
        ######
        self.sa = MultipleHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    
class BiagramLanguageModel(nn.Module):
    def __init__(self, n_layer, n_head, n_embd):
        super().__init__()
        self.tok_emb = nn.Embedding(unique_tokens, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, unique_tokens)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.tok_emb(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    
if __name__ == "__main__":
    # Set precision for matmul
    torch.set_float32_matmul_precision('high')

    # Initialize your model
    model = BiagramLanguageModel(n_layer, n_head, n_embd)
    model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Training loop
    for iter in range(max_iters):
        # Evaluation and printing loss at the evaluation interval
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # Fetch the next batch
        xb, yb = train_loader.next_batch()
        xb, yb = xb.to(device), yb.to(device)

        # Forward pass, loss computation, and backward pass
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Save the model every 5000 iterations
        if iter % 2500 == 0 and iter > 0:
            save_path = f'bigram_language_model_iter_{iter}.pth'
            torch.save(model.state_dict(), save_path)
            print(f"Model saved at iteration {iter}")

    # Save the final trained model
    torch.save(model.state_dict(), 'bigram_language_model_final.pth')
    print("Final model saved.")


