import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import Counter
import numpy as np
import os
import json
import math
import inspect
from tqdm import tqdm
import multiprocessing as mp
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import LambdaLR

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 300000
eval_interval = 200
device = 'mps'
eval_iters = 20
n_embd = 256
n_head = 1
n_layer = 1
dropout = 0
# ------------
# Initialize lists to store losses
train_losses = []
val_losses = []

# Set the path for saving models to Google Drive
#save_dir = '/content/drive/MyDrive/nlp_models_5'
#os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists
MASKED_DATA_DIR = "edu_fineweb10B_masked_small"
torch.manual_seed(1337)

tokenizer = tiktoken.get_encoding("gpt2")

# Add a [MASK] token
MASK_TOKEN = '[MASK]'
MASK_TOKEN_ID = 50257  # Assign a new token ID for [MASK]

PAD_TOKEN = '[PAD]'
PAD_TOKEN_ID = tokenizer.n_vocab + 1  # Assign a new token ID for [PAD]


unique_tokens = 50304
vocab_size = unique_tokens  # Ensure this matches your tokenizer's vocab size


max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10000
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_iters - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)


def save_losses_to_json():
    losses_data = {
        'train_losses': train_losses,
        'val_losses': val_losses
    }
    losses_file = os.path.join(save_dir, 'losses.json')
    with open(losses_file, mode='w') as file:
        json.dump(losses_data, file)
    print(f"Losses saved to {losses_file}")


def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt



class MaskedTextDataset(Dataset):
    def __init__(self, data_dir, split):
        self.data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if split in f and f.endswith('.npz')]
        self.data = []
        for file in tqdm(self.data_files, desc=f"Loading {split} data"):
            data = np.load(file)
            input_ids = data['input_ids']
            labels = data['labels']
            # Split into chunks of block_size
            for i in range(0, len(input_ids) - block_size + 1, block_size):
                self.data.append((
                    input_ids[i:i+block_size],
                    labels[i:i+block_size]
                ))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.data[idx][0], dtype=torch.long)
        labels = torch.tensor(self.data[idx][1], dtype=torch.long)
        return input_ids, labels

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

# Create datasets
train_dataset = MaskedTextDataset(MASKED_DATA_DIR, split='train')
val_dataset = MaskedTextDataset(MASKED_DATA_DIR, split='val')

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=RandomSampler(train_dataset))
val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=SequentialSampler(val_dataset))


class Head(nn.Module):
    """ One head of self-attention or cross-attention """
    def __init__(self, head_size, masking=False):
        super().__init__()
        self.masking = masking
        self.k = nn.Linear(n_embd, head_size, bias=False)
        self.q = nn.Linear(n_embd, head_size, bias=False)
        self.v = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        if masking:
            self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x, key_input=None, value_input=None):
        B,T,C = x.shape
        if key_input is None and value_input is None:
            # Self-attention
            k = self.k(x)
            q = self.q(x)
            v = self.v(x)
        else:
            # Cross-attention
            k = self.k(key_input)
            q = self.q(x)
            v = self.v(value_input)

        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        if self.masking:
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultipleHeadAttention(nn.Module):
    """ Multiple heads of self-attention or cross-attention in parallel """
    def __init__(self, num_heads, head_size, masking=True):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, masking=masking) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_input=None, value_input=None):
        out = torch.cat([h(x, key_input, value_input) for h in self.heads], dim=-1)
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

class EncoderBlock(nn.Module):
    """ Transformer encoder block """
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultipleHeadAttention(n_head, head_size, masking=False)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class MaskedLanguageModel(nn.Module):
    def __init__(self, n_layer, n_head, n_embd, vocab_size, dropout):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd) 
        self.blocks = nn.Sequential(
            *[EncoderBlock(n_embd, n_head) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, input_ids, labels=None):
        B, T = input_ids.size()
        position_ids = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        x = self.tok_emb(input_ids) + self.pos_emb(position_ids)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if labels is not None:
            # Shift logits and labels for MLM
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

if __name__ == "__main__":
    # Set precision for matmul
    torch.set_float32_matmul_precision('high')

    model = MaskedLanguageModel(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    vocab_size=vocab_size,
    dropout=dropout
    )

    model.to(device)

    # Optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=0.01)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: get_lr(step) / max_lr)

    # Training and evaluation functions
    def train_epoch(epoch):
        model.train()
        total_loss = 0
        for step, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch}")):
            input_ids, labels = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, labels=labels)
            loss = outputs[1]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            if step % eval_interval == 0 and step > 0:
                avg_loss = total_loss / (step + 1)
                print(f"Epoch {epoch}, Step {step}, Training Loss: {avg_loss:.4f}")

    def evaluate():
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                input_ids, labels = batch
                input_ids = input_ids.to(device)
                labels = labels.to(device)

                outputs = model(input_ids, labels=labels)
                loss = outputs[1]
                total_loss += loss.item()

        avg_loss = total_loss / len(val_loader)
        print(f"Validation Loss: {avg_loss:.4f}")
        model.train()
        return avg_loss

    # Main training loop
    best_val_loss = float('inf')
    num_epochs = 1
    for epoch in range(1, num_epochs + 1):
        train_epoch(epoch)
        val_loss = evaluate()

        # Save the model if it has the best validation loss so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(save_dir, 'best_model.pth')
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved with validation loss: {best_val_loss:.4f}")