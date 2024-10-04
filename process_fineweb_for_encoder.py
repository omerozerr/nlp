import os
import numpy as np
from tqdm import tqdm
import multiprocessing as mp

# Directory where your processed data files are stored
DATA_CACHE_DIR = "edu_fineweb10B"  # Adjust if different
# Directory to save the masked data
MASKED_DATA_DIR = "edu_fineweb10B_masked"
os.makedirs(MASKED_DATA_DIR, exist_ok=True)

# Parameters
masking_prob = 0.15  # Probability of masking a token
mask_token_id = 50257  # Replace with the actual mask token ID from your tokenizer

def apply_masking(file_name):
    input_path = os.path.join(DATA_CACHE_DIR, file_name)
    output_path = os.path.join(MASKED_DATA_DIR, file_name.replace('.npy', '.npz'))

    # Load the tokens
    tokens_np = np.load(input_path)
    
    # Create labels (copy of original tokens)
    labels_np = np.copy(tokens_np).astype(np.int32)  # Use int32 to accommodate -100

    # Apply masking
    num_tokens = tokens_np.shape[0]
    num_mask = int(masking_prob * num_tokens)
    if num_mask == 0:
        num_mask = 1  # Ensure at least one token is masked

    mask_indices = np.random.choice(num_tokens, size=num_mask, replace=False)
    tokens_np[mask_indices] = mask_token_id  # Replace tokens with mask token
    labels_np[~np.isin(np.arange(num_tokens), mask_indices)] = -100  # Ignore unmasked tokens in loss computation

    # Save the masked data
    np.savez(output_path, input_ids=tokens_np, labels=labels_np)

def process_all_files():
    files = [f for f in os.listdir(DATA_CACHE_DIR) if f.endswith('.npy')]
    nprocs = max(1, os.cpu_count() // 2)
    with mp.Pool(nprocs) as pool:
        list(tqdm(pool.imap_unordered(apply_masking, files), total=len(files), desc="Processing files"))

if __name__ == '__main__':
    process_all_files()
