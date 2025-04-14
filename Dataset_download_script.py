from datasets import load_dataset
from torchvision import transforms
from transformers import CLIPTokenizer
from PIL import Image
import torch
import random
import os
from tqdm import tqdm

# MY_TOKEN = "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"
SAVE_PATH = "flickr30k_preprocessed_1pct.pt"

# Check if GPU is available
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# Load dataset
dataset = load_dataset("nlphuji/flickr30k", split="test[:1%]", trust_remote_code=True)
# Preprocessing
image_size = 512
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # [-1, 1] normalization
])
# Load tokenizer
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
# Preprocess & Save
processed_data = []
for entry in tqdm(dataset, desc="Processing Images", unit="sample"):
    # Convert and transform image, move to GPU
    image = transform(entry['image'].convert("RGB")).to(device)
    # Select a random caption
    caption = random.choice(entry['caption'])  
    # Tokenize the caption
    tokens = tokenizer(caption, padding="max_length", truncation=True, max_length=77, return_tensors="pt")   
    # Move tokenized input to GPU
    input_ids = tokens["input_ids"].squeeze(0).to(device)
    attention_mask = tokens["attention_mask"].squeeze(0).to(device)   
    # Append to processed_data
    processed_data.append({
        "image": image,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "text": caption
    })

# Save the preprocessed data to a .pt file (list of dicts)
torch.save(processed_data, SAVE_PATH)
print(f"Saved {len(processed_data)} samples to {SAVE_PATH}")




















# from datasets import load_dataset
# from torchvision import transforms
# from transformers import CLIPTokenizer
# from PIL import Image
# import torch
# import random
# import os

# MY_TOKEN = "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"
# SAVE_PATH = "flickr30k_preprocessed_1pct.pt"

# # Load dataset
# dataset = load_dataset("nlphuji/flickr30k", split="train[:1%]", use_auth_token=MY_TOKEN)

# # Preprocessing
# image_size = 512
# transform = transforms.Compose([
#     transforms.Resize((image_size, image_size)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.5], [0.5])  # [-1, 1] normalization
# ])

# # Load tokenizer
# tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

# # Preprocess & Save
# processed_data = []
# for entry in dataset:
#     image = transform(entry['image'].convert("RGB"))
#     caption = random.choice(entry['caption'])
#     tokens = tokenizer(caption, padding="max_length", truncation=True, max_length=77, return_tensors="pt")
#     processed_data.append({
#         "image": image,
#         "input_ids": tokens["input_ids"].squeeze(0),
#         "attention_mask": tokens["attention_mask"].squeeze(0),
#         "text": caption
#     })

# # Save as a .pt file (list of dicts)
# torch.save(processed_data, SAVE_PATH)
# print(f"Saved {len(processed_data)} samples to {SAVE_PATH}")
