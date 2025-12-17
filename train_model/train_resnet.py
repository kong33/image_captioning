import time

import os
import json
import random
import re
from collections import Counter
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import ImageCaptioningModel_Resnet

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get("DATA_ROOT")
if DATA_ROOT is None:
    raise ValueError("ERROR: DATA_ROOT is not set!")
BACKBONE_NAME = "Resnet50"

CAPTIONS_PATH = os.path.join(DATA_ROOT, "captions.txt")
IMAGES_DIR = os.path.join(DATA_ROOT, "Images")
CHECKPOINT_DIR = os.path.join(
    BASE_DIR,
    "checkpoints",
    f"{BACKBONE_NAME}_enc{4}_dec{6}",
)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

HISTORY_JSON_PATH = os.path.join(CHECKPOINT_DIR, "history.json")
HISTORY_PLOT_PATH = os.path.join(CHECKPOINT_DIR, "history.png")
TOKENIZER_VOCAB_PATH = os.path.join(CHECKPOINT_DIR, "tokenizer_vocab.txt")
MODEL_WEIGHTS_PATH = os.path.join(CHECKPOINT_DIR, "caption_model.pt")

TRAIN_LIST_PATH = os.path.join(CHECKPOINT_DIR, "train_images.txt")
VAL_LIST_PATH = os.path.join(CHECKPOINT_DIR, "val_images.txt")
TEST_LIST_PATH = os.path.join(CHECKPOINT_DIR, "test_images.txt")

MAX_LENGTH = 40
VOCABULARY_SIZE = 10000
BATCH_SIZE = 32
BUFFER_SIZE = 1000
EMBEDDING_DIM = 512
IMAGE_FEATURE_DIM = 512
UNITS = 512
NUM_ENCODER_LAYERS = 4
NUM_DECODER_LAYERS = 6
NUM_HEADS_ENCODER = 8
NUM_HEADS_DECODER = 8
EPOCHS = 10
DIM_FF = 4 * IMAGE_FEATURE_DIM
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] device = {device}")

def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = text.strip()
    text = "[start] " + text + " [end]"
    return text

print(f"[INFO] Loading captions from: {CAPTIONS_PATH}")
captions_df = pd.read_csv(CAPTIONS_PATH)
captions_df["image"] = captions_df["image"].apply(
    lambda x: os.path.join(IMAGES_DIR, x)
)
captions_df["caption"] = captions_df["caption"].apply(preprocess)

img_to_cap_vector: Dict[str, List[str]] = {}
for img, cap in zip(captions_df["image"], captions_df["caption"]):
    img_to_cap_vector.setdefault(img, []).append(cap)

img_keys = list(img_to_cap_vector.keys())
random.shuffle(img_keys)

n_imgs = len(img_keys)
train_end = int(n_imgs * 0.8)
val_end = int(n_imgs * 0.9)

img_name_train_keys = img_keys[:train_end]
img_name_val_keys = img_keys[train_end:val_end]
img_name_test_keys = img_keys[val_end:]

print(f"[INFO] # total unique images: {n_imgs}")
print(f"[INFO] # train images: {len(img_name_train_keys)}")
print(f"[INFO] # val   images: {len(img_name_val_keys)}")
print(f"[INFO] # test  images: {len(img_name_test_keys)}")

with open(TRAIN_LIST_PATH, "w", encoding="utf-8") as f:
    for p in img_name_train_keys:
        f.write(p + "\n")
with open(VAL_LIST_PATH, "w", encoding="utf-8") as f:
    for p in img_name_val_keys:
        f.write(p + "\n")
with open(TEST_LIST_PATH, "w", encoding="utf-8") as f:
    for p in img_name_test_keys:
        f.write(p + "\n")

print(f"[INFO] Saved split lists to:")
print(f"  - {TRAIN_LIST_PATH}")
print(f"  - {VAL_LIST_PATH}")
print(f"  - {TEST_LIST_PATH}")

train_imgs, train_captions = [], []
for imgt in img_name_train_keys:
    caps = img_to_cap_vector[imgt]
    train_imgs.extend([imgt] * len(caps))
    train_captions.extend(caps)

val_imgs, val_captions = [], []
for imgv in img_name_val_keys:
    caps = img_to_cap_vector[imgv]
    val_imgs.extend([imgv] * len(caps))
    val_captions.extend(caps)

test_imgs, test_captions = [], []
for imgt in img_name_test_keys:
    caps = img_to_cap_vector[imgt]
    test_imgs.extend([imgt] * len(caps))
    test_captions.extend(caps)

print(f"[INFO] # train samples: {len(train_imgs)}")
print(f"[INFO] # val   samples: {len(val_captions)}")
print(f"[INFO] # test  samples: {len(test_captions)}")

def build_vocab(texts: List[str], max_size: int) -> Tuple[Dict[str, int], List[str]]:
    counter = Counter()
    for t in texts:
        tokens = t.split()
        counter.update(tokens)
    vocab_tokens = [PAD_TOKEN, UNK_TOKEN]
    for tok, _ in counter.most_common(max_size - 2):
        if tok not in vocab_tokens:
            vocab_tokens.append(tok)
    token2id = {tok: idx for idx, tok in enumerate(vocab_tokens)}
    return token2id, vocab_tokens

token2id, vocab_tokens = build_vocab(train_captions, VOCABULARY_SIZE)
vocab_size = len(vocab_tokens)
pad_id = token2id[PAD_TOKEN]
unk_id = token2id[UNK_TOKEN]

with open(TOKENIZER_VOCAB_PATH, "w", encoding="utf-8") as f:
    for tok in vocab_tokens:
        f.write(tok + "\n")
print(f"[INFO] Saved tokenizer vocab to: {TOKENIZER_VOCAB_PATH}")

def encode_caption(text: str) -> List[int]:
    tokens = text.split()
    ids = [token2id.get(tok, unk_id) for tok in tokens]
    if len(ids) > MAX_LENGTH:
        ids = ids[:MAX_LENGTH]
    else:
        ids = ids + [pad_id] * (MAX_LENGTH - len(ids))
    return ids

class FlickrDataset(Dataset):
    def __init__(self, img_paths: List[str], captions: List[str], transform=None):
        self.img_paths = img_paths
        self.captions = captions
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        cap_text = self.captions[idx]

        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        cap_ids = torch.tensor(encode_caption(cap_text), dtype=torch.long)
        return img, cap_ids

IMG_SIZE = 299

img_transform = T.Compose(
    [
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ]
)

train_dataset = FlickrDataset(train_imgs, train_captions, transform=img_transform)
val_dataset = FlickrDataset(val_imgs, val_captions, transform=img_transform)
test_dataset = FlickrDataset(test_imgs, test_captions, transform=img_transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

model = ImageCaptioningModel_Resnet(
    vocab_size=vocab_size,
    d_model=IMAGE_FEATURE_DIM,
    n_heads_enc=NUM_HEADS_ENCODER,
    n_layers_enc=NUM_ENCODER_LAYERS,
    n_heads_dec=NUM_HEADS_DECODER,
    n_layers_dec=NUM_DECODER_LAYERS,
    dim_ff=DIM_FF,
    max_len=MAX_LENGTH,
    pad_id=pad_id,
).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

if os.path.exists(MODEL_WEIGHTS_PATH):
    print(f"[INFO] Loading existing weights from: {MODEL_WEIGHTS_PATH}")
    state = torch.load(MODEL_WEIGHTS_PATH, map_location=device)
    model.load_state_dict(state)
else:
    print("[INFO] No existing weights found. Training from scratch.")

def run_epoch(loader, model, optimizer=None):
    """
    loader: train_loader or val_loader
    optimizer: None -> eval 모드
    """
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_tokens = 0
    total_correct = 0

    for imgs, caps in loader:
        imgs = imgs.to(device)
        caps = caps.to(device)

        y_inp = caps[:, :-1]
        y_tgt = caps[:, 1:]

        if is_train:
            optimizer.zero_grad()

        logits = model(imgs, y_inp)
        logits = logits[:, : y_inp.size(1), :]

        B, Lm1, V = logits.shape
        loss = criterion(
            logits.reshape(B * Lm1, V),
            y_tgt.reshape(B * Lm1),
        )

        if is_train:
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            preds = logits.argmax(-1)
            mask = (y_tgt != pad_id)
            correct = (preds == y_tgt) & mask
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()
            total_loss += loss.item() * mask.sum().item()

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    avg_acc = total_correct / total_tokens if total_tokens > 0 else 0.0
    return avg_loss, avg_acc

history = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": [],
    "epoch_time_sec": [],
    "epoch_gpu_mem_gb": [],
}

best_val_loss = float("inf")
patience = 2
no_improve = 0
total_start_time = time.time()

for epoch in range(1, EPOCHS + 1):
    print(f"\n===== Epoch {epoch}/{EPOCHS} =====")
    epoch_start_time = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    train_loss, train_acc = run_epoch(train_loader, model, optimizer)
    val_loss, val_acc = run_epoch(val_loader, model, optimizer=None)
    epoch_time = time.time() - epoch_start_time
    if torch.cuda.is_available():
        peak_mem_bytes = torch.cuda.max_memory_allocated(device)
        peak_mem_gb = peak_mem_bytes / (1024 ** 3)
    else:
        peak_mem_gb = 0.0

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    history["epoch_time_sec"].append(epoch_time)
    history["epoch_gpu_mem_gb"].append(peak_mem_gb)

    total_train_time = time.time() - total_start_time
    history["total_train_time_sec"] = total_train_time
    print(f"[INFO] Total training time: {total_train_time:.2f} sec")

    print(f"[EPOCH {epoch}] train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
          f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
    print(
        f"[EPOCH {epoch}] time={epoch_time:.2f} sec, "
        f"peak_gpu_mem={peak_mem_gb:.3f} GB"
    )
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve = 0
        torch.save(model.state_dict(), MODEL_WEIGHTS_PATH)
        print(f"[INFO] Saved best model to: {MODEL_WEIGHTS_PATH}")
    else:
        no_improve += 1
        print(f"[INFO] No improvement for {no_improve} epoch(s)")
        if no_improve >= patience:
            print("[INFO] Early stopping triggered.")
            break


with open(HISTORY_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(history, f, indent=4)
print(f"[INFO] Saved training history to: {HISTORY_JSON_PATH}")

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history["train_loss"], label="train_loss")
plt.plot(history["val_loss"], label="val_loss")
plt.title("Loss over epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history["train_acc"], label="train_acc")
plt.plot(history["val_acc"], label="val_acc")
plt.title("Accuracy over epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(HISTORY_PLOT_PATH)
print(f"[INFO] Saved training curves PNG to: {HISTORY_PLOT_PATH}")

print("===== Final Metrics =====")
print(f"Train loss: {history['train_loss'][-1] if history['train_loss'] else None}")
print(f"Val   loss: {history['val_loss'][-1] if history['val_loss'] else None}")
print(f"Train acc : {history['train_acc'][-1] if history['train_acc'] else None}")
print(f"Val   acc : {history['val_acc'][-1] if history['val_acc'] else None}")
