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
from torch.cuda.amp import autocast, GradScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import ImageCaptioningModel_EfficientNet

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get("DATA_ROOT")
if DATA_ROOT is None:
    raise ValueError("ERROR: DATA_ROOT is not set!")
BACKBONE_NAME = "efficientNetb4"

CAPTIONS_PATH = os.path.join(DATA_ROOT, "captions.txt")
IMAGES_DIR = os.path.join(DATA_ROOT, "Images")
CHECKPOINT_DIR = os.path.join(
    BASE_DIR,
    "checkpoints",
    f"{BACKBONE_NAME}_final_SCST_enc{3}_dec{5}",
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
BATCH_SIZE = 16
BUFFER_SIZE = 1000
EMBEDDING_DIM = 512
IMAGE_FEATURE_DIM = 512
UNITS = 512 
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 5
NUM_HEADS_ENCODER = 8
NUM_HEADS_DECODER = 8
EPOCHS = 10
DIM_FF = 4 * IMAGE_FEATURE_DIM
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

USE_L1_SCST = True
LAMBDA_SCST = 0.1
USE_IMAGE_GROUND_LOSS = True
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

START_TOKEN = "[start]"
END_TOKEN = "[end]"
start_id = token2id[START_TOKEN]
end_id = token2id[END_TOKEN]

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

model = ImageCaptioningModel_EfficientNet(
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
scaler = GradScaler() if device.type == "cuda" else None
if os.path.exists(MODEL_WEIGHTS_PATH):
    print(f"[INFO] Loading existing weights from: {MODEL_WEIGHTS_PATH}")
    state = torch.load(MODEL_WEIGHTS_PATH, map_location=device)
    model.load_state_dict(state)
else:
    print("[INFO] No existing weights found. Training from scratch.")


def strip_special_tokens(seq, pad_id, start_id, end_id):
    return [
        t for t in seq
        if t not in (pad_id, start_id, end_id)
    ]








def compute_reward(
    candidate_ids,
    reference_ids,
    pad_id,
    start_id,
    end_id,
    max_n=4,
    eps=1e-8
):
    """
    CIDEr-style TF-IDF-based reward.

    논문: "CIDEr: Consensus-based Image Description Evaluation" (Vedantam et al., CVPR 2015)
    본 구현은 SCST 학습 단계에서 사용하는 경량화 버전이며,
    계산 구조는 TF-IDF 기반 cosine similarity 형태를 유지함.
    """

    cand = strip_special_tokens(candidate_ids, pad_id, start_id, end_id)
    ref  = strip_special_tokens(reference_ids, pad_id, start_id, end_id)

    if len(cand) == 0 or len(ref) == 0:
        return 0.0

    def get_ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

    scores = []

    for n in range(1, max_n + 1):
        cand_ngrams = Counter(get_ngrams(cand, n))
        ref_ngrams  = Counter(get_ngrams(ref,  n))

        if len(cand_ngrams) == 0 or len(ref_ngrams) == 0:
            continue
        
        cand_vec = []
        ref_vec = []

        all_keys = set(cand_ngrams.keys()) | set(ref_ngrams.keys())

        for g in all_keys:
            tf_c = cand_ngrams[g] / max(1, sum(cand_ngrams.values()))
            tf_r = ref_ngrams[g]  / max(1, sum(ref_ngrams.values()))

            w_c = tf_c  
            w_r = tf_r

            cand_vec.append(w_c)
            ref_vec.append(w_r)

        cand_vec = np.array(cand_vec)
        ref_vec  = np.array(ref_vec)

        sim = np.dot(cand_vec, ref_vec) / (np.linalg.norm(cand_vec) * np.linalg.norm(ref_vec) + eps)
        scores.append(sim)

    cider_score = np.mean(scores)

    return float(cider_score)


def decode_greedy(model, images, max_len, start_id, end_id):
    """
    Greedy decoding: argmax로 caption 생성
    images: (B,3,H,W)
    반환: (B, L) tensor (start 포함)
    """
    model.eval()
    B = images.size(0)
    device_ = images.device

    seqs = torch.full((B, 1), start_id, dtype=torch.long, device=device_)
    finished = torch.zeros(B, dtype=torch.bool, device=device_)

    with torch.no_grad():
        for _ in range(max_len - 1):
            logits = model(images, seqs)
            next_logit = logits[:, -1, :]
            next_token = next_logit.argmax(dim=-1)

            next_token = next_token.masked_fill(finished, pad_id)
            seqs = torch.cat([seqs, next_token.unsqueeze(1)], dim=1)

            finished = finished | (next_token == end_id)
            if finished.all():
                break
    return seqs


def decode_sample(model, images, max_len, start_id, end_id, temperature=1.0):
    """
    Sampling decoding: multinomial sampling으로 caption 생성
    """
    model.eval()
    B = images.size(0)
    device_ = images.device

    seqs = torch.full((B, 1), start_id, dtype=torch.long, device=device_)
    finished = torch.zeros(B, dtype=torch.bool, device=device_)

    with torch.no_grad():
        for _ in range(max_len - 1):
            logits = model(images, seqs)
            next_logit = logits[:, -1, :]

            if temperature != 1.0:
                next_logit = next_logit / temperature

            probs = F.softmax(next_logit, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)

            next_token = next_token.masked_fill(finished, pad_id)
            seqs = torch.cat([seqs, next_token.unsqueeze(1)], dim=1)

            finished = finished | (next_token == end_id)
            if finished.all():
                break
    return seqs
















def compute_scst_loss(model, images, gt_caps, pad_id, start_id, end_id, lambda_scst):
    """
    SCST:
    - greedy caption과 sampling caption을 생성
    - reward(candidate, gt) 계산
    - advantage = r_sample - r_greedy
    - loss = -E[ advantage * log p(sample) ]
    """
    B = images.size(0)
    device_ = images.device

    with torch.no_grad():
        greedy_seqs = decode_greedy(model, images, MAX_LENGTH, start_id, end_id)
        sample_seqs = decode_sample(model, images, MAX_LENGTH, start_id, end_id)

        rewards_greedy = []
        rewards_sample = []
        for b in range(B):
            ref_ids = gt_caps[b].tolist()

            r_g = compute_reward(greedy_seqs[b].tolist(), ref_ids,
                                 pad_id, start_id, end_id)
            r_s = compute_reward(sample_seqs[b].tolist(), ref_ids,
                                 pad_id, start_id, end_id)
            rewards_greedy.append(r_g)
            rewards_sample.append(r_s)

        rewards_greedy = torch.tensor(rewards_greedy, dtype=torch.float32, device=device_)
        rewards_sample = torch.tensor(rewards_sample, dtype=torch.float32, device=device_)

        advantage = rewards_sample - rewards_greedy

    model.train()

    B_s, L_s = sample_seqs.shape
    if L_s <= 1:
        return torch.tensor(0.0, device=device_)

    inp = sample_seqs[:, :-1]
    target = sample_seqs[:, 1:]

    logits = model(images, inp)
    V = logits.size(-1)

    log_probs_all = F.log_softmax(logits, dim=-1)

    log_probs = log_probs_all.gather(2, target.unsqueeze(-1)).squeeze(-1)

    mask = (target != pad_id) & (target != end_id)
    log_probs = log_probs * mask

    log_probs_sum = log_probs.sum(dim=1)

    loss_rl = -(advantage.detach() * log_probs_sum).mean()

    return lambda_scst * loss_rl













def run_epoch(loader, model, optimizer=None, use_scst=False, scaler=None, use_amp=False):
    """
    loader: train_loader or val_loader
    optimizer: None -> eval 모드 (SCST도 off)
    use_scst: True일 때만 L1 항 추가 (그리고 training일 때만)
    scaler: GradScaler (AMP용, train일 때만 사용)
    use_amp: True면 autocast 사용
    """
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_tokens = 0
    total_correct = 0

    for imgs, caps in loader:
        imgs = imgs.to(device, non_blocking=True)
        caps = caps.to(device, non_blocking=True)

        y_inp = caps[:, :-1]
        y_tgt = caps[:, 1:]

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        if is_train and use_amp and scaler is not None:
            with autocast():
                logits = model(imgs, y_inp)
                logits = logits[:, : y_inp.size(1), :]

                B, Lm1, V = logits.shape

                ce_loss = criterion(
                    logits.reshape(B * Lm1, V),
                    y_tgt.reshape(B * Lm1),
                )

                total_loss_to_backprop = ce_loss

                if use_scst:
                    scst_loss = compute_scst_loss(
                        model,
                        imgs,
                        caps,
                        pad_id,
                        start_id,
                        end_id,
                        lambda_scst=LAMBDA_SCST,
                    )
                    total_loss_to_backprop = ce_loss + scst_loss
                else:
                    scst_loss = None
        else:
            logits = model(imgs, y_inp)
            logits = logits[:, : y_inp.size(1), :]

            B, Lm1, V = logits.shape

            ce_loss = criterion(
                logits.reshape(B * Lm1, V),
                y_tgt.reshape(B * Lm1),
            )

            total_loss_to_backprop = ce_loss

            if is_train and use_scst:
                scst_loss = compute_scst_loss(
                    model,
                    imgs,
                    caps,
                    pad_id,
                    start_id,
                    end_id,
                    lambda_scst=LAMBDA_SCST,
                )
                total_loss_to_backprop = ce_loss + scst_loss
            else:
                scst_loss = None

        if is_train:
            if use_amp and scaler is not None:
                scaler.scale(total_loss_to_backprop).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss_to_backprop.backward()
                optimizer.step()

        with torch.no_grad():
            preds = logits.argmax(-1)
            mask = (y_tgt != pad_id)
            correct = (preds == y_tgt) & mask
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()
            total_loss += ce_loss.item() * mask.sum().item()

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
patience = 1
no_improve = 0
total_start_time = time.time()







for epoch in range(1, EPOCHS + 1):
    print(f"\n===== Epoch {epoch}/{EPOCHS} =====")
    epoch_start_time = time.time()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    USE_AMP = (device.type == "cuda")

    train_loss, train_acc = run_epoch(
        train_loader,
        model,
        optimizer=optimizer,
        use_scst=USE_L1_SCST,
        scaler=scaler,
        use_amp=USE_AMP,
    )

    val_loss, val_acc = run_epoch(
        val_loader,
        model,
        optimizer=None,
        use_scst=False,
        scaler=None,
        use_amp=False,
    )

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

    print(
        f"[EPOCH {epoch}] "
        f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
        f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
    )
    print(
        f"[INFO] epoch_time={epoch_time:.2f} sec, "
        f"peak_gpu_mem={peak_mem_gb:.3f} GB, "
        f"total_time={total_train_time:.2f} sec"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve = 0
        torch.save(model.state_dict(), MODEL_WEIGHTS_PATH)
        print(f"[INFO] 🔥 Saved best model to: {MODEL_WEIGHTS_PATH}")
    else:
        no_improve += 1
        print(f"[INFO] No improvement for {no_improve} epoch(s) (patience={patience})")

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
plt.title("Loss over epochs (CE only)")
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
