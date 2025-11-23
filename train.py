# train.py
import os
import json
import random
import re

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")  # SSH 환경에서 GUI 없이 PNG 저장
import matplotlib.pyplot as plt

from model import (
    CNN_Encoder,
    TransformerEncoder,
    TransformerDecoderLayer,
    ImageCaptioningModel,
)

# =========================
# 경로 / 하이퍼파라미터 설정
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "flickr8k")
CAPTIONS_PATH = os.path.join(DATA_DIR, "captions.txt")
IMAGES_DIR = os.path.join(DATA_DIR, "Images")

CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints", "two_layers_encoder")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

HISTORY_JSON_PATH = os.path.join(CHECKPOINT_DIR, "history.json")
HISTORY_PLOT_PATH = os.path.join(CHECKPOINT_DIR, "history.png")
TOKENIZER_VOCAB_PATH = os.path.join(CHECKPOINT_DIR, "tokenizer_vocab.txt")
MODEL_WEIGHTS_PREFIX = os.path.join(CHECKPOINT_DIR, "caption_model")

MAX_LENGTH = 40
VOCABULARY_SIZE = 10000
BATCH_SIZE = 32
BUFFER_SIZE = 1000
EMBEDDING_DIM = 512
IMAGE_FEATURE_DIM = 2048
UNITS = 512
NUM_ENCODER_LAYERS = 2
NUM_HEADS_ENCODER = 1
NUM_HEADS_DECODER = 8
EPOCHS = 10

# ================
# 데이터 준비
# ================
def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = text.strip()
    text = "[start] " + text + " [end]"
    return text


print(f"Loading captions from: {CAPTIONS_PATH}")
captions = pd.read_csv(CAPTIONS_PATH)
captions["image"] = captions["image"].apply(
    lambda x: os.path.join(IMAGES_DIR, x)
)
captions["caption"] = captions["caption"].apply(preprocess)

# 이미지 경로 → 여러 캡션 매핑
img_to_cap_vector = {}
for img, cap in zip(captions["image"], captions["caption"]):
    img_to_cap_vector.setdefault(img, []).append(cap)

img_keys = list(img_to_cap_vector.keys())
random.shuffle(img_keys)

slice_index = int(len(img_keys) * 0.9)
img_name_train_keys = img_keys[:slice_index]
img_name_val_keys = img_keys[slice_index:]

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

print(f"# train samples: {len(train_imgs)}")
print(f"# val samples  : {len(val_imgs)}")

# =================
# 토크나이저 준비
# =================
tokenizer = tf.keras.layers.TextVectorization(
    max_tokens=VOCABULARY_SIZE,
    standardize=None,
    output_sequence_length=MAX_LENGTH,
)
tokenizer.adapt(train_captions)

vocab = tokenizer.get_vocabulary()

# 저장해두면 test_coco.py 에서 재사용 가능
with open(TOKENIZER_VOCAB_PATH, "w", encoding="utf-8") as f:
    for token in vocab:
        f.write(token + "\n")
print(f"Saved tokenizer vocab to: {TOKENIZER_VOCAB_PATH}")

word2idx = tf.keras.layers.StringLookup(
    mask_token="",
    vocabulary=vocab,
)
idx2word = tf.keras.layers.StringLookup(
    mask_token="",
    vocabulary=vocab,
    invert=True,
)

# ============
# TF Dataset
# ============
def load_data(img_path, caption):
    img = tf.io.read_file(img_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.keras.layers.Resizing(299, 299)(img)
    img = tf.image.convert_image_dtype(img, tf.float32)
    caption = tokenizer(caption)
    return img, caption


train_dataset = tf.data.Dataset.from_tensor_slices((train_imgs, train_captions))
train_dataset = (
    train_dataset
    .shuffle(BUFFER_SIZE)
    .map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

val_dataset = tf.data.Dataset.from_tensor_slices((val_imgs, val_captions))
val_dataset = (
    val_dataset
    .shuffle(BUFFER_SIZE)
    .map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

# ===========
# 이미지 증강
# ===========
image_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomContrast(0.3),
    ]
)

# ===========
# 모델 구성
# ===========
cnn_model = CNN_Encoder()
encoder = TransformerEncoder(
    embed_dim=IMAGE_FEATURE_DIM,
    num_heads=NUM_HEADS_ENCODER,
    num_layers=NUM_ENCODER_LAYERS,
)

decoder = TransformerDecoderLayer(
    vocab_size=len(vocab),
    embed_dim=EMBEDDING_DIM,
    units=UNITS,
    num_heads=NUM_HEADS_DECODER,
    max_len=MAX_LENGTH,
    image_feature_dim=IMAGE_FEATURE_DIM,
)

cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=False,
    reduction="none",
)

caption_model = ImageCaptioningModel(
    cnn_model=cnn_model,
    encoder=encoder,
    decoder=decoder,
    loss_fn=cross_entropy,
    image_aug=image_augmentation,
)

caption_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    patience=3,
    restore_best_weights=True,
    monitor="val_loss",
)

# ===========================
# 기존 체크포인트가 있으면 로드
# ===========================
if tf.io.gfile.glob(MODEL_WEIGHTS_PREFIX + "*"):
    print(f"Loading existing weights from: {MODEL_WEIGHTS_PREFIX}")
    caption_model.load_weights(MODEL_WEIGHTS_PREFIX).expect_partial()
else:
    print("No existing weights found. Training from scratch.")

# ==========
# 학습 실행
# ==========
history = caption_model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=val_dataset,
    callbacks=[early_stopping],
)

# ======================
# 1) history JSON 저장
# ======================
history_dict = history.history
with open(HISTORY_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(history_dict, f, indent=4)
print(f"Saved training history to: {HISTORY_JSON_PATH}")

# ======================
# 2) 모델 체크포인트 저장
# ======================
caption_model.save_weights(MODEL_WEIGHTS_PREFIX)
print(f"Saved model weights to prefix: {MODEL_WEIGHTS_PREFIX}")

# ======================
# 3) history 그래프 PNG 저장
# ======================
plt.figure(figsize=(12, 4))

# Loss
plt.subplot(1, 2, 1)
plt.plot(history_dict.get("loss", []), label="train_loss")
if "val_loss" in history_dict:
    plt.plot(history_dict["val_loss"], label="val_loss")
plt.title("Loss over epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(history_dict.get("accuracy", []), label="train_acc")
if "val_accuracy" in history_dict:
    plt.plot(history_dict["val_accuracy"], label="val_acc")
plt.title("Accuracy over epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(HISTORY_PLOT_PATH)
print(f"Saved training curves PNG to: {HISTORY_PLOT_PATH}")

# ======================
# 마지막 loss / acc 출력
# ======================
def last_or_none(lst, key):
    arr = lst.get(key)
    return arr[-1] if arr else None

final_train_loss = last_or_none(history_dict, "loss")
final_val_loss = last_or_none(history_dict, "val_loss")
final_train_acc = last_or_none(history_dict, "accuracy")
final_val_acc = last_or_none(history_dict, "val_accuracy")

print("===== Final Metrics =====")
print(f"Train loss: {final_train_loss}")
print(f"Val   loss: {final_val_loss}")
print(f"Train acc : {final_train_acc}")
print(f"Val   acc : {final_val_acc}")
