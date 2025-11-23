# test_coco.py
import os
import glob
import json
import csv
import subprocess
from collections import defaultdict

import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
from html import escape
from tqdm import tqdm

from model import (
    CNN_Encoder,
    TransformerEncoder,
    TransformerDecoderLayer,
    ImageCaptioningModel,
)

# ==========
# 경로 / 설정
# ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_ROOT = os.path.join(BASE_DIR, "data", "coco2017")
IMG_DIR = os.path.join(DATA_ROOT, "val2017")
ANN_DIR = os.path.join(DATA_ROOT, "annotations")
ANN_PATH = os.path.join(ANN_DIR, "captions_val2017.json")

CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints", "two_layers_encoder")
MODEL_WEIGHTS_PREFIX = os.path.join(CHECKPOINT_DIR, "caption_model")
TOKENIZER_VOCAB_PATH = os.path.join(CHECKPOINT_DIR, "tokenizer_vocab.txt")

RESULTS_DIR = os.path.join(BASE_DIR, "results", "coco_inception")
os.makedirs(RESULTS_DIR, exist_ok=True)
JSON_PATH = os.path.join(RESULTS_DIR, "coco_test_preds.json")
CSV_PATH = os.path.join(RESULTS_DIR, "coco_test_preds.csv")
HTML_PATH = os.path.join(RESULTS_DIR, "coco_val2017_gallery.html")

MAX_LENGTH = 40
VOCABULARY_SIZE = 10000
BATCH_SIZE = 32
EMBEDDING_DIM = 512
IMAGE_FEATURE_DIM = 2048
UNITS = 512
NUM_ENCODER_LAYERS = 2
NUM_HEADS_ENCODER = 1
NUM_HEADS_DECODER = 8

NUM_IMAGES = 20  # 평가할 이미지 개수 (전체 쓰려면 None)

# =====================
# 1) 토크나이저 로딩
# =====================
print(f"Loading tokenizer vocab from: {TOKENIZER_VOCAB_PATH}")
with open(TOKENIZER_VOCAB_PATH, "r", encoding="utf-8") as f:
    vocab = [line.strip() for line in f if line.strip()]

tokenizer = tf.keras.layers.TextVectorization(
    max_tokens=VOCABULARY_SIZE,
    standardize=None,
    output_sequence_length=MAX_LENGTH,
    vocabulary=vocab,
)

word2idx = tf.keras.layers.StringLookup(
    mask_token="",
    vocabulary=vocab,
)
idx2word = tf.keras.layers.StringLookup(
    mask_token="",
    vocabulary=vocab,
    invert=True,
)

# ======================
# 2) 모델 구성 + 가중치 로드
# ======================
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
    image_aug=None,
)
caption_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))

print(f"Loading model weights from: {MODEL_WEIGHTS_PREFIX}")
caption_model.load_weights(MODEL_WEIGHTS_PREFIX).expect_partial()
print("Model weights loaded.")

# =====================
# 3) COCO 데이터 준비
# =====================
def count_jpg(d):
    return len(glob.glob(os.path.join(d, "*.jpg"))) if os.path.isdir(d) else 0


def ensure_coco_val2017():
    os.makedirs(DATA_ROOT, exist_ok=True)

    if count_jpg(IMG_DIR) == 0:
        print("⬇️ Downloading COCO val2017 images...")
        img_zip = os.path.join(DATA_ROOT, "val2017.zip")
        subprocess.run(
            f'wget -q http://images.cocodataset.org/zips/val2017.zip -O "{img_zip}"',
            shell=True,
            check=True,
        )
        subprocess.run(
            f'unzip -q "{img_zip}" -d "{DATA_ROOT}"',
            shell=True,
            check=True,
        )
        os.remove(img_zip)

    if not os.path.isfile(ANN_PATH):
        print("⬇️ Downloading COCO annotations (train/val 2017)...")
        ann_zip = os.path.join(DATA_ROOT, "annotations_trainval2017.zip")
        subprocess.run(
            f'wget -q http://images.cocodataset.org/annotations/annotations_trainval2017.zip '
            f'-O "{ann_zip}"',
            shell=True,
            check=True,
        )
        subprocess.run(
            f'unzip -q "{ann_zip}" -d "{DATA_ROOT}"',
            shell=True,
            check=True,
        )
        os.remove(ann_zip)

    n_imgs = count_jpg(IMG_DIR)
    assert n_imgs > 0, f"No images found in {IMG_DIR}"
    assert os.path.isfile(ANN_PATH), f"Missing {ANN_PATH}"
    print(f"✅ COCO val2017 ready: {n_imgs} images")
    print(f"✅ Annotations: {ANN_PATH}")


ensure_coco_val2017()

# =====================
# 4) caption 생성 함수
# =====================
def load_image_from_path(img_path):
    img = tf.io.read_file(img_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.keras.layers.Resizing(299, 299)(img)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img


def generate_caption(img_path):
    img = load_image_from_path(img_path)
    img = tf.expand_dims(img, axis=0)
    img_embed = caption_model.cnn_model(img, training=False)
    img_encoded = caption_model.encoder(img_embed, training=False)

    y_inp = "[start]"
    for i in range(MAX_LENGTH - 1):
        tokenized = tokenizer([y_inp])[:, :-1]
        mask = tf.cast(tokenized != 0, tf.int32)
        pred = caption_model.decoder(
            tokenized, img_encoded, training=False, mask=mask
        )

        pred_idx = np.argmax(pred[0, i, :])
        pred_word = idx2word(pred_idx).numpy().decode("utf-8")

        y_inp += " " + pred_word

        if pred_word == "[end]":
            break

    y_inp = y_inp.replace("[start]", "").replace("[end]", "").strip()
    return y_inp

# =====================
# 5) GT 불러오기 + 추론
# =====================
print("Loading COCO annotations...")
with open(ANN_PATH, "r", encoding="utf-8") as f:
    coco = json.load(f)
id2file = {img["id"]: img["file_name"] for img in coco["images"]}

gt_captions = defaultdict(list)
for ann in coco["annotations"]:
    fn = id2file.get(ann["image_id"])
    if fn:
        gt_captions[fn].append(ann["caption"])

test_image_paths = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")))
if NUM_IMAGES is not None:
    test_image_paths = test_image_paths[:NUM_IMAGES]

print(f"Running inference on {len(test_image_paths)} images...")

results = []
for img_path in tqdm(test_image_paths, desc="Generating captions"):
    fn = os.path.basename(img_path)
    try:
        pred = generate_caption(img_path)
    except Exception as e:
        pred = f"[Error: {e}]"
    results.append(
        {
            "filename": fn,
            "pred": pred,
            "gts": gt_captions.get(fn, []),
            "abs_path": img_path,
        }
    )

# =====================
# 6) JSON / CSV 저장
# =====================
with open(JSON_PATH, "w", encoding="utf-8") as f:
    json.dump({r["filename"]: r["pred"] for r in results}, f, indent=2, ensure_ascii=False)

with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "pred", "gt1", "gt2", "gt3", "gt4", "gt5", "abs_path"])
    for r in results:
        gts = (r["gts"] + [""] * 5)[:5]
        writer.writerow([r["filename"], r["pred"], *gts, r["abs_path"]])

print(f"✅ Saved JSON: {JSON_PATH}")
print(f"✅ Saved CSV : {CSV_PATH}")

# =====================
# 7) HTML 갤러리 저장
# =====================
def to_data_uri(img_path, max_h=180):
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        w, h = im.size
        if h > max_h:
            new_w = int(w * (max_h / float(h)))
            im = im.resize((max(1, new_w), max_h))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"


def make_html_from_results(items, title="COCO val2017 — Pred vs GT"):
    rows = []
    for it in items:
        fn = escape(it["filename"])
        pred = escape(it.get("pred", ""))
        gts = [escape(x) for x in it.get("gts", [])]

        try:
            src = to_data_uri(it["abs_path"], max_h=180)
            img_html = (
                '<img src="{src}" style="max-height:180px;object-fit:contain;'
                'border:1px solid #ddd;padding:4px;border-radius:8px;">'
            ).format(src=src)
        except Exception as e:
            img_html = f'<div style="color:#b00;">[이미지 로드 실패: {escape(str(e))}]</div>'

        gt_list_html = "".join([f"<li>{g}</li>" for g in gts])
        rows.append(
            f"""
        <div style="display:flex;gap:12px;align-items:flex-start;margin-bottom:18px;">
          {img_html}
          <div>
            <div style="font-weight:600">{fn}</div>
            <div style="margin-top:6px;"><b>Pred:</b> {pred}</div>
            <div style="margin-top:6px;"><b>GT x{len(gts)}:</b>
              <ul style="margin:6px 0 0 20px;">{gt_list_html}</ul>
            </div>
          </div>
        </div>"""
        )

    body = "\n".join(rows)
    return f"""<!doctype html><html><head><meta charset="utf-8">
    <title>{escape(title)}</title></head>
    <body style="font-family:system-ui,Arial,sans-serif;max-width:1000px;margin:32px auto;padding:0 16px;">
      <h1 style="margin-bottom:6px;">{escape(title)}</h1>
      <div style="color:#666;margin-bottom:24px;">총 {len(items)}장</div>
      {body}
    </body></html>"""


html_content = make_html_from_results(results)
with open(HTML_PATH, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"✅ HTML gallery saved to: {HTML_PATH}")
print("열어서 예측/GT 캡션 비교하면 됨.")
