# model.py
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tv_models


class CNNEncoder(nn.Module):
    """
    ResNet50 backbone.
    Input:  (B, 3, 299, 299)
    Output: (B, S, d_model)  where S = H*W
    """

    def __init__(self, d_model: int):
        super().__init__()
        #resnet50 을 imageNET weight 사용 (실제로 공식문서에서 imagenet 가중치 사용)
        base = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V1)
        # 마지막 conv block까지만 사용 (avgpool (공간 정보 없애는 애), fc 제거(클래시파이어여서 필요 없음, 쓰면 x))
        modules = list(base.children())[:-2]  # (B, 2048, H, W)
        # 남은 모듈들 이어붙여서 cnn encoder 로 사용하쟈
        self.cnn = nn.Sequential(*modules)
        #일단 2048크기인 걸 알아서 이렇게 해놨는데 동적으로 바뀌게 코드 바꿔야함 .
        self.conv_channels = 2048
#         self.cnn.eval()
# +        with torch.no_grad():
# +            dummy = torch.zeros(1, 3, 224, 224)
# +            conv_channels = self.cnn(dummy).shape[1]
# +        self.cnn.train()
# +
# +        self.conv_channels = conv_channels
        # d_model 차원으로 바꿔주는 거 
        self.proj = nn.Linear(self.conv_channels, d_model)

    # 순전파 코드
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, 299, 299)
        return: (B, S, d_model)
        """
        # feature map 얻기 
        feats = self.cnn(x)  # (B, C, H, W)
        B, C, H, W = feats.shape
        # flatten 작업 얘네는 이미지 관련
        feats = feats.view(B, C, H * W).permute(0, 2, 1)  # (B, S, C)
        # d_model 차원으로 바꿔주기
        return self.proj(feats)  # (B, S, d_model)

class CNNEncoder_efficientNet(nn.Module):
    """
    EfficientNet backbone 기반 CNN 인코더.

    Input:  (B, 3, H, W)  - 학습 시에는 (B, 3, 299, 299)로 리사이즈해서 넣고 있음
    Output: (B, S, d_model)  where S = H_out * W_out (공간 위치 개수)

    - EfficientNet-B0 기준:
      입력이 299x299이면 대략 H_out, W_out ≈ 10x10 정도의 feature map이 나와서
      S ≈ 100개의 토큰이 생김.
    """

    def __init__(self, d_model: int, backbone: str = "b0", pretrained: bool = True):
        super().__init__()

        # 1) EfficientNet backbone 선택
        if backbone == "b0":
            weights = tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            eff = tv_models.efficientnet_b0(weights=weights)
        elif backbone == "b1":
            weights = tv_models.EfficientNet_B1_Weights.IMAGENET1K_V1 if pretrained else None
            eff = tv_models.efficientnet_b1(weights=weights)
        elif backbone == "b2":
            weights = tv_models.EfficientNet_B2_Weights.IMAGENET1K_V1 if pretrained else None
            eff = tv_models.efficientnet_b2(weights=weights)
        elif backbone == "b3":
            weights = tv_models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
            eff = tv_models.efficientnet_b3(weights=weights)
        elif backbone == "b4":
            weights = tv_models.EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
            eff = tv_models.efficientnet_b4(weights=weights)
        elif backbone == "b5":
            weights = tv_models.EfficientNet_B5_Weights.IMAGENET1K_V1 if pretrained else None
            eff = tv_models.efficientnet_b5(weights=weights)
        elif backbone == "b6":
            weights = tv_models.EfficientNet_B6_Weights.IMAGENET1K_V1 if pretrained else None
            eff = tv_models.efficientnet_b6(weights=weights)
        elif backbone == "b7":
            weights = tv_models.EfficientNet_B7_Weights.IMAGENET1K_V1 if pretrained else None
            eff = tv_models.efficientnet_b7(weights=weights)
        else:
            raise ValueError(f"Unknown EfficientNet backbone: {backbone}")

        # 2) classifier 부분(fc 등)은 버리고, feature 추출기만 사용
        #    torchvision EfficientNet은 .features 가 conv stack임
        self.cnn = eff.features  # (B, C, H_out, W_out)

        # 3) 출력 채널 수(conv_channels)를 동적으로 추론
        #    (resnet 처럼 2048 고정 상수 쓰지 않고, backbone 바뀌어도 자동 대응)
        with torch.no_grad():
            self.cnn.eval()
            dummy = torch.zeros(1, 3, 299, 299)  # 학습에서 쓰는 입력 크기 기준
            out = self.cnn(dummy)
            conv_channels = out.shape[1]  # C
        self.conv_channels = conv_channels

        # 4) 최종적으로 d_model 차원으로 projection
        self.proj = nn.Linear(self.conv_channels, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, H, W)
        return: (B, S, d_model)  where S = H_out * W_out
        """
        feats = self.cnn(x)           # (B, C, H_out, W_out)
        B, C, H, W = feats.shape

        # (B, C, H, W) → (B, S, C) : S = H * W
        feats = feats.view(B, C, H * W).permute(0, 2, 1)  # (B, S, C)

        # (B, S, C) → (B, S, d_model)
        return self.proj(feats)
class PositionalEncoding(nn.Module):
    """
    그 위치 정보 알려주기!!
    짝수 , 홀수마다 뭔 sin, cos 으로 처리해서 위치정보를 알려줌,,,
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (L, D)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (L, 1) 
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, L, D)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, D)
        """
        L = x.size(1)
        return x + self.pe[:, :L, :]

class CNNEncoder_inceptionV3(nn.Module):
    """
    Inception-V3 backbone.
    Input:  (B, 3, 299, 299)
    Output: (B, S, d_model),  S = H_out * W_out
    """

    def __init__(self, d_model: int, pretrained: bool = True):
        super().__init__()

        weights = (
            tv_models.Inception_V3_Weights.IMAGENET1K_V1
            if pretrained else None
        )

        # Inception-V3 전체 모델 로드
        base = tv_models.inception_v3(weights=weights)

        # feature extractor 부분만 사용하기 위해 classifier 제거
        # Inception 구조에서 중요한 건 Mixed_7c 출력 (2048채널, 8x8 정도)
        self.cnn = nn.Sequential(
            base.Conv2d_1a_3x3,
            base.Conv2d_2a_3x3,
            base.Conv2d_2b_3x3,
            nn.MaxPool2d(3, stride=2),
            base.Conv2d_3b_1x1,
            base.Conv2d_4a_3x3,
            nn.MaxPool2d(3, stride=2),
            base.Mixed_5b,
            base.Mixed_5c,
            base.Mixed_5d,
            base.Mixed_6a,
            base.Mixed_6b,
            base.Mixed_6c,
            base.Mixed_6d,
            base.Mixed_6e,
            base.Mixed_7a,
            base.Mixed_7b,
            base.Mixed_7c,     # 최종 conv feature
        )

        # 출력 채널을 동적으로 추론
        with torch.no_grad():
            self.cnn.eval()
            dummy = torch.zeros(1, 3, 299, 299)
            out = self.cnn(dummy)
            conv_channels = out.shape[1]
        self.conv_channels = conv_channels   # 보통 2048

        # d_model로 projection
        self.proj = nn.Linear(self.conv_channels, d_model)

    def forward(self, x):
        """
        x: (B, 3, 299, 299)
        return: (B, S, d_model)
        """
        feats = self.cnn(x)   # (B, C, H, W)
        B, C, H, W = feats.shape

        # Flatten: (B, C, H, W) → (B, S, C)
        feats = feats.view(B, C, H*W).permute(0, 2, 1)

        # Projection: (B, S, C) → (B, S, d_model)
        return self.proj(feats)

class ImageCaptioningModel_Resnet(nn.Module):
    """
    CNN(ResNet50) + TransformerEncoder + TransformerDecoder

    - 이미지: CNNEncoder -> TransformerEncoder (memory)
    - 텍스트: Embedding + PosEnc -> TransformerDecoder (tgt)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads_enc: int,
        n_layers_enc: int,
        n_heads_dec: int,
        n_layers_dec: int,
        dim_ff: int,
        max_len: int,
        pad_id: int,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_id = pad_id

        # 1) CNN 인코더로 encoding
        self.cnn_encoder = CNNEncoder(d_model=d_model)

        # 2) TransformerEncoder layer 1개
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads_enc,
            dim_feedforward=dim_ff, # fnn할 때 몇배 될건지 결정
            batch_first=True,
        )
        # layer num_layers 만큼 쌓을 수 있게!!
        self.img_encoder = nn.TransformerEncoder(
            enc_layer, num_layers=n_layers_enc
        )

        # 3) 텍스트 임베딩 -> 토큰 크기 d_model 로 바꿔주기  (flaten , projection )
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id) 
        # 위치 정보 알려주기 (위에 선언함)
        self.pos_encoding = PositionalEncoding(d_model=d_model, max_len=max_len)

        # TransformerDecoder layer 1개
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads_dec,
            dim_feedforward=dim_ff,
            batch_first=True,
        )
        # 여러개 
        self.decoder = nn.TransformerDecoder(
            dec_layer, num_layers=n_layers_dec
        )

        # 5) 출력 projection
        self.output_proj = nn.Linear(d_model, vocab_size)

    # 디코더에서 masked self attention 하라고 마스크 만들어주는거 ! 미래 못보게! 
    def make_subsequent_mask(self, size: int, device) -> torch.Tensor:
        """
        causal mask for decoder.
        shape: (T, T)
        """
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

    def forward(
        self, images: torch.Tensor, captions_in: torch.Tensor
    ) -> torch.Tensor:
        """
        images:     (B, 3, 299, 299)
        captions_in:(B, L)  -- [start] ... [second-last token]
        return:     (B, L, vocab_size)
        """
        device = images.device # 이미지 텐서 있는 장치 가져오기

        # 1) 이미지 인코딩 ( 아까 만들어둔 cnnencoder -> transformer encoder )
        src = self.cnn_encoder(images)       # (B, S, d_model)
        memory = self.img_encoder(src)       # (B, S, d_model) #

        # 2) 텍스트 임베딩 + 포지셔널 인코딩
        tgt_emb = self.token_embedding(captions_in)  # (B, L, D)
        tgt_emb = self.pos_encoding(tgt_emb)         # (B, L, D)

        # 3) 마스크 생성
        B, L = captions_in.shape
        tgt_mask = self.make_subsequent_mask(L, device=device)  # (L, L)
        tgt_key_padding_mask = (captions_in == self.pad_id)     # (B, L)

        # 4) 디코딩
        dec_out = self.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )  # (B, L, D)

        # 5) 로짓/ L : 입력 길이, v : 어휘 수 (어떤 포인트에 어떤 어휘 점수가 높은지 ? )
        logits = self.output_proj(dec_out)  # (B, L, V)
        return logits
class ImageCaptioningModel_EfficientNet(nn.Module):
    """
    CNN(ResNet50) + TransformerEncoder + TransformerDecoder

    - 이미지: CNNEncoder -> TransformerEncoder (memory)
    - 텍스트: Embedding + PosEnc -> TransformerDecoder (tgt)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads_enc: int,
        n_layers_enc: int,
        n_heads_dec: int,
        n_layers_dec: int,
        dim_ff: int,
        max_len: int,
        pad_id: int,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_id = pad_id

        # 1) CNN 인코더로 encoding
        self.cnn_encoder = CNNEncoder_efficientNet(d_model=d_model , backbone="b4")

        # 2) TransformerEncoder layer 1개
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads_enc,
            dim_feedforward=dim_ff, # fnn할 때 몇배 될건지 결정
            batch_first=True,
        )
        # layer num_layers 만큼 쌓을 수 있게!!
        self.img_encoder = nn.TransformerEncoder(
            enc_layer, num_layers=n_layers_enc
        )

        # 3) 텍스트 임베딩 -> 토큰 크기 d_model 로 바꿔주기  (flaten , projection )
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id) 
        # 위치 정보 알려주기 (위에 선언함)
        self.pos_encoding = PositionalEncoding(d_model=d_model, max_len=max_len)

        # TransformerDecoder layer 1개
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads_dec,
            dim_feedforward=dim_ff,
            batch_first=True,
        )
        # 여러개 
        self.decoder = nn.TransformerDecoder(
            dec_layer, num_layers=n_layers_dec
        )

        # 5) 출력 projection
        self.output_proj = nn.Linear(d_model, vocab_size)

    # 디코더에서 masked self attention 하라고 마스크 만들어주는거 ! 미래 못보게! 
    def make_subsequent_mask(self, size: int, device) -> torch.Tensor:
        """
        causal mask for decoder.
        shape: (T, T)
        """
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

    def forward(
        self, images: torch.Tensor, captions_in: torch.Tensor
    ) -> torch.Tensor:
        """
        images:     (B, 3, 299, 299)
        captions_in:(B, L)  -- [start] ... [second-last token]
        return:     (B, L, vocab_size)
        """
        device = images.device # 이미지 텐서 있는 장치 가져오기

        # 1) 이미지 인코딩 ( 아까 만들어둔 cnnencoder -> transformer encoder )
        src = self.cnn_encoder(images)       # (B, S, d_model)
        memory = self.img_encoder(src)       # (B, S, d_model) #

        # 2) 텍스트 임베딩 + 포지셔널 인코딩
        tgt_emb = self.token_embedding(captions_in)  # (B, L, D)
        tgt_emb = self.pos_encoding(tgt_emb)         # (B, L, D)

        # 3) 마스크 생성
        B, L = captions_in.shape
        tgt_mask = self.make_subsequent_mask(L, device=device)  # (L, L)
        tgt_key_padding_mask = (captions_in == self.pad_id)     # (B, L)

        # 4) 디코딩
        dec_out = self.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )  # (B, L, D)

        # 5) 로짓/ L : 입력 길이, v : 어휘 수 (어떤 포인트에 어떤 어휘 점수가 높은지 ? )
        logits = self.output_proj(dec_out)  # (B, L, V)
        return logits

class ImageCaptioningModel_InceptionV3(nn.Module):
    """
    CNN(ResNet50) + TransformerEncoder + TransformerDecoder

    - 이미지: CNNEncoder -> TransformerEncoder (memory)
    - 텍스트: Embedding + PosEnc -> TransformerDecoder (tgt)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads_enc: int,
        n_layers_enc: int,
        n_heads_dec: int,
        n_layers_dec: int,
        dim_ff: int,
        max_len: int,
        pad_id: int,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_id = pad_id

        # 1) CNN 인코더로 encoding
        self.cnn_encoder = CNNEncoder_inceptionV3(d_model=d_model)

        # 2) TransformerEncoder layer 1개
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads_enc,
            dim_feedforward=dim_ff, # fnn할 때 몇배 될건지 결정
            batch_first=True,
        )
        # layer num_layers 만큼 쌓을 수 있게!!
        self.img_encoder = nn.TransformerEncoder(
            enc_layer, num_layers=n_layers_enc
        )

        # 3) 텍스트 임베딩 -> 토큰 크기 d_model 로 바꿔주기  (flaten , projection )
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id) 
        # 위치 정보 알려주기 (위에 선언함)
        self.pos_encoding = PositionalEncoding(d_model=d_model, max_len=max_len)

        # TransformerDecoder layer 1개
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads_dec,
            dim_feedforward=dim_ff,
            batch_first=True,
        )
        # 여러개 
        self.decoder = nn.TransformerDecoder(
            dec_layer, num_layers=n_layers_dec
        )

        # 5) 출력 projection
        self.output_proj = nn.Linear(d_model, vocab_size)

    # 디코더에서 masked self attention 하라고 마스크 만들어주는거 ! 미래 못보게! 
    def make_subsequent_mask(self, size: int, device) -> torch.Tensor:
        """
        causal mask for decoder.
        shape: (T, T)
        """
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

    def forward(
        self, images: torch.Tensor, captions_in: torch.Tensor, return_aux: bool= False,
    ) -> torch.Tensor:
        """
        images:     (B, 3, 299, 299)
        captions_in:(B, L)  -- [start] ... [second-last token]
        return:     (B, L, vocab_size)
        """
        device = images.device # 이미지 텐서 있는 장치 가져오기

        # 1) 이미지 인코딩 ( 아까 만들어둔 cnnencoder -> transformer encoder )
        src = self.cnn_encoder(images)       # (B, S, d_model)
        memory = self.img_encoder(src)       # (B, S, d_model) #

        # 2) 텍스트 임베딩 + 포지셔널 인코딩
        tgt_emb = self.token_embedding(captions_in)  # (B, L, D)
        tgt_emb = self.pos_encoding(tgt_emb)         # (B, L, D)

        # 3) 마스크 생성
        B, L = captions_in.shape
        tgt_mask = self.make_subsequent_mask(L, device=device)  # (L, L)
        tgt_key_padding_mask = (captions_in == self.pad_id)     # (B, L)

        # 4) 디코딩
        dec_out = self.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )  # (B, L, D)
        logits = self.output_proj(dec_out)  # (B, L, V)
        if not return_aux:
            return logits

        # ---- image / caption global embedding 계산 (loss용) ----
        # 이미지: 토큰 차원(S) 평균
        img_global = memory.mean(dim=1)   # (B, d_model)

        # 캡션: PAD는 빼고 평균
        non_pad_mask = (captions_in != self.pad_id).unsqueeze(-1)  # (B, L, 1)
        masked_dec = dec_out * non_pad_mask
        lengths = non_pad_mask.sum(dim=1).clamp(min=1)             # (B,1)
        cap_global = (masked_dec.sum(dim=1) / lengths)             # (B, d_model)

        return logits, img_global, cap_global
        # 5) 로짓/ L : 입력 길이, v : 어휘 수 (어떤 포인트에 어떤 어휘 점수가 높은지 ? )
        
        

class ImageCaptioningModel_EfficientNet_New(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads_enc: int,
        n_layers_enc: int,
        n_heads_dec: int,
        n_layers_dec: int,
        dim_ff: int,
        max_len: int,
        pad_id: int,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_id = pad_id

        # ----- CNN Encoder -----
        self.cnn_encoder = CNNEncoder_efficientNet(d_model=d_model, backbone="b4")

        # ----- Image Transformer Encoder -----
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads_enc,
            dim_feedforward=dim_ff,
            batch_first=True,
        )
        self.img_encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers_enc)

        # ----- Text Embedding -----
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_encoding = PositionalEncoding(d_model=d_model, max_len=max_len)

        # ----- Decoder -----
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads_dec,
            dim_feedforward=dim_ff,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=n_layers_dec)

        # ----- Output -----
        self.output_proj = nn.Linear(d_model, vocab_size)

    def make_subsequent_mask(self, size: int, device) -> torch.Tensor:
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float("-inf"))

    def forward(
        self,
        images: torch.Tensor,
        captions_in: torch.Tensor,
        return_attn: bool = False,
    ):
        device = images.device

        # 1) 이미지 인코딩
        src = self.cnn_encoder(images)       # (B, S, d_model)
        memory = self.img_encoder(src)       # (B, S, d_model)

        # 2) 텍스트 인코딩
        tgt_emb = self.token_embedding(captions_in)  # (B, L, D)
        tgt_emb = self.pos_encoding(tgt_emb)         # (B, L, D)

        B, L = captions_in.shape
        tgt_mask = self.make_subsequent_mask(L, device=device)
        tgt_key_padding_mask = (captions_in == self.pad_id)

        # 3) 디코딩
        dec_out = self.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )  # (B, L, D)

        logits = self.output_proj(dec_out)   # (B, L, V)

        if not return_attn:
            return logits

        # 4) 학생 cross-attention 비슷한 거 직접 계산
        # dec_out: (B, L, D), memory: (B, S, D)
        # → 유사도: (B, L, S)
        attn_logits = torch.einsum("bld,bsd->bls", dec_out, memory)  # (B,L,S)
        student_attn = attn_logits.softmax(dim=-1)                   # (B,L,S)

        return logits, student_attn


class ImageCaptioningModel_EfficientNet_clip(nn.Module):
    """
    CNN(EfficientNet) + TransformerEncoder + TransformerDecoder
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads_enc: int,
        n_layers_enc: int,
        n_heads_dec: int,
        n_layers_dec: int,
        dim_ff: int,
        max_len: int,
        pad_id: int,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_id = pad_id

        # 1) EfficientNet encoder
        self.cnn_encoder = CNNEncoder_efficientNet(d_model=d_model, backbone="b4")

        # 2) TransformerEncoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads_enc,
            dim_feedforward=dim_ff,
            batch_first=True,
        )
        self.img_encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers_enc)

        # 3) 텍스트 임베딩 + 포지셔널 인코딩
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_encoding = PositionalEncoding(d_model=d_model, max_len=max_len)

        # 4) TransformerDecoder
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads_dec,
            dim_feedforward=dim_ff,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_layers_dec)

        # 5) output projection
        self.output_proj = nn.Linear(d_model, vocab_size)

    def make_subsequent_mask(self, size: int, device) -> torch.Tensor:
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

    def forward(
        self,
        images: torch.Tensor,
        captions_in: torch.Tensor,
        return_img_feat: bool = False,   # ★ 추가
    ):
        """
        images: (B, 3, 299, 299)
        captions_in: (B, L)  -- [start] ... (end-1)
        return:
          - 기본: logits (B, L, vocab_size)
          - return_img_feat=True: (logits, img_global)  where img_global: (B, d_model)
        """
        device = images.device

        # 1) 이미지 인코딩
        src = self.cnn_encoder(images)      # (B, S, D)
        memory = self.img_encoder(src)      # (B, S, D)

        # ★ CLIP distillation용 global feature: 공간 위치 평균
        img_global = memory.mean(dim=1)     # (B, D)

        # 2) 텍스트 쪽
        tgt_emb = self.token_embedding(captions_in)   # (B, L, D)
        tgt_emb = self.pos_encoding(tgt_emb)          # (B, L, D)

        B, L = captions_in.shape
        tgt_mask = self.make_subsequent_mask(L, device=device)      # (L, L)
        tgt_key_padding_mask = (captions_in == self.pad_id)         # (B, L)

        # 3) 디코딩
        dec_out = self.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )  # (B, L, D)

        logits = self.output_proj(dec_out)  # (B, L, V)

        if return_img_feat:
            return logits, img_global
        return logits
