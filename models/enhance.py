import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image


class ResidualBlock(nn.Module):
    def __init__(self, n_feats=256, kernel_size=3, res_scale=0.1):
        super().__init__()
        self.res_scale = res_scale
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size // 2),
        )

    def forward(self, x):
        return x + self.body(x) * self.res_scale


class EDSR(nn.Module):
    def __init__(self, n_colors=1, n_feats=256, n_resblocks=32, scale=4):
        super().__init__()
        self.scale = scale

        self.head = nn.Conv2d(n_colors, n_feats, 3, padding=1)

        self.body = nn.Sequential(*[
            ResidualBlock(n_feats) for _ in range(n_resblocks)
        ])
        self.body_tail = nn.Conv2d(n_feats, n_feats, 3, padding=1)

        self.upsampler = nn.Sequential(
            nn.Conv2d(n_feats, n_feats * (scale // 2), 3, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(n_feats, n_feats * (scale // 2), 3, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(n_feats, n_colors, 3, padding=1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res = self.body_tail(res)
        x = x + res
        x = self.upsampler(x)
        return x


class IRSuperResolution:
    def __init__(self, scale=4, device=None):
        self.scale = scale
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            self.model = EDSR(n_colors=1, n_feats=256, n_resblocks=32, scale=self.scale)
            state = torch.load(
                'checkpoints/edsr_base_4x.pt',
                map_location=self.device,
                weights_only=True,
            )
            self.model.load_state_dict(state)
            self.model.to(self.device)
            self.model.eval()
        except (FileNotFoundError, RuntimeError):
            self.model = None

    @staticmethod
    def preprocess(image: np.ndarray) -> np.ndarray:
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        denoised = cv2.medianBlur(image, 3)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        return enhanced

    def enhance(self, image: np.ndarray) -> np.ndarray:
        enhanced = self.preprocess(image)
        if self.model is not None:
            enhanced = self._super_resolve(enhanced)
        else:
            h, w = enhanced.shape[:2]
            enhanced = cv2.resize(enhanced, (w * self.scale, h * self.scale),
                                  interpolation=cv2.INTER_CUBIC)
            enhanced = cv2.detailEnhance(
                cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR),
                sigma_s=10, sigma_r=0.15
            )
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        return enhanced

    def _super_resolve(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        pad_h = (4 - h % 4) % 4
        pad_w = (4 - w % 4) % 4
        if pad_h or pad_w:
            image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect')

        tensor = torch.from_numpy(image.astype(np.float32) / 255.0)
        tensor = tensor.view(1, 1, *tensor.shape).to(self.device)

        with torch.no_grad():
            output = self.model(tensor)

        output = output.squeeze().cpu().numpy()
        output = np.clip(output * 255.0, 0, 255).astype(np.uint8)

        out_h = h * self.scale
        out_w = w * self.scale
        output = output[:out_h, :out_w]
        return output

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.enhance(image)
