"""DeepSTARR oracle loader + model (vendored from de Almeida et al. 2022).

Oracle for fly enhancer activity: predicts 2 scalar outputs (dev, hk) from 249bp
one-hot inputs. Source architecture:
  D3-DNA-Discrete-Diffusion/model_zoo/deepstarr/deepstarr.py
"""

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


INPUT_LEN = 249
N_FEATURES = 2


class DeepSTARR(nn.Module):
    """DeepSTARR CNN (de Almeida et al., 2022). 4 conv blocks + 2 FC -> output_dim."""

    def __init__(self, output_dim, d=256):
        super().__init__()
        self.activation = nn.ReLU()
        self.dropout4 = nn.Dropout(0.4)
        self.flatten = nn.Flatten()

        self.conv1_filters = nn.Parameter(torch.zeros(d, 4, 7))
        nn.init.kaiming_normal_(self.conv1_filters)
        self.batchnorm1 = nn.BatchNorm1d(d)
        self.activation1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(2)

        self.conv2_filters = nn.Parameter(torch.zeros(60, d, 3))
        nn.init.kaiming_normal_(self.conv2_filters)
        self.batchnorm2 = nn.BatchNorm1d(60)
        self.maxpool2 = nn.MaxPool1d(2)

        self.conv3_filters = nn.Parameter(torch.zeros(60, 60, 5))
        nn.init.kaiming_normal_(self.conv3_filters)
        self.batchnorm3 = nn.BatchNorm1d(60)
        self.maxpool3 = nn.MaxPool1d(2)

        self.conv4_filters = nn.Parameter(torch.zeros(120, 60, 3))
        nn.init.kaiming_normal_(self.conv4_filters)
        self.batchnorm4 = nn.BatchNorm1d(120)
        self.maxpool4 = nn.MaxPool1d(2)

        self.fc5 = nn.LazyLinear(256, bias=True)
        self.batchnorm5 = nn.BatchNorm1d(256)
        self.fc6 = nn.Linear(256, 256, bias=True)
        self.batchnorm6 = nn.BatchNorm1d(256)
        self.fc7 = nn.Linear(256, output_dim)

    def forward(self, x):
        cnn = torch.conv1d(x, self.conv1_filters, stride=1, padding="same")
        cnn = self.maxpool1(self.activation1(self.batchnorm1(cnn)))

        cnn = torch.conv1d(cnn, self.conv2_filters, stride=1, padding="same")
        cnn = self.maxpool2(self.activation(self.batchnorm2(cnn)))

        cnn = torch.conv1d(cnn, self.conv3_filters, stride=1, padding="same")
        cnn = self.maxpool3(self.activation(self.batchnorm3(cnn)))

        cnn = torch.conv1d(cnn, self.conv4_filters, stride=1, padding="same")
        cnn = self.maxpool4(self.activation(self.batchnorm4(cnn)))

        cnn = self.flatten(cnn)
        cnn = self.dropout4(self.activation(self.batchnorm5(self.fc5(cnn))))
        cnn = self.dropout4(self.activation(self.batchnorm6(self.fc6(cnn))))
        return self.fc7(cnn)


class DeepStarrOracle:
    """DeepSTARR oracle: predicts (N, 2) [dev, hk] activity from (N, 4, 249) one-hot."""

    def __init__(self, model, device):
        self.model = model
        self.device = device

    @torch.no_grad()
    def predict(self, x, batch_size=128, progress=True):
        x = np.asarray(x, dtype=np.float32)
        if x.shape[-1] != INPUT_LEN:
            raise ValueError(f"DeepSTARR expects L={INPUT_LEN}, got {x.shape[-1]}")
        out = []
        it = range(0, len(x), batch_size)
        if progress:
            it = tqdm(it, desc="DeepSTARR predict")
        for i in it:
            b = torch.tensor(x[i : i + batch_size], dtype=torch.float32, device=self.device)
            out.append(self.model(b).cpu().numpy())
        return np.concatenate(out, axis=0)


def _strip_prefix(sd, prefix="model."):
    pref_len = len(prefix)
    return {k[pref_len:] if k.startswith(prefix) else k: v for k, v in sd.items()}


def load(checkpoint_path, device):
    model = DeepSTARR(output_dim=N_FEATURES)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    sd = _strip_prefix(sd, "model.")
    # Warm up LazyLinear (needs batch>=2 in train-mode BN; use eval)
    model.eval()
    with torch.no_grad():
        model(torch.zeros(2, 4, INPUT_LEN))
    model.load_state_dict(sd, strict=False)
    return DeepStarrOracle(model.to(device).eval(), device)
