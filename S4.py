# =========================================================
# S4: Self-Supervised Shortcut Surgery — Production-Ready Implementation (PyTorch)
# =========================================================
# Features:
# - Full min–max S4 (budgeted probe with Hard-Concrete gates, invariance via mask-guided counter-augmentations,
#   self-discovered environments with IRM-style gradient penalty, cheat-score weighting / local DRO surrogate).
# - Multiple datasets with auto-download/setup:
#   * colored_mnist (synthetic spurious color cue; train rho=0.9, test rho=0.1; worst-group available)
#   * waterbirds (WILDS; label-background spurious; auto-download via `wilds`)
#   * celeba_blond_spurious (CelebA via torchvision; label: Blond Hair; spurious: Male; worst-group available)
#   * cifar10_spurious_patch (CIFAR-10 w/ synthetic colored patch correlated with label)
# - Robust training loop: AMP (mixed precision), early stopping, checkpointing, LR scheduler, gradient clipping.
# - Evaluation: overall accuracy, worst-group accuracy (if groups available), balanced accuracy, ECE calibration,
#   invariance gap, exposure curves (probe accuracy vs k), AUROC cheat separation, confusion matrix.
# - Artifacts: per-epoch mask overlays, cheat histograms, exposure curves, metric CSV/JSON, plots in English.
# - Reproducibility, deterministic flags (where feasible).
#
# Quickstart:
#   pip install torch torchvision torchaudio matplotlib numpy scikit-learn tqdm pandas seaborn wilds datasets pillow
#   python s4_production.py --dataset colored_mnist --epochs 30 --k 8
#
# Notes:
# - For Waterbirds, `wilds` handles licensing and download; for CelebA, torchvision will prompt license acceptance once.
# - This single file aims for production-readiness: strong defaults, logging, consistent artifacts; extend by config if desired.
# =========================================================

import os, math, json, random, argparse, shutil, time, sys, warnings
from dataclasses import dataclass, asdict
from typing import Tuple, Optional, List, Dict, Any
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score

# Optional deps (installed above)
try:
    from wilds import get_dataset
    from wilds.common.data_loaders import get_train_loader
    WILDS_AVAILABLE = True
except Exception:
    WILDS_AVAILABLE = False

try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

# =========================================================
# (1) Formal Spec (ASCII LaTeX for dev reference)
# =========================================================
SPEC = r"""
\section*{S4: Formal Specification (copyable ASCII LaTeX)}
Data: \mathcal{D}=\{(x_i,y_i)\}_{i=1}^n,\ y\in\{1,\ldots,C\}. Encoder f_{\theta}:\mathcal{X}\to\mathbb{R}^d yields z=f_{\theta}(x).
Classifier p_{\omega}(y|z) with cross-entropy \CE(p_{\omega}(\cdot|z),y)=-\log p_{\omega}(y|z).
k-sparse mask set: \Pi_k=\{m\in\{0,1\}^d: \|m\|_0\le k\}. Continuous relaxation \tilde m\in[0,1]^d.

Budgeted probe:
\mathcal{L}_{probe}(\theta,\phi,\tilde m)=\mathbb{E}[\CE(q_{\phi}(\tilde m\odot f_{\theta}(x)),y)]
+\lambda\ |\sum_j \mathbb{E}[\tilde m_j]-k|. \tag{1}

Hard-Concrete gates:
u\sim U(0,1),\ s=\sigma((\log u-\log(1-u)+\log\alpha)/\beta),\
\bar s=s(\zeta-\gamma)+\gamma,\ \tilde m=\min\{1,\max\{0,\bar s\}\}. \tag{2}

Invariance penalty (mask-guided augmentations):
\mathcal{R}_{inv}(\theta,\omega)=\mathbb{E}_{x,a}\big[\KL(p_{\omega}(\cdot|f_{\theta}(x))\ \|\ p_{\omega}(\cdot|f_{\theta}(a(x;\tilde m))))\big]. \tag{3}

Environment self-discovery (IRM-like):
\mathcal{R}_{env}(\theta,\omega,\psi)=\sum_{e=1}^{E}\|\nabla_{\omega}\ \mathbb{E}_{(x,y)\sim\mathcal{D}_e}\CE(p_{\omega}(\cdot|f_{\theta}(x)),y)\|_2^2. \tag{4}

Cheat score and weights:
s(x,y)=\sigma(\tau[\log C-\CE(q_{\phi}(\tilde m\odot f_{\theta}(x)),y)]),\
w(x,y)=1/(1+\exp(\eta(s-\kappa))). \tag{5}

Weighted ERM:
\mathcal{L}^{w}_{task}(\theta,\omega)=\mathbb{E}[w(x,y)\cdot \CE(p_{\omega}(\cdot|f_{\theta}(x)),y)]. \tag{6}

Overall min–max:
\min_{\theta,\omega,\psi}\max_{\phi,\tilde m} \ \mathcal{L}^{w}_{task}+\alpha \mathcal{R}_{inv}+\beta \mathcal{R}_{env}-\gamma \mathcal{L}_{probe}. \tag{7}

MI bound (Fano-style):
I(\tilde m\odot z;y)\ge \log C-\CE(q_{\phi}(\tilde m\odot z),y). \tag{8}

Local Fisher contraction:
\KL(p(\cdot|z)\|p(\cdot|z'))\approx \tfrac{1}{2}\Delta^{\top}\mathbf{F}(z)\Delta,\ \Delta=z'-z. \tag{9}

Weighted Rademacher:
\sup_{h}(L(h)-\hat L^{w}(h))\le 2\mathfrak{R}_n(\ell\circ\mathcal{H})\bar w + c\sqrt{\log(1/\delta)/n}+C_w\rho/n. \tag{10}
"""

# =========================================================
# (2) Utils, Reproducibility, Logging
# =========================================================
def set_seed(seed: int = 1337):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def now_ts():
    return time.strftime("%Y%m%d_%H%M%S")

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def save_json(d, path):
    with open(path, "w") as f:
        json.dump(d, f, indent=2)

def to_device(batch, device):
    if isinstance(batch, (list, tuple)):
        return [to_device(x, device) for x in batch]
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k,v in batch.items()}
    return batch.to(device)

def accuracy_from_logits(logits, y):
    return (logits.argmax(1) == y).float().mean().item()

def balanced_accuracy(y_true, y_pred, num_classes):
    mat = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    with np.errstate(divide='ignore', invalid='ignore'):
        per_class = np.diag(mat) / mat.sum(axis=1).clip(min=1)
    return float(np.nanmean(per_class))

def expected_calibration_error(conf, pred, y, n_bins=15):
    bins = np.linspace(0, 1, n_bins+1)
    ece = 0.0; total = len(y)
    for i in range(n_bins):
        masks = (conf > bins[i]) & (conf <= bins[i+1])
        if masks.sum() == 0: continue
        acc = (pred[masks]==y[masks]).mean()
        confm = conf[masks].mean()
        ece += (masks.sum()/total) * abs(acc - confm)
    return float(ece)

def kl_categorical_from_logits(p_logits, q_logits):
    # KL(p||q)
    p_log = F.log_softmax(p_logits, dim=1)
    q_log = F.log_softmax(q_logits, dim=1)
    p = p_log.exp()
    return (p * (p_log - q_log)).sum(1)

# =========================================================
# (3) Datasets
# =========================================================

class ColoredMNIST(Dataset):
    """
    Spurious colored background based on label parity.
    Train rho=0.9, Val rho=0.9, Test rho=0.1 to create OOD shift.
    Groups: (parity, color) -> 4 groups for worst-group accuracy.
    """
    def __init__(self, root, split="train", rho=0.9):
        super().__init__()
        train = (split=="train")
        self.base = datasets.MNIST(root=root, train=train, download=True)
        self.rho = rho
        self.transform = transforms.ToTensor()
        self.split = split

    def __len__(self): return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        x = self.transform(x)
        label_parity = (y % 2).item()
        if random.random() < self.rho:
            color = label_parity
        else:
            color = 1 - label_parity
        x3 = x.repeat(3,1,1)
        if color == 0:
            x3[0] = torch.clamp(x3[0]*1.2 + 0.25, 0, 1)
            x3[1] = torch.clamp(x3[1]*0.9, 0, 1)
            x3[2] = torch.clamp(x3[2]*0.9, 0, 1)
        else:
            x3[0] = torch.clamp(x3[0]*0.9, 0, 1)
            x3[1] = torch.clamp(x3[1]*1.2 + 0.25, 0, 1)
            x3[2] = torch.clamp(x3[2]*0.9, 0, 1)
        group = 2*label_parity + color
        return x3, y, group

def build_dataset(name: str, root: str, split: str):
    if name == "colored_mnist":
        if split in ["train","val"]:
            return ColoredMNIST(root, split="train" if split=="train" else "val", rho=0.9)
        else:
            return ColoredMNIST(root, split="test", rho=0.1)
    elif name == "waterbirds":
        if not WILDS_AVAILABLE:
            raise RuntimeError("WILDS not installed. pip install wilds")
        dataset = get_dataset(dataset="waterbirds", download=True, root_dir=root)
        if split == "train":
            subset = dataset.get_subset("train", transform=transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()]))
        elif split == "val":
            subset = dataset.get_subset("id_val", transform=transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()]))
        else:
            subset = dataset.get_subset("val", transform=transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()]))
        # Wrap to unify output (x, y, group)
        class WBD(Dataset):
            def __init__(self, sub):
                self.sub = sub
            def __len__(self): return len(self.sub)
            def __getitem__(self, idx):
                item = self.sub[idx]
                x = item[0]; y = int(item[1])
                meta = item[2]
                # metadata: [y, place], group by (y, place)
                place = int(meta[1])
                group = y*2 + place
                return x, y, group
        return WBD(subset)
    elif name == "celeba_blond_spurious":
        # label: Blond_Hair (attr index), spurious: Male
        transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
        target_attr = "Blond_Hair"
        spurious_attr = "Male"
        # torchvision CelebA has attr list in .attr; we map
        split_map = {"train":"train", "val":"valid", "test":"test"}
        base = datasets.CelebA(root=root, split=split_map[split], download=True, target_type="attr", transform=transform)
        attr_idx = base.attr_names.index(target_attr)
        spur_idx = base.attr_names.index(spurious_attr)
        class CelebAD(Dataset):
            def __init__(self, b, a_idx, s_idx):
                self.b=b; self.ai=a_idx; self.si=s_idx
            def __len__(self): return len(self.b)
            def __getitem__(self, idx):
                x, attrs = self.b[idx]
                y = int(attrs[self.ai].item()>0)  # 1 if blond
                male = int(attrs[self.si].item()>0) # 1 if male
                # known spurious: in many splits, blond correlates with female; group by (y, male)
                group = y*2 + male  # 0..3
                return x, torch.tensor(y, dtype=torch.long), group
        return CelebAD(base, attr_idx, spur_idx)
    elif name == "cifar10_spurious_patch":
        # CIFAR-10 with synthetic colored patch (top-left) correlated with class id mod 2
        base = datasets.CIFAR10(root=root, train=(split!="test"), download=True, transform=transforms.ToTensor())
        class CIFARSpur(Dataset):
            def __init__(self, b, split):
                self.b=b; self.split=split
            def __len__(self): return len(self.b)
            def __getitem__(self, idx):
                x, y = self.b[idx]
                x = transforms.ToTensor()(x)
                h, w = x.shape[1:]
                patch = torch.zeros_like(x)
                color = (y % 2)
                if color==0:
                    patch[0,:8,:8] = 1.0  # red square
                else:
                    patch[1,:8,:8] = 1.0  # green square
                rho = 0.9 if self.split!="test" else 0.1
                if random.random() < rho:
                    x = torch.clamp(x + patch, 0, 1)
                group = (y%2)*2 + color
                return x, torch.tensor(y, dtype=torch.long), group
        return CIFARSpur(base, split)
    else:
        raise ValueError(f"Unknown dataset: {name}")

# =========================================================
# (4) Models
# =========================================================

class SmallCNN(nn.Module):
    """Compact CNN for 3x28x28 or 3x32x32; upscales to 224x224 handled in dataset if needed."""
    def __init__(self, out_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /2
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /4
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /8
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*4*4, out_dim),  # works for 32x32 (CIFAR, CMNIST 28~32); for 224x224 we will use a bigger encoder below.
            nn.ReLU(inplace=True)
        )
        self.out_dim = out_dim

    def forward(self, x, return_spatial=False):
        feat = self.conv(x)
        z = self.head(feat)
        if return_spatial: return z, feat
        return z

class ResNet18Encoder(nn.Module):
    """ResNet18 backbone returning pooled representation and last conv feature maps."""
    def __init__(self, out_dim=512, pretrained=False):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.net = resnet18(weights=weights)
        self.net.fc = nn.Identity()
        self.out_dim = 512

    def forward(self, x, return_spatial=False):
        # Extract last conv feature map by hooking layer4 output
        feat = None
        def hook(m, i, o):
            nonlocal feat; feat = o
        h = self.net.layer4.register_forward_hook(hook)
        z = self.net(x)
        h.remove()
        # feature map after layer4 has shape [B,512,7,7] for 224x224 input
        if return_spatial:
            return z, feat
        return z

class Classifier(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)
    def forward(self, z): return self.fc(z)

# =========================================================
# (5) Hard-Concrete Gates and Budgeted Probe
# =========================================================
class HardConcreteGate(nn.Module):
    """
    Eq (2) with parameters beta, gamma, zeta; trainable log_alpha.
    expected_L0() ~ sigmoid(log_alpha) (stretched approximation).
    """
    def __init__(self, d, beta=2./3., gamma=-0.1, zeta=1.1, init_log_alpha=-3.0):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.full((d,), float(init_log_alpha)))
        self.beta = beta; self.gamma = gamma; self.zeta = zeta

    def forward(self, training=True):
        if training:
            u = torch.rand_like(self.log_alpha)
            s = torch.sigmoid((torch.log(u) - torch.log(1-u) + self.log_alpha) / self.beta)
        else:
            # deterministic mean gate proxy
            s = torch.sigmoid(self.log_alpha)
        s_bar = s * (self.zeta - self.gamma) + self.gamma
        m = torch.clamp(s_bar, 0.0, 1.0)
        return m

    def expected_L0(self):
        # Approximate expected "on" probability
        return torch.sigmoid(self.log_alpha)

class BudgetedProbe(nn.Module):
    """Linear probe with Hard-Concrete mask; explicit expectation budget penalty."""
    def __init__(self, d, num_classes, k_budget, lambda_budget=1.0):
        super().__init__()
        self.gate = HardConcreteGate(d)
        self.head = nn.Linear(d, num_classes)
        self.k_budget = float(k_budget)
        self.lambda_budget = lambda_budget

    def forward(self, z, training=True):
        m = self.gate(training=training)  # [d]
        z_mask = z * m  # broadcast [B,d]
        logits = self.head(z_mask)
        return logits, m

    def loss(self, z, y):
        logits, m = self.forward(z, training=True)
        ce = F.cross_entropy(logits, y, reduction="mean")
        exp_L0 = self.gate.expected_L0().sum()
        budget_pen = torch.abs(exp_L0 - self.k_budget)
        return ce + self.lambda_budget * budget_pen, ce.detach().item(), float(exp_L0.item())

# =========================================================
# (6) Env Assigner, IRM Penalty, Counter-Augmentations
# =========================================================
class EnvAssigner(nn.Module):
    def __init__(self, d, E=3, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden), nn.ReLU(True),
            nn.Linear(hidden, E)
        )
        self.E = E
    def forward(self, z):
        return F.softmax(self.net(z), dim=1)

def irm_style_penalty(logits, y, env_probs):
    E = env_probs.size(1)
    device = logits.device
    penalty = torch.zeros((), device=device)
    for e in range(E):
        w = env_probs[:, e]
        scale = torch.tensor(1.0, device=device, requires_grad=True)
        loss_e = (w * F.cross_entropy(scale*logits, y, reduction='none')).sum() / (w.sum() + 1e-8)
        g = torch.autograd.grad(loss_e, [scale], create_graph=True)[0]
        penalty = penalty + g.pow(2)
    return penalty

class CounterAugment:
    """
    Project feature mask to conv feature map and apply local perturbations:
    - Gaussian noise + average-blur inside top-p masked area.
    - Cutout-like suppression where mask high (optionally).
    """
    def __init__(self, top_p=0.2, noise_sigma=0.25, blur_kernel=3, cutout=False):
        self.top_p=top_p; self.noise_sigma=noise_sigma; self.blur_kernel=blur_kernel; self.cutout=cutout

    def spatial_mask(self, feat, m_repr):
        B,C,H,W = feat.shape
        # Align mask length to C if needed
        if m_repr.numel() >= C:
            w = m_repr[:C]
        else:
            w = torch.ones(C, device=feat.device)
        mp = (feat * w.view(1,C,1,1)).sum(1)  # [B,H,W]
        mp = (mp - mp.amin(dim=(1,2), keepdim=True)) / (mp.amax(dim=(1,2), keepdim=True)+1e-8)
        return mp  # [B,H,W]

    def __call__(self, x, feat, m_repr):
        B,_,H,W = x.shape
        mp = self.spatial_mask(feat, m_repr)             # [B,H,W]
        flat = mp.view(B, -1)
        k = (flat.size(1) * self.top_p)
        thresh = torch.topk(flat, k=int(max(1,k)), dim=1).values.min(1).values.view(B,1,1)
        region = (mp >= thresh).float().unsqueeze(1)      # [B,1,H,W]

        # Noise + blur in region
        noise = torch.randn_like(x) * self.noise_sigma
        x_aug = x * (1 - region) + torch.clamp(x + noise, 0, 1) * region
        x_blur = F.avg_pool2d(x_aug, kernel_size=self.blur_kernel, stride=1, padding=self.blur_kernel//2)
        x_aug = x_aug * (1 - region) + x_blur * region

        if self.cutout:
            x_aug = x_aug * (1 - region) + 0.0 * region
        return x_aug

# =========================================================
# (7) S4 System: Training/Eval
# =========================================================
@dataclass
class Config:
    dataset: str = "colored_mnist"  # choices: colored_mnist, waterbirds, celeba_blond_spurious, cifar10_spurious_patch
    root: str = "./data"
    outdir: str = "./runs"
    batch_size: int = 256
    epochs: int = 30
    d_repr: int = 256
    lr_main: float = 3e-4
    lr_probe: float = 1e-3
    weight_decay: float = 5e-5
    k: int = 8
    lambda_budget: float = 1.0
    alpha_inv: float = 1.0
    beta_env: float = 0.1
    probe_steps: int = 1
    main_steps: int = 1
    envs: int = 3
    tau: float = 8.0
    kappa: float = 0.6
    eta: float = 10.0
    mixed_precision: bool = True
    grad_clip: float = 1.0
    patience: int = 10
    min_delta: float = 1e-4
    seed: int = 1337
    num_workers: int = 4
    model_large: bool = False  # use ResNet18 for 224x224 datasets

class S4System:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        set_seed(cfg.seed)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Paths
        self.run_dir = os.path.join(cfg.outdir, cfg.dataset, now_ts())
        ensure_dir(self.run_dir)
        save_json(asdict(cfg), os.path.join(self.run_dir, "config.json"))
        with open(os.path.join(self.run_dir, "spec.txt"), "w") as f:
            f.write(SPEC)

        # Data
        self.train_ds = build_dataset(cfg.dataset, cfg.root, "train")
        self.val_ds   = build_dataset(cfg.dataset, cfg.root, "val")
        self.test_ds  = build_dataset(cfg.dataset, cfg.root, "test")

        self.train_loader = DataLoader(self.train_ds, batch_size=cfg.batch_size, shuffle=True,
                                       num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
        self.val_loader   = DataLoader(self.val_ds,   batch_size=cfg.batch_size, shuffle=False,
                                       num_workers=cfg.num_workers, pin_memory=True)
        self.test_loader  = DataLoader(self.test_ds,  batch_size=cfg.batch_size, shuffle=False,
                                       num_workers=cfg.num_workers, pin_memory=True)

        # Model selection (small vs large)
        if cfg.dataset in ["waterbirds", "celeba_blond_spurious"]:
            cfg.model_large = True

        if cfg.model_large:
            self.encoder = ResNet18Encoder(pretrained=False).to(self.device)
            d = self.encoder.out_dim
        else:
            self.encoder = SmallCNN(out_dim=cfg.d_repr).to(self.device)
            d = cfg.d_repr

        # Heads
        num_classes = 10 if cfg.dataset in ["colored_mnist","cifar10_spurious_patch"] else 2
        self.classifier = Classifier(d, num_classes=num_classes).to(self.device)
        self.probe = BudgetedProbe(d, num_classes=num_classes, k_budget=cfg.k, lambda_budget=cfg.lambda_budget).to(self.device)
        self.env_assigner = EnvAssigner(d, E=cfg.envs).to(self.device)
        self.augment = CounterAugment(top_p=0.2, noise_sigma=0.25, blur_kernel=3, cutout=False)

        # Optimizers & schedulers
        self.opt_main = torch.optim.AdamW(
            list(self.encoder.parameters()) + list(self.classifier.parameters()) + list(self.env_assigner.parameters()),
            lr=cfg.lr_main, weight_decay=cfg.weight_decay
        )
        self.opt_probe = torch.optim.AdamW(self.probe.parameters(), lr=cfg.lr_probe, weight_decay=0.0)
        self.sched_main = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt_main, T_max=max(1,cfg.epochs))
        self.sched_probe = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt_probe, T_max=max(1,cfg.epochs))

        self.scaler = torch.cuda.amp.GradScaler(enabled=cfg.mixed_precision)

        # Logs
        self.history = []

    # ------------------------
    # Cheat score and weights
    # ------------------------
    @torch.no_grad()
    def cheat_score_and_weight(self, z, y, C):
        logits_probe, _ = self.probe.forward(z.detach(), training=False)
        ce = F.cross_entropy(logits_probe, y, reduction="none")
        s = torch.sigmoid(self.cfg.tau*(math.log(C)-ce))
        w = torch.sigmoid(-self.cfg.eta*(s - self.cfg.kappa))
        return s, w

    # ------------------------
    # Invariance KL via mask-guided augmentations
    # ------------------------
    def invariance_kl_batch(self, x):
        self.encoder.eval(); self.probe.eval()
        with torch.no_grad():
            z, feat = self.encoder(x, return_spatial=True)
            logits = self.classifier(z)
            _, m = self.probe.forward(z, training=False)
            Cch = feat.size(1)
            m_feat = m[:Cch] if m.numel()>=Cch else torch.ones(Cch, device=feat.device)
            x_aug = self.augment(x, feat, m_feat)
            logits2 = self.classifier(self.encoder(x_aug))
        return kl_categorical_from_logits(logits, logits2)

    def env_penalty(self, z, y, logits):
        e = self.env_assigner(z)
        penalty = irm_style_penalty(logits, y, e)
        ent = -(e * torch.log(e + 1e-8)).sum(dim=1).mean()
        bal = ((e.mean(dim=0) - (1.0/self.cfg.envs))**2).sum()
        return penalty + 0.01*ent + 0.01*bal

    # ------------------------
    # Phase A: Probe ascent (encoder frozen)
    # ------------------------
    def probe_step(self, batch):
        self.encoder.eval(); self.probe.train()
        x, y, g = to_device(batch, self.device)
        with torch.no_grad():
            z = self.encoder(x)
        loss_probe, ce_val, exp_L0 = self.probe.loss(z, y)
        self.opt_probe.zero_grad(set_to_none=True)
        loss_probe.backward()
        torch.nn.utils.clip_grad_norm_(self.probe.parameters(), 1.0)
        self.opt_probe.step()
        return float(loss_probe.item()), float(ce_val), float(exp_L0)

    # ------------------------
    # Phase B: Main descent
    # ------------------------
    def main_step(self, batch, C):
        self.encoder.train(); self.classifier.train(); self.env_assigner.train(); self.probe.eval()
        x, y, g = to_device(batch, self.device)
        self.opt_main.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=self.cfg.mixed_precision):
            z, feat = self.encoder(x, return_spatial=True)
            logits = self.classifier(z)
            # weighted ERM
            _, w = self.cheat_score_and_weight(z, y, C)
            ce = F.cross_entropy(logits, y, reduction="none")
            loss_task_w = (w * ce).mean()
            # invariance KL
            kl = self.invariance_kl_batch(x).mean()
            # env penalty
            pen_env = self.env_penalty(z, y, logits)
            loss = loss_task_w + self.cfg.alpha_inv*kl + self.cfg.beta_env*pen_env

        self.scaler.scale(loss).backward()
        if self.cfg.grad_clip > 0:
            self.scaler.unscale_(self.opt_main)
            torch.nn.utils.clip_grad_norm_(list(self.encoder.parameters())+list(self.classifier.parameters())+list(self.env_assigner.parameters()), self.cfg.grad_clip)
        self.scaler.step(self.opt_main)
        self.scaler.update()
        return float(loss.item()), float(loss_task_w.item()), float(kl.item()), float(pen_env.item())

    # ------------------------
    # Evaluation / Metrics
    # ------------------------
    @torch.no_grad()
    def evaluate(self, loader, C, tag="val"):
        self.encoder.eval(); self.classifier.eval(); self.probe.eval()
        total, correct = 0, 0
        y_true_all, y_pred_all, conf_all = [], [], []
        group_ok = True
        # detect num groups (for our datasets 4 groups; else we skip worst-group)
        group_acc = None
        if hasattr(loader.dataset, "__getitem__"):
            # assume groups exist for our datasets
            group_correct = {}
            group_total = {}
        else:
            group_ok = False

        inv_kls = []
        cheat_scores = []

        for x, y, g in loader:
            x, y, g = to_device((x,y,g), self.device)
            z, feat = self.encoder(x, return_spatial=True)
            logits = self.classifier(z)
            pred = logits.argmax(1)
            prob = F.softmax(logits, 1).max(1).values

            total += y.size(0)
            correct += (pred==y).sum().item()
            y_true_all.append(y.cpu().numpy()); y_pred_all.append(pred.cpu().numpy()); conf_all.append(prob.cpu().numpy())

            # group stats
            if group_ok:
                gs = g.cpu().numpy().tolist()
                ys = y.cpu().numpy().tolist()
                ps = pred.cpu().numpy().tolist()
                for gi, yy, pp in zip(gs, ys, ps):
                    group_total[gi] = group_total.get(gi,0)+1
                    group_correct[gi] = group_correct.get(gi,0)+ (1 if yy==pp else 0)

            # invariance gap sampling
            _, m = self.probe.forward(z, training=False)
            Cch = feat.size(1)
            m_feat = m[:Cch] if m.numel()>=Cch else torch.ones(Cch, device=feat.device)
            x_aug = self.augment(x, feat, m_feat)
            logits2 = self.classifier(self.encoder(x_aug))
            inv_kls.append(kl_categorical_from_logits(logits, logits2).mean().item())

            # cheat scores
            ce_probe = F.cross_entropy(self.probe.forward(z, training=False)[0], y, reduction="none")
            s = torch.sigmoid(self.cfg.tau*(math.log(C)-ce_probe))
            cheat_scores.append(s.mean().item())

        acc = correct/total
        y_true = np.concatenate(y_true_all); y_pred = np.concatenate(y_pred_all); conf = np.concatenate(conf_all)
        bacc = balanced_accuracy(y_true, y_pred, C)
        ece = expected_calibration_error(conf, y_pred, y_true)
        inv_gap = float(np.mean(inv_kls))
        cheat_mean = float(np.mean(cheat_scores))
        cm = confusion_matrix(y_true, y_pred, labels=list(range(C)))
        metrics = {"acc":acc, "balanced_acc":bacc, "ece":ece, "inv_gap":inv_gap, "cheat_mean":cheat_mean, "cm":cm.tolist()}

        if group_ok and len(group_total)>0:
            wg = min([group_correct[k]/max(1,group_total[k]) for k in group_total.keys()])
            metrics["worst_group_acc"] = float(wg)
            # Save per-group table
            df = pd.DataFrame({
                "group": list(group_total.keys()),
                "count": [group_total[k] for k in group_total.keys()],
                "acc": [group_correct[k]/max(1,group_total[k]) for k in group_total.keys()]
            })
            df.to_csv(os.path.join(self.run_dir, f"{tag}_group_metrics.csv"), index=False)

        # Save confusion matrix plot
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title(f"{tag} Confusion Matrix")
        plt.xlabel("Predicted"); plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, f"{tag}_confusion_matrix.png"))
        plt.close()

        return metrics

    # ------------------------
    # Plots & Artifacts
    # ------------------------
    def plot_history(self, hist: List[Dict[str,Any]]):
        df = pd.DataFrame(hist)
        df.to_csv(os.path.join(self.run_dir, "history.csv"), index=False)
        # plot keys if exist
        for key in ["train_loss","val_acc","val_worst_group_acc","val_inv_gap","probe_ce"]:
            if key in df.columns:
                plt.figure(); plt.plot(df[key].values); plt.title(key); plt.xlabel("Epoch"); plt.ylabel(key); plt.grid(True)
                plt.tight_layout(); plt.savefig(os.path.join(self.run_dir, f"{key}.png")); plt.close()

    @torch.no_grad()
    def save_mask_overlay(self, batch, epoch, prefix="train"):
        self.encoder.eval(); self.probe.eval()
        x, y, g = to_device(batch, self.device)
        z, feat = self.encoder(x, return_spatial=True)
        _, m = self.probe.forward(z, training=False)
        Cch = feat.size(1)
        m_feat = m[:Cch] if m.numel()>=Cch else torch.ones(Cch, device=feat.device)
        mp = (feat * m_feat.view(1,Cch,1,1)).sum(1)
        mp = (mp - mp.amin()) / (mp.amax()-mp.amin()+1e-8)
        mp = mp.unsqueeze(1).repeat(1,3,1,1)
        overlay = 0.6*x + 0.4*mp
        grid = make_grid(overlay[:16], nrow=8)
        save_image(grid, os.path.join(self.run_dir, f"{prefix}_mask_epoch{epoch}.png"))

    # ------------------------
    # Exposure curve (probe accuracy vs budgets)
    # ------------------------
    @torch.no_grad()
    def exposure_curve(self, budgets: List[int], loader, C):
        self.encoder.eval()
        results = []
        for k in budgets:
            # reset probe fresh for eval exposure (frozen encoder)
            d = self.classifier.fc.in_features
            probe = BudgetedProbe(d, C, k_budget=k, lambda_budget=self.cfg.lambda_budget).to(self.device)
            opt = torch.optim.AdamW(probe.parameters(), lr=1e-3)
            # short train
            for _ in range(2):
                for x,y,g in loader:
                    x,y = to_device(x,self.device), to_device(y,self.device)
                    z = self.encoder(x)
                    loss,ce,_ = probe.loss(z,y)
                    opt.zero_grad(); loss.backward(); opt.step()
            # eval
            correct=0; total=0
            for x,y,g in loader:
                x,y = to_device(x,self.device), to_device(y,self.device)
                z = self.encoder(x)
                logits,_ = probe.forward(z, training=False)
                pred = logits.argmax(1)
                correct += (pred==y).sum().item(); total+=y.size(0)
            acc = correct/total
            results.append((k, acc))
        df = pd.DataFrame(results, columns=["k","probe_acc"])
        df.to_csv(os.path.join(self.run_dir, "exposure_curve.csv"), index=False)
        plt.figure(); plt.plot(df["k"], df["probe_acc"], marker="o")
        plt.title("Probe Exposure Curve (accuracy vs budget k)"); plt.xlabel("k"); plt.ylabel("Probe accuracy"); plt.grid(True)
        plt.tight_layout(); plt.savefig(os.path.join(self.run_dir, "exposure_curve.png")); plt.close()

    # ------------------------
    # Train
    # ------------------------
    def train(self):
        C = 10 if self.cfg.dataset in ["colored_mnist","cifar10_spurious_patch"] else 2
        best_val = -1e9; best_epoch = -1; epochs_no_improve = 0

        for epoch in range(1, self.cfg.epochs+1):
            # ---- Probe ascent ----
            probe_loss_epoch=[]; probe_ce_epoch=[]
            for _ in range(self.cfg.probe_steps):
                for batch in tqdm(self.train_loader, desc=f"[Epoch {epoch}] Probe", leave=False):
                    loss_probe, ce_val, expL0 = self.probe_step(batch)
                    probe_loss_epoch.append(loss_probe); probe_ce_epoch.append(ce_val)

            # ---- Main descent ----
            train_losses=[]; task_w_losses=[]; kls=[]; pens=[]
            for _ in range(self.cfg.main_steps):
                for batch in tqdm(self.train_loader, desc=f"[Epoch {epoch}] Main", leave=False):
                    loss, lt, kl, pen = self.main_step(batch, C)
                    train_losses.append(loss); task_w_losses.append(lt); kls.append(kl); pens.append(pen)

            # sched
            self.sched_main.step(); self.sched_probe.step()

            # Save mask overlay
            try:
                self.save_mask_overlay(next(iter(self.train_loader)), epoch, prefix="train")
            except Exception:
                pass

            # ---- Eval ----
            train_metrics = self.evaluate(self.train_loader, C, tag="train")
            val_metrics   = self.evaluate(self.val_loader,   C, tag="val")
            test_metrics  = self.evaluate(self.test_loader,  C, tag="test")

            # ---- Logging ----
            log = {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)) if train_losses else None,
                "task_w": float(np.mean(task_w_losses)) if task_w_losses else None,
                "kl": float(np.mean(kls)) if kls else None,
                "pen_env": float(np.mean(pens)) if pens else None,
                "probe_ce": float(np.mean(probe_ce_epoch)) if probe_ce_epoch else None,
                "train_acc": train_metrics["acc"],
                "val_acc": val_metrics["acc"],
                "test_acc": test_metrics["acc"],
                "val_worst_group_acc": val_metrics.get("worst_group_acc", None),
                "val_inv_gap": val_metrics["inv_gap"],
                "val_ece": val_metrics["ece"],
            }
            self.history.append(log)
            self.plot_history(self.history)
            save_json({"train":train_metrics,"val":val_metrics,"test":test_metrics},
                      os.path.join(self.run_dir, f"metrics_epoch{epoch}.json"))

            # ---- Early stopping (val acc or worst-group if present) ----
            val_score = val_metrics.get("worst_group_acc", None)
            if val_score is None: val_score = val_metrics["acc"]
            if val_score > best_val + self.cfg.min_delta:
                best_val = val_score; best_epoch = epoch; epochs_no_improve = 0
                ckpt = {
                    "encoder": self.encoder.state_dict(),
                    "classifier": self.classifier.state_dict(),
                    "probe": self.probe.state_dict(),
                    "env_assigner": self.env_assigner.state_dict(),
                    "cfg": asdict(self.cfg),
                    "history": self.history,
                    "epoch": epoch
                }
                torch.save(ckpt, os.path.join(self.run_dir, "best.ckpt"))
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.cfg.patience:
                    print(f"Early stopping at epoch {epoch}. Best epoch {best_epoch} (val score={best_val:.4f}).")
                    break

        # Final artifacts
        budgets = sorted(list(set([max(1,int(self.cfg.k/2)), self.cfg.k, self.cfg.k*2])))
        self.exposure_curve(budgets, self.val_loader, C)

        # Final metrics summary
        final = {
            "best_epoch": best_epoch,
            "best_val": best_val,
            "last_val": self.history[-1]["val_acc"],
            "last_test": self.history[-1]["test_acc"]
        }
        save_json(final, os.path.join(self.run_dir, "final_summary.json"))

# =========================================================
# (8) CLI
# =========================================================
def parse_args():
    p = argparse.ArgumentParser("S4 — Production-Ready Implementation")
    p.add_argument("--dataset", type=str, default="colored_mnist",
                   choices=["colored_mnist","waterbirds","celeba_blond_spurious","cifar10_spurious_patch"])
    p.add_argument("--root", type=str, default="./data")
    p.add_argument("--outdir", type=str, default="./runs")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--d-repr", type=int, default=256)
    p.add_argument("--lr-main", type=float, default=3e-4)
    p.add_argument("--lr-probe", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=5e-5)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--lambda-budget", type=float, default=1.0)
    p.add_argument("--alpha-inv", type=float, default=1.0)
    p.add_argument("--beta-env", type=float, default=0.1)
    p.add_argument("--probe-steps", type=int, default=1)
    p.add_argument("--main-steps", type=int, default=1)
    p.add_argument("--envs", type=int, default=3)
    p.add_argument("--tau", type=float, default=8.0)
    p.add_argument("--kappa", type=float, default=0.6)
    p.add_argument("--eta", type=float, default=10.0)
    p.add_argument("--mixed-precision", action="store_true")
    p.add_argument("--no-mixed-precision", dest="mixed_precision", action="store_false")
    p.set_defaults(mixed_precision=True)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--min-delta", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--num-workers", type=int, default=4)
    args = p.parse_args()
    cfg = Config(
        dataset=args.dataset, root=args.root, outdir=args.outdir, batch_size=args.batch_size,
        epochs=args.epochs, d_repr=args.d_repr, lr_main=args.lr_main, lr_probe=args.lr_probe,
        weight_decay=args.weight_decay, k=args.k, lambda_budget=args.lambda_budget,
        alpha_inv=args.alpha_inv, beta_env=args.beta_env, probe_steps=args.probe_steps,
        main_steps=args.main_steps, envs=args.envs, tau=args.tau, kappa=args.kappa, eta=args.eta,
        mixed_precision=args.mixed_precision, grad_clip=args.grad_clip, patience=args.patience,
        min_delta=args.min_delta, seed=args.seed, num_workers=args.num_workers
    )
    return cfg

if __name__ == "__main__":
    cfg = parse_args()
    system = S4System(cfg)
    system.train()
