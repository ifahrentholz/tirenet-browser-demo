"""Eigenes Spezialmodell: "Ist im ROI-Ausschnitt ein Reifen, und wo?"

Backbone: torchvision MobileNetV3-Small (BSD-3, ImageNet-Gewichte), eigener Kopf.
Ausgabe [p, cx, cy, w, h] relativ zum quadratischen ROI (cx/cy dürfen außerhalb 0..1 liegen).
Training ausschließlich mit Open-Images-Daten (CC BY 4.0 / CC BY 2.0) – kein Ultralytics-Code.
"""
import json, math, random, sys, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFilter, ImageEnhance
import torchvision

ROOT = Path(__file__).parent
DATA = ROOT / "data"
INPUT = 224
# Geometrie wie in der Web-Demo: ROI = 1.6 × Kreisdurchmesser
ROI_FACTOR = 1.6
R_ROI = 1 / (2 * ROI_FACTOR)
CENTER_TOL, SIZE_MIN, SIZE_MAX = 0.20, 0.70, 1.15


def fits(p, cx, cy, w, h, thr=0.5):
    if p < thr:
        return False
    center = math.hypot(cx - 0.5, cy - 0.5) / R_ROI
    size = max(w, h) / (2 * R_ROI)
    return center <= CENTER_TOL and SIZE_MIN <= size <= SIZE_MAX


# ---------------------------------------------------------------- Daten
class Crops(Dataset):
    def __init__(self, meta, length, train=True, seed=0, size=INPUT):
        self.size = size
        self.pos = [(i, [b["box"] for b in boxes]) for i, boxes in meta["pos"].items()]
        self.neg = meta["neg"]
        self.length, self.train, self.seed = length, train, seed

    def __len__(self):
        return self.length

    def _load(self, image_id):
        return Image.open(DATA / "img" / f"{image_id}.jpg").convert("RGB")

    def __getitem__(self, idx):
        rnd = random.Random(self.seed * 1_000_003 + idx) if not self.train else random
        u = rnd.random()
        if u < 0.65:   # Crop um einen Reifen herum (ROI-ähnlich)
            image_id, boxes = rnd.choice(self.pos)
            im = self._load(image_id)
            W, H = im.size
            px = [[b[0] * W, b[1] * H, b[2] * W, b[3] * H] for b in boxes]
            big = [b for b in px if max(b[2] - b[0], b[3] - b[1]) >= 40] or px
            b = rnd.choice(big)
            bs = max(b[2] - b[0], b[3] - b[1])
            if u < 0.30:
                # "Grenzfall-Modus": Reifen nahe der Kreisposition – knappe Treffer und knappe Fehlschüsse
                s = rnd.uniform(0.38, 0.80)
                off = 0.10
            else:
                s = rnd.uniform(0.25, 1.0)
                off = 0.30 if rnd.random() < 0.7 else 0.5
            side = bs / s
            cx = (b[0] + b[2]) / 2 + rnd.uniform(-off, off) * side
            cy = (b[1] + b[3]) / 2 + rnd.uniform(-off, off) * side
            crop = (cx - side / 2, cy - side / 2, side)
        elif u < 0.80:  # zufälliger Crop aus Positivbild
            image_id, boxes = rnd.choice(self.pos)
            im = self._load(image_id)
            W, H = im.size
            px = [[b[0] * W, b[1] * H, b[2] * W, b[3] * H] for b in boxes]
            side = rnd.uniform(0.3, 1.0) * min(W, H)
            crop = (rnd.uniform(0, W - side), rnd.uniform(0, H - side), side)
        else:           # Negativbild
            im = self._load(rnd.choice(self.neg))
            W, H = im.size
            px = []
            side = rnd.uniform(0.3, 1.0) * min(W, H)
            crop = (rnd.uniform(0, W - side), rnd.uniform(0, H - side), side)

        x0, y0, side = crop
        # Zielbox: größte sichtbare Fläche innerhalb Crop ∩ Bild
        target, best = None, 0.0
        for b in px:
            area = (b[2] - b[0]) * (b[3] - b[1])
            ix = max(0, min(b[2], x0 + side, W) - max(b[0], x0, 0))
            iy = max(0, min(b[3], y0 + side, H) - max(b[1], y0, 0))
            vis = ix * iy
            if vis > best:
                best, target = vis, (b, vis / max(area, 1e-6))
        present = 0.0
        tgt = [0.5, 0.5, 0.0, 0.0]
        if target:
            b, frac = target
            rel = [(b[0] - x0) / side, (b[1] - y0) / side, (b[2] - x0) / side, (b[3] - y0) / side]
            size = max(rel[2] - rel[0], rel[3] - rel[1])
            if frac >= 0.5 and size >= 0.15:
                present = 1.0
                tgt = [(rel[0] + rel[2]) / 2, (rel[1] + rel[3]) / 2, rel[2] - rel[0], rel[3] - rel[1]]

        # Crop mit schwarzem Rand außerhalb des Bildes (wie ROI im Prototyp)
        canvas = Image.new("RGB", (self.size, self.size))
        k = self.size / side
        src = im.crop((max(0, x0), max(0, y0), min(W, x0 + side), min(H, y0 + side)))
        if src.width > 0 and src.height > 0:
            dw, dh = max(1, round(src.width * k)), max(1, round(src.height * k))
            canvas.paste(src.resize((dw, dh), Image.BILINEAR), (round((max(0, x0) - x0) * k), round((max(0, y0) - y0) * k)))
        img = canvas

        if self.train:
            if rnd.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                tgt[0] = 1 - tgt[0]
            img = ImageEnhance.Brightness(img).enhance(rnd.uniform(0.5, 1.4))
            img = ImageEnhance.Contrast(img).enhance(rnd.uniform(0.6, 1.4))
            img = ImageEnhance.Color(img).enhance(rnd.uniform(0.3, 1.3))
            if rnd.random() < 0.25:
                img = img.filter(ImageFilter.GaussianBlur(rnd.uniform(0.5, 2.0)))
            if rnd.random() < 0.3:  # schwarze Seitenbalken (Hochformat-Video im ROI)
                a = np.asarray(img).copy()
                bar = int(rnd.uniform(0.05, 0.25) * self.size)
                a[:, :bar] = 0
                a[:, self.size - bar:] = 0
                img = Image.fromarray(a)
        x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        return x, torch.tensor([present, *tgt], dtype=torch.float32)


# ---------------------------------------------------------------- Modell
class TireNet(nn.Module):
    def __init__(self):
        super().__init__()
        bb = torchvision.models.mobilenet_v3_small(weights="IMAGENET1K_V1")
        self.features = bb.features
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(576, 256), nn.Hardswish(),
                                  nn.Dropout(0.1), nn.Linear(256, 5))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def raw(self, x):  # x: RGB 0..1
        return self.head(self.features((x - self.mean) / self.std))

    @staticmethod
    def decode(r):
        p = torch.sigmoid(r[:, 0:1])
        c = torch.sigmoid(r[:, 1:3]) * 1.6 - 0.3
        wh = torch.sigmoid(r[:, 3:5]) * 1.5
        return torch.cat([p, c, wh], 1)

    def forward(self, x):
        return self.decode(self.raw(x))


def loss_fn(r, y):
    pred = TireNet.decode(r)
    cls = F.binary_cross_entropy_with_logits(r[:, 0], y[:, 0])
    m = y[:, 0] > 0.5
    box = F.smooth_l1_loss(pred[m, 1:], y[m, 1:], beta=0.05) if m.any() else r.sum() * 0
    return cls + 5 * box, cls.item(), box.item()


# ---------------------------------------------------------------- Auswertung
def metrics(preds, ys):
    preds, ys = np.asarray(preds), np.asarray(ys)
    pres_acc = ((preds[:, 0] >= 0.5) == (ys[:, 0] >= 0.5)).mean()
    m = ys[:, 0] >= 0.5
    def iou(a, b):
        ax0, ay0, ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
        bx0, by0, bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
        i = max(0, min(ax1, bx1) - max(ax0, bx0)) * max(0, min(ay1, by1) - max(ay0, by0))
        return i / (a[2] * a[3] + b[2] * b[3] - i + 1e-9)
    ious = [iou(p[1:], y[1:]) if p[0] >= 0.5 else 0.0 for p, y in zip(preds[m], ys[m])]
    gt = np.array([fits(*y) for y in ys])
    pr = np.array([fits(*p) for p in preds])
    tp, fp, fn = (gt & pr).sum(), (~gt & pr).sum(), (gt & ~pr).sum()
    return {
        "Präsenz-Genauigkeit": round(float(pres_acc), 3),
        "mittl. IoU (Positiv)": round(float(np.mean(ious)), 3),
        "Entscheidung grün korrekt": round(float((gt == pr).mean()), 3),
        "grün: Precision": round(float(tp / max(tp + fp, 1)), 3),
        "grün: Recall": round(float(tp / max(tp + fn, 1)), 3),
        "Fälle grün (GT)": int(gt.sum()), "Fälle gesamt": int(len(gt)),
    }


def main():
    meta = json.load(open(DATA / "meta.json"))
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    train_ds = Crops(meta["train"], 12000, train=True)
    val_ds = Crops(meta["val"], 3000, train=False, seed=7)
    tl = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=8, persistent_workers=True)
    vl = DataLoader(val_ds, batch_size=128, num_workers=8)
    model = TireNet().to(dev)
    opt = torch.optim.AdamW([{"params": model.features.parameters(), "lr": 5e-4},
                             {"params": model.head.parameters(), "lr": 2e-3}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[5e-4, 2e-3], total_steps=epochs * len(tl), pct_start=0.15)
    best = -1
    for ep in range(epochs):
        model.train()
        t0, lc, lb, n = time.time(), 0, 0, 0
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            loss, c, b = loss_fn(model.raw(x), y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            lc += c; lb += b; n += 1
        model.eval(); P, Y = [], []
        with torch.no_grad():
            for x, y in vl:
                P += model(x.to(dev)).cpu().tolist(); Y += y.tolist()
        m = metrics(P, Y)
        print(f"Epoche {ep + 1}/{epochs} {time.time() - t0:.0f}s  loss cls {lc / n:.3f} box {lb / n:.4f}  {m}", flush=True)
        score = m["Entscheidung grün korrekt"] + m["mittl. IoU (Positiv)"]
        if score > best:
            best = score
            torch.save(model.state_dict(), ROOT / "tirenet.pt")
    print("bestes Modell gespeichert", flush=True)


if __name__ == "__main__":
    main()
