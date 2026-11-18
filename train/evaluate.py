"""Bewertet TireNet so, wie die Web-Demo entscheidet.

1. Open-Images-Validierung: 3000 Ausschnitte (feste Saat) – stimmt die Entscheidung "grün" mit der Wahrheit überein?
2. Echte Handyfotos aus train/realtest – je Foto drei Varianten (zentriert = grün erwartet,
   Kreis auf der Karosserie / seitlich versetzt = orange erwartet)
3. Laufzeit mit ONNX Runtime CPU, 1 Thread (grober Anhaltspunkt für Browser-WASM)
"""
import json, sys, time
import numpy as np
import onnxruntime as ort
from PIL import Image
from train import Crops, metrics, fits, DATA, ROOT, ROI_FACTOR

THRESHOLD = float(sys.argv[1]) if len(sys.argv) > 1 else 0.8   # wie der Standard in der Web-Demo


def session(path):
    o = ort.SessionOptions(); o.intra_op_num_threads = 1
    return ort.InferenceSession(str(path), o, providers=["CPUExecutionProvider"])


def predict(s, img):  # img: 224x224x3 float 0..1
    return s.run(None, {s.get_inputs()[0].name: img.transpose(2, 0, 1)[None].astype(np.float32)})[0][0].tolist()


def roi_from_photo(im, center=None):
    """Ausschnitt wie in der Web-Demo: Kreisdurchmesser = 0.9 × Bildbreite, Ausschnitt = 1.6 × Durchmesser."""
    W, H = im.size
    cx, cy = center or (W / 2, H / 2)
    side = 0.9 * W * ROI_FACTOR
    x0, y0 = cx - side / 2, cy - side / 2
    canvas = Image.new("RGB", (224, 224))
    k = 224 / side
    src = im.crop((max(0, x0), max(0, y0), min(W, x0 + side), min(H, y0 + side)))
    canvas.paste(src.resize((max(1, round(src.width * k)), max(1, round(src.height * k)))),
                 (round((max(0, x0) - x0) * k), round((max(0, y0) - y0) * k)))
    return np.asarray(canvas, dtype=np.float32) / 255


def main():
    s = session(ROOT / "tirenet.onnx")
    meta = json.load(open(DATA / "meta.json"))
    ds = Crops(meta["val"], 3000, train=False, seed=7)
    P, Y, t = [], [], 0.0
    for i in range(len(ds)):
        x, y = ds[i]
        t0 = time.perf_counter(); P.append(predict(s, x.permute(1, 2, 0).numpy())); t += time.perf_counter() - t0
        Y.append(y.tolist())
    print(f"== Open-Images-Validierung ({len(Y)} Ausschnitte)")
    print(metrics(P, Y), f"{1000 * t / len(Y):.1f} ms/Bild")
    fp_none = sum(1 for p, y in zip(P, Y) if fits(*p, thr=THRESHOLD) and y[0] < 0.5)
    print(f"grün ohne Reifen im Bild bei Schwelle {THRESHOLD}: {fp_none}")

    real = sorted((ROOT / "realtest").glob("*.jpg"))
    if real:
        print("\n== Echte Handyfotos")
        ok = total = 0
        for f in real:
            im = Image.open(f).convert("RGB"); W, H = im.size
            for name, center, expect in [("zentriert", None, True),
                                         ("Kreis auf Karosserie", (W / 2, H * 0.17), False),
                                         ("seitlich versetzt", (W * 0.85, H / 2), False)]:
                p = predict(s, roi_from_photo(im, center))
                got = fits(*p, thr=THRESHOLD)
                ok += got == expect; total += 1
                print(f"{f.stem:8s} {name:21s} erwartet {'grün  ' if expect else 'orange'} → "
                      f"{'grün  ' if got else 'orange'}  (p={p[0]:.2f}, Größe={max(p[3], p[4]) * ROI_FACTOR:.2f})")
        print(f"richtig: {ok}/{total}")


if __name__ == "__main__":
    main()
