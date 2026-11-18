"""Lädt Open-Images-Bilder mit Reifen/Rad (Positiv) und ohne Fahrzeugbezug (Negativ),
verkleinert sie auf max. 640 px und schreibt eine Annotationsdatei (JSON).

Lizenz Open Images: Annotationen CC BY 4.0, Bilder CC BY 2.0 (Namensnennung).
"""
import csv, json, random, sys, io, collections
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import urllib.request
from PIL import Image

ROOT = Path(__file__).parent / "data"
OI = ROOT / "oi"
POS = {"/m/0h9mv": "Tire", "/m/083wq": "Wheel"}
# Bilder mit diesen Klassen nicht als Negativ verwenden (könnten unannotierte Räder enthalten)
VEHICLE_NAMES = {"Ambulance", "Cart", "Bicycle", "Bus", "Bicycle wheel", "Vehicle registration plate",
                 "Limousine", "Land vehicle", "Snowmobile", "Golf cart", "Motorcycle", "Skateboard", "Segway",
                 "Tank", "Train", "Truck", "Vehicle", "Wheel", "Auto part", "Unicycle", "Van", "Tire", "Car",
                 "Taxi", "Wheelchair"}
# "harte" Negative: runde Dinge
HARD_NEG_NAMES = {"Ball", "Bagel", "Clock", "Coin", "Drum", "Bowl", "Plate", "Pizza", "Wall clock", "Doughnut"}
CSV_URL = "https://storage.googleapis.com/openimages/v5/{}"
OI.mkdir(parents=True, exist_ok=True)
for f in ("class-descriptions-boxable.csv", "test-annotations-bbox.csv", "validation-annotations-bbox.csv"):
    if not (OI / f).exists():
        print("lade", f, flush=True)
        urllib.request.urlretrieve(CSV_URL.format(f), OI / f)
_NAMES = {name: mid for mid, name in csv.reader(open(OI / "class-descriptions-boxable.csv"))}
VEHICLE = {_NAMES[n] for n in VEHICLE_NAMES}
HARD_NEG = {_NAMES[n] for n in HARD_NEG_NAMES}
MAX_SIDE = 640
random.seed(42)


def load(split):
    pos = collections.defaultdict(list)
    labels = collections.defaultdict(set)
    for r in csv.DictReader(open(OI / f"{split}-annotations-bbox.csv")):
        labels[r["ImageID"]].add(r["LabelName"])
        if r["LabelName"] in POS and r["IsGroupOf"] == "0" and r["IsDepiction"] == "0":
            box = [float(r["XMin"]), float(r["YMin"]), float(r["XMax"]), float(r["YMax"])]
            pos[r["ImageID"]].append({"cls": POS[r["LabelName"]], "box": box,
                                      "trunc": r["IsTruncated"] == "1", "occl": r["IsOccluded"] == "1"})
    # mind. eine Box >= 15 % der Bildseite
    pos = {i: b for i, b in pos.items() if any(max(x["box"][2] - x["box"][0], x["box"][3] - x["box"][1]) >= 0.15 for x in b)}
    neg_all = [i for i, l in labels.items() if not (l & VEHICLE)]
    hard = [i for i in neg_all if labels[i] & HARD_NEG]
    return pos, neg_all, hard


def fetch(split, image_id):
    out = ROOT / "img" / f"{image_id}.jpg"
    if out.exists():
        return image_id, True
    url = f"https://open-images-dataset.s3.amazonaws.com/{split}/{image_id}.jpg"
    try:
        data = urllib.request.urlopen(url, timeout=30).read()
        im = Image.open(io.BytesIO(data)).convert("RGB")
        im.thumbnail((MAX_SIDE, MAX_SIDE))
        im.save(out, quality=90)
        return image_id, True
    except Exception as e:  # noqa
        print("fail", image_id, e, file=sys.stderr)
        return image_id, False


def main():
    (ROOT / "img").mkdir(parents=True, exist_ok=True)
    plan = {"train": ("test", 5000, 1500, 1000), "val": ("validation", 1200, 400, 200)}
    meta = {}
    for name, (split, n_pos, n_neg, n_hard) in plan.items():
        pos, neg_all, hard = load(split)
        pos_ids = random.sample(sorted(pos), min(n_pos, len(pos)))
        hard_ids = random.sample(sorted(hard), min(n_hard, len(hard)))
        rest = sorted(set(neg_all) - set(hard_ids))
        neg_ids = random.sample(rest, n_neg) + hard_ids
        jobs = [(split, i) for i in pos_ids + neg_ids]
        ok = set()
        with ThreadPoolExecutor(32) as ex:
            for n, (i, good) in enumerate(ex.map(lambda j: fetch(*j), jobs)):
                if good:
                    ok.add(i)
                if n % 500 == 0:
                    print(name, n, "/", len(jobs), flush=True)
        meta[name] = {"pos": {i: pos[i] for i in pos_ids if i in ok}, "neg": [i for i in neg_ids if i in ok]}
        print(name, "pos", len(meta[name]["pos"]), "neg", len(meta[name]["neg"]), "(davon hart", len(hard_ids), ")")
    json.dump(meta, open(ROOT / "meta.json", "w"))
    print("fertig – Attribution der Bilder: Open Images V7, CC BY 2.0", flush=True)


if __name__ == "__main__":
    main()
