"""Utilidades compartidas: datos (YOLO -> COCO), modelo DETR con N queries, entrenamiento y evaluación."""
import contextlib, io, json, random
from pathlib import Path
import numpy as np
import torch
import torchvision
from PIL import Image

VISDRONE_CLASSES = ["pedestrian", "people", "bicycle", "car", "van", "truck",
                    "tricycle", "awning-tricycle", "bus", "motor"]


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------- datos
def yolo_to_coco(images_dir, labels_dir, out_json, class_names=VISDRONE_CLASSES, min_wh=2.0):
    """Convierte etiquetas YOLO (normalizadas) a COCO JSON usando el tamaño REAL de cada imagen.
    category_id = clase_yolo + 1 (evita el id 0)."""
    images_dir, labels_dir = Path(images_dir), Path(labels_dir)
    exts = {".jpg", ".jpeg", ".png"}
    images, anns, aid = [], [], 1
    files = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in exts)
    for iid, p in enumerate(files, start=1):
        with Image.open(p) as im:
            W, H = im.size
        images.append({"id": iid, "file_name": p.name, "width": W, "height": H})
        lf = labels_dir / (p.stem + ".txt")
        if not lf.exists():
            continue
        for line in lf.read_text().strip().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            c = int(float(parts[0])); xc, yc, w, h = map(float, parts[1:5])
            x, y = (xc - w / 2) * W, (yc - h / 2) * H
            x, y = max(0.0, x), max(0.0, y)
            bw, bh = min(w * W, W - x), min(h * H, H - y)
            if bw < min_wh or bh < min_wh:
                continue
            anns.append({"id": aid, "image_id": iid, "category_id": c + 1, "bbox": [x, y, bw, bh],
                         "area": bw * bh, "iscrowd": 0}); aid += 1
    cats = [{"id": i + 1, "name": n} for i, n in enumerate(class_names)]
    json.dump({"images": images, "annotations": anns, "categories": cats}, open(out_json, "w"))
    return len(images), len(anns)


def make_subset(coco_json, n_images, out_json, seed=42):
    d = json.load(open(coco_json))
    rng = random.Random(seed)
    ids = set(rng.sample([im["id"] for im in d["images"]], min(n_images, len(d["images"]))))
    d["images"] = [im for im in d["images"] if im["id"] in ids]
    d["annotations"] = [a for a in d["annotations"] if a["image_id"] in ids]
    json.dump(d, open(out_json, "w"))
    return len(d["images"]), len(d["annotations"])


def objects_per_image(coco_json):
    d = json.load(open(coco_json))
    cnt = {im["id"]: 0 for im in d["images"]}
    for a in d["annotations"]:
        cnt[a["image_id"]] += 1
    return cnt


def density_groups(coco_json):
    """Terciles de objetos por imagen -> {'low','medium','high'} con listas de image_id."""
    cnt = objects_per_image(coco_json)
    ids = np.array(list(cnt)); n = np.array([cnt[i] for i in ids])
    q1, q2 = np.quantile(n, [1 / 3, 2 / 3])
    return {"low": ids[n <= q1].tolist(), "medium": ids[(n > q1) & (n <= q2)].tolist(),
            "high": ids[n > q2].tolist()}, (float(q1), float(q2))


def oracle_recall(cnt, N):
    """Techo de recall (sin errores del modelo): sum(min(n, N)) / sum(n)."""
    n = np.array(list(cnt.values()))
    return float(np.minimum(n, N).sum() / max(1, n.sum()))


class CocoDet(torchvision.datasets.CocoDetection):
    """shift=1 -> category_id-1 (etiquetas 0..K-1 para fine-tuning); shift=0 -> sin cambios (COCO original)."""
    def __init__(self, img_folder, ann_file, processor, shift=1):
        super().__init__(img_folder, ann_file)
        self.processor, self.shift = processor, shift

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        for a in target:
            a["category_id"] = a["category_id"] - self.shift
        enc = self.processor(images=img, annotations={"image_id": self.ids[idx], "annotations": target},
                             return_tensors="pt")
        return enc["pixel_values"].squeeze(0), enc["labels"][0]


def make_loader(img_dir, ann_json, processor, bs=2, shuffle=False, shift=1, workers=2):
    ds = CocoDet(img_dir, ann_json, processor, shift)

    def collate(batch):
        enc = processor.pad([b[0] for b in batch], return_tensors="pt")
        return {"pixel_values": enc["pixel_values"], "pixel_mask": enc["pixel_mask"],
                "labels": [b[1] for b in batch]}
    return torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=shuffle, collate_fn=collate,
                                       num_workers=workers)


# ---------------------------------------------------------------- modelo
def build_model(num_queries=100, num_labels=10, ckpt="facebook/detr-resnet-50", freeze_backbone=True,
                noise_scale=0.1):
    """DETR preentrenado en COCO con cabeza nueva (num_labels) y N object queries.
    Las primeras min(N,100) queries se copian del preentrenado; las nuevas (N>100) se inicializan como
    copia de queries preentrenadas aleatorias + ruido gaussiano (noise_scale * std)."""
    from transformers import DetrForObjectDetection
    m = DetrForObjectDetection.from_pretrained(ckpt, num_labels=num_labels, ignore_mismatched_sizes=True)
    old = m.model.query_position_embeddings.weight.data.clone()
    if num_queries != old.shape[0]:
        new = torch.nn.Embedding(num_queries, old.shape[1])
        k = min(num_queries, old.shape[0])
        new.weight.data[:k] = old[:k]
        if num_queries > k:
            g = torch.Generator().manual_seed(0)
            idx = torch.randint(0, old.shape[0], (num_queries - k,), generator=g)
            new.weight.data[k:] = old[idx] + torch.randn(num_queries - k, old.shape[1], generator=g) * old.std() * noise_scale
        m.model.query_position_embeddings = new
    m.config.num_queries = num_queries
    if freeze_backbone:
        for p in m.model.backbone.parameters():
            p.requires_grad = False
    return m


def train(model, loader, epochs, lr=1e-4, lr_backbone=1e-5, wd=1e-4, device="cuda", amp=True, log=print):
    model.to(device).train()
    groups = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad], "lr": lr},
        {"params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad], "lr": lr_backbone},
    ]
    opt = torch.optim.AdamW([g for g in groups if len(g["params"]) > 0], weight_decay=wd)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    hist = []
    for ep in range(epochs):
        tot, n = 0.0, 0
        for b in loader:
            pv, pm = b["pixel_values"].to(device), b["pixel_mask"].to(device)
            labels = [{k: v.to(device) for k, v in t.items()} for t in b["labels"]]
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                loss = model(pixel_values=pv, pixel_mask=pm, labels=labels).loss
            opt.zero_grad()
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            scaler.step(opt); scaler.update()
            tot += loss.item(); n += 1
        hist.append(tot / n); log(f"época {ep + 1}/{epochs}  loss={hist[-1]:.4f}")
    return hist


# ---------------------------------------------------------------- evaluación
def coco_stat(E, ap=1, iouThr=None, areaRng="all", maxDets=100):
    """Equivalente a COCOeval._summarize (que es una función local y no se puede llamar desde fuera)."""
    p = E.params
    aind = [i for i, a in enumerate(p.areaRngLbl) if a == areaRng]
    mind = [i for i, m in enumerate(p.maxDets) if m == maxDets]
    s = E.eval["precision"] if ap == 1 else E.eval["recall"]
    if iouThr is not None:
        s = s[np.where(np.isclose(p.iouThrs, iouThr))[0]]
    s = s[:, :, :, aind, mind] if ap == 1 else s[:, :, aind, mind]
    s = s[s > -1]
    return float(np.mean(s)) if s.size else float("nan")


@torch.no_grad()
def predict(model, processor, img_dir, ann_json, device="cuda", bs=4, label_offset=1, shift=1):
    """Devuelve las N predicciones por imagen (sin umbral) en formato COCO results."""
    model.eval().to(device)
    loader = make_loader(img_dir, ann_json, processor, bs=bs, shuffle=False, shift=shift)
    dets = []
    for b in loader:
        out = model(pixel_values=b["pixel_values"].to(device), pixel_mask=b["pixel_mask"].to(device))
        sizes = torch.stack([t["orig_size"] for t in b["labels"]])
        res = processor.post_process_object_detection(out, threshold=0.0, target_sizes=sizes)
        for t, r in zip(b["labels"], res):
            iid = int(t["image_id"].item())
            for box, sc, lb in zip(r["boxes"].tolist(), r["scores"].tolist(), r["labels"].tolist()):
                dets.append({"image_id": iid, "category_id": int(lb) + label_offset, "score": float(sc),
                             "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]]})
    return dets


def coco_eval(ann_json, dets, img_ids=None, max_dets=(1, 10, 100, 500)):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(ann_json); dt = gt.loadRes(dets)
        E = COCOeval(gt, dt, "bbox"); E.params.maxDets = list(max_dets)
        if img_ids is not None:
            E.params.imgIds = list(img_ids)
        E.evaluate(); E.accumulate()
    return {"AP": coco_stat(E, 1, maxDets=100), "AP50": coco_stat(E, 1, iouThr=0.5, maxDets=100),
            "AP_S": coco_stat(E, 1, areaRng="small", maxDets=100), "AP_M": coco_stat(E, 1, areaRng="medium", maxDets=100),
            "AP_L": coco_stat(E, 1, areaRng="large", maxDets=100), "AP@500dets": coco_stat(E, 1, maxDets=500),
            "AR@100": coco_stat(E, 0, maxDets=100), "AR@500": coco_stat(E, 0, maxDets=500)}


def evaluate_all(model, processor, img_dir, ann_json, groups=None, **kw):
    dets = predict(model, processor, img_dir, ann_json, **kw)
    res = {"overall": coco_eval(ann_json, dets)}
    for name, ids in (groups or {}).items():
        res[name] = coco_eval(ann_json, dets, img_ids=ids)
    return res
