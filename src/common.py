"""Utilidades compartidas: datos (VisDrone YOLO -> COCO), modelo DETR con N queries,
entrenamiento y evaluación (COCOeval con maxDets hasta 500, recall por imagen vs techo min(1, N/n)).

Curso 1INF52 Deep Learning, PUCP 2026-2 (Grupo 3).
Las funciones de datos y análisis (yolo_to_coco, make_subset, objects_per_image, oracle_recall,
density_groups, per_image_recall) solo usan numpy/PIL; torch y transformers se importan
dentro de las funciones que los necesitan.
"""
import contextlib
import io
import json
import os
import random
import time

import numpy as np
from PIL import Image

CKPT = 'facebook/detr-resnet-50'

# Orden estándar de la copia YOLO de VisDrone-DET (10 clases; 'ignored regions' y 'others' ya excluidas).
# VERIFICAR contra el data.yaml de la copia de Kaggle (ver read_yolo_names).
VISDRONE_CLASSES = ['pedestrian', 'people', 'bicycle', 'car', 'van',
                    'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']

# Mapeo COCO (id de categoría) -> VisDrone (id en nuestro JSON = índice YOLO + 1).
# Solo 6 de 10 clases tienen equivalente: van, tricycle, awning-tricycle y people quedan fuera
# en la evaluación zero-shot. 'person' se asigna a 'pedestrian' (people = personas sentadas/no de pie).
COCO_TO_VISDRONE = {1: 1,   # person     -> pedestrian
                    2: 3,   # bicycle    -> bicycle
                    3: 4,   # car        -> car
                    4: 10,  # motorcycle -> motor
                    6: 9,   # bus        -> bus
                    8: 6}   # truck      -> truck

IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp')


# ----------------------------------------------------------------------------- reproducibilidad
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass


def env_report(path=None):
    """Versiones exactas del entorno (para el README / design log)."""
    lines = []
    import platform, sys
    lines.append(f'python {sys.version.split()[0]} | {platform.platform()}')
    for mod in ('torch', 'torchvision', 'transformers', 'timm', 'pycocotools', 'scipy', 'numpy'):
        try:
            m = __import__(mod)
            lines.append(f'{mod} {getattr(m, "__version__", "?")}')
        except ImportError:
            lines.append(f'{mod} (no instalado)')
    try:
        import torch
        if torch.cuda.is_available():
            lines.append(f'GPU {torch.cuda.get_device_name(0)} | CUDA {torch.version.cuda}')
    except ImportError:
        pass
    txt = '\n'.join(lines)
    if path:
        with open(path, 'w') as f:
            f.write(txt + '\n')
    return txt


# ----------------------------------------------------------------------------- datos
def read_yolo_names(yaml_path):
    """Lee los nombres de clase de un data.yaml de YOLO (sin depender de pyyaml)."""
    try:
        import yaml
        d = yaml.safe_load(open(yaml_path))
        names = d['names']
        return [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
    except ImportError:
        txt = open(yaml_path).read()
        raise RuntimeError('Instala pyyaml o revisa manualmente el campo names:\n' + txt)


def yolo_to_coco(img_dir, lbl_dir, out_json, class_names=VISDRONE_CLASSES, min_size=2.0):
    """Convierte etiquetas YOLO (cls cx cy w h normalizados) a COCO JSON.
    Usa el tamaño REAL de cada imagen (VisDrone tiene tamaños variables) y descarta cajas
    con ancho o alto < min_size px (D8). category_id = índice YOLO + 1."""
    images, anns = [], []
    stats = {'imagenes': 0, 'cajas': 0, 'descartadas_pequenas': 0, 'sin_label': 0, 'clase_invalida': 0}
    files = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(IMG_EXT))
    aid = 1
    for iid, fn in enumerate(files, 1):
        with Image.open(os.path.join(img_dir, fn)) as im:   # solo lee la cabecera
            W, H = im.size
        images.append({'id': iid, 'file_name': fn, 'width': W, 'height': H})
        stats['imagenes'] += 1
        lp = os.path.join(lbl_dir, os.path.splitext(fn)[0] + '.txt')
        if not os.path.exists(lp):
            stats['sin_label'] += 1
            continue
        for line in open(lp):
            p = line.split()
            if len(p) < 5:
                continue
            c = int(float(p[0]))
            if not 0 <= c < len(class_names):
                stats['clase_invalida'] += 1
                continue
            cx, cy, w, h = (float(v) for v in p[1:5])
            x0, y0 = max(0.0, (cx - w / 2) * W), max(0.0, (cy - h / 2) * H)
            x1, y1 = min(W, (cx + w / 2) * W), min(H, (cy + h / 2) * H)
            bw, bh = x1 - x0, y1 - y0
            if bw < min_size or bh < min_size:
                stats['descartadas_pequenas'] += 1
                continue
            anns.append({'id': aid, 'image_id': iid, 'category_id': c + 1,
                         'bbox': [round(x0, 2), round(y0, 2), round(bw, 2), round(bh, 2)],
                         'area': round(bw * bh, 2), 'iscrowd': 0})
            aid += 1
            stats['cajas'] += 1
    cats = [{'id': i + 1, 'name': n} for i, n in enumerate(class_names)]
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    json.dump({'images': images, 'annotations': anns, 'categories': cats}, open(out_json, 'w'))
    return stats


def make_subset(in_json, n, out_json, seed=42):
    """Submuestreo aleatorio (semilla fija) de n imágenes con sus anotaciones.
    n=None o n >= total -> se usan todas."""
    d = json.load(open(in_json))
    imgs = sorted(d['images'], key=lambda x: x['id'])
    if n is not None and n < len(imgs):
        rng = random.Random(seed)
        imgs = sorted(rng.sample(imgs, n), key=lambda x: x['id'])
    keep = {im['id'] for im in imgs}
    anns = [a for a in d['annotations'] if a['image_id'] in keep]
    json.dump({**d, 'images': imgs, 'annotations': anns}, open(out_json, 'w'))
    return {'imagenes': len(imgs), 'cajas': len(anns), 'cajas_por_imagen': round(len(anns) / max(1, len(imgs)), 1)}


def objects_per_image(ann_json):
    """{image_id: número de objetos} (incluye imágenes sin objetos)."""
    d = json.load(open(ann_json)) if isinstance(ann_json, str) else ann_json
    cnt = {im['id']: 0 for im in d['images']}
    for a in d['annotations']:
        if not a.get('iscrowd', 0):
            cnt[a['image_id']] += 1
    return cnt


def oracle_recall(cnt, N):
    """Techo de recall a nivel de dataset con N queries: sum(min(n_i, N)) / sum(n_i).
    Es una cota analítica (un detector perfecto con N queries), no una medición."""
    n = np.asarray(list(cnt.values()) if isinstance(cnt, dict) else cnt, dtype=float)
    return float(np.minimum(n, N).sum() / max(1.0, n.sum()))


def density_groups(ann_json, extra_thresholds=(100,)):
    """Agrupa imágenes por densidad: terciles de objetos/imagen (low/medium/high)
    y, además, grupos 'gt{T}' con más de T objetos (los que superan el techo de N=T)."""
    cnt = objects_per_image(ann_json)
    v = np.array(list(cnt.values()))
    q1, q2 = (float(x) for x in np.quantile(v, [1 / 3, 2 / 3]))
    g = {'low': [i for i, c in cnt.items() if c <= q1],
         'medium': [i for i, c in cnt.items() if q1 < c <= q2],
         'high': [i for i, c in cnt.items() if c > q2]}
    for t in extra_thresholds:
        g[f'gt{t}'] = [i for i, c in cnt.items() if c > t]
    return g, (q1, q2)


# ----------------------------------------------------------------------------- dataset / loader
class CocoDetrDataset:
    """Dataset COCO -> (pixel_values, labels) con el DetrImageProcessor.
    Las etiquetas del JSON (1..K) se pasan al modelo como 0..K-1 (label_shift=-1)."""

    def __init__(self, img_dir, ann_json, processor, label_shift=-1):
        d = json.load(open(ann_json))
        self.img_dir, self.processor, self.shift = img_dir, processor, label_shift
        self.images = sorted(d['images'], key=lambda x: x['id'])
        self.anns = {im['id']: [] for im in self.images}
        for a in d['annotations']:
            self.anns[a['image_id']].append(a)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        im = self.images[i]
        img = Image.open(os.path.join(self.img_dir, im['file_name'])).convert('RGB')
        anns = [{**a, 'category_id': a['category_id'] + self.shift} for a in self.anns[im['id']]]
        enc = self.processor(images=img, annotations={'image_id': im['id'], 'annotations': anns},
                             return_tensors='pt')
        return enc['pixel_values'][0], enc['labels'][0]


def make_loader(img_dir, ann_json, processor, bs=2, shuffle=True, num_workers=2, seed=42):
    import torch
    from torch.utils.data import DataLoader
    ds = CocoDetrDataset(img_dir, ann_json, processor)

    def collate(batch):
        pv = [b[0] for b in batch]
        enc = processor.pad(pv, return_tensors='pt')
        return {'pixel_values': enc['pixel_values'], 'pixel_mask': enc['pixel_mask'],
                'labels': [b[1] for b in batch]}

    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, collate_fn=collate,
                      num_workers=num_workers, generator=g)


# ----------------------------------------------------------------------------- modelo
_PRETRAINED_QUERIES = {}


def pretrained_queries(ckpt=CKPT):
    """Embeddings de las 100 queries preentrenadas en COCO (100 x 256)."""
    if ckpt not in _PRETRAINED_QUERIES:
        from transformers import DetrForObjectDetection
        m = DetrForObjectDetection.from_pretrained(ckpt)
        _PRETRAINED_QUERIES[ckpt] = m.model.query_position_embeddings.weight.detach().clone()
        del m
    return _PRETRAINED_QUERIES[ckpt]


def build_model(num_queries=100, num_labels=10, freeze_backbone=True, ckpt=CKPT,
                class_names=VISDRONE_CLASSES, query_noise=0.1, seed=42):
    """DETR-R50 preentrenado en COCO con N queries y cabeza de clases nueva (num_labels + 'no object').

    Inicialización de queries (D6), IDÉNTICA para todos los N, incluido el control N=100:
        q_i = Q_pre[i mod 100] + query_noise * std(Q_pre) * ruido gaussiano
    Así N=100 no conserva las queries "limpias" mientras las demás se reinician: el único
    cambio entre corridas es N.
    """
    import torch
    from transformers import DetrForObjectDetection
    id2label = {i: n for i, n in enumerate(class_names[:num_labels])}
    with contextlib.redirect_stderr(io.StringIO()):     # oculta el aviso esperado de pesos no coincidentes
        model = DetrForObjectDetection.from_pretrained(
            ckpt, num_queries=num_queries, num_labels=num_labels, id2label=id2label,
            label2id={v: k for k, v in id2label.items()}, ignore_mismatched_sizes=True)
    Q = pretrained_queries(ckpt)
    g = torch.Generator().manual_seed(seed)
    idx = torch.arange(num_queries) % Q.shape[0]
    W = Q[idx] + query_noise * Q.std() * torch.randn(num_queries, Q.shape[1], generator=g)
    with torch.no_grad():
        model.model.query_position_embeddings.weight.copy_(W)
    if freeze_backbone:
        for p in model.model.backbone.parameters():
            p.requires_grad = False
    return model


def count_params(model, trainable_only=False):
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


# ----------------------------------------------------------------------------- entrenamiento
def train(model, loader, epochs, device='cuda', lr=1e-4, weight_decay=1e-4, clip=0.1,
          ckpt_path=None, log_every=100):
    """Fine-tuning con AdamW (lr 1e-4, wd 1e-4, clip 0.1 como en DETR), fp32.
    Devuelve el historial: pérdida media por época y sus componentes."""
    import torch
    model.to(device).train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    hist = []
    for ep in range(epochs):
        t0, tot, comps, nb = time.time(), 0.0, {}, 0
        for it, batch in enumerate(loader):
            labels = [{k: v.to(device) for k, v in t.items()} for t in batch['labels']]
            out = model(pixel_values=batch['pixel_values'].to(device),
                        pixel_mask=batch['pixel_mask'].to(device), labels=labels)
            opt.zero_grad(set_to_none=True)
            out.loss.backward()
            if clip:
                torch.nn.utils.clip_grad_norm_(params, clip)
            opt.step()
            tot += out.loss.item()
            nb += 1
            for k, v in (out.loss_dict or {}).items():
                comps[k] = comps.get(k, 0.0) + float(v)
            if log_every and it % log_every == 0:
                print(f'  ep {ep + 1} it {it}/{len(loader)} loss {out.loss.item():.3f}')
        rec = {'epoch': ep + 1, 'loss': tot / max(1, nb), 'seg': round(time.time() - t0, 1),
               **{k: v / max(1, nb) for k, v in comps.items()}}
        hist.append(rec)
        print(f'época {ep + 1}/{epochs}: loss {rec["loss"]:.4f} ({rec["seg"]} s)')
        if ckpt_path:
            torch.save(model.state_dict(), ckpt_path)
    model.eval()
    return hist


# ----------------------------------------------------------------------------- inferencia
def predict(model, processor, img_dir, ann_json, device='cuda', label_offset=1, label_map=None,
            bs=4, threshold=0.0):
    """Detecciones en formato COCO results.
    DETR emite UNA detección por query (clase argmax sin 'no object'), así que una imagen
    nunca recibe más de N detecciones: el techo min(1, N/n) se respeta en la evaluación (D10).
      label_offset: category_id = etiqueta predicha + label_offset (1 para nuestros modelos VisDrone,
                    0 para el modelo COCO original, cuyas etiquetas ya son ids COCO).
      label_map:    dict {id predicho -> id del JSON}; las predicciones fuera del mapa se descartan
                    (evaluación zero-shot COCO -> VisDrone)."""
    import torch
    d = json.load(open(ann_json))
    imgs = sorted(d['images'], key=lambda x: x['id'])
    model.to(device).eval()
    dets = []
    for s in range(0, len(imgs), bs):
        chunk = imgs[s:s + bs]
        pil = [Image.open(os.path.join(img_dir, im['file_name'])).convert('RGB') for im in chunk]
        enc = processor(images=pil, return_tensors='pt')
        with torch.no_grad():
            out = model(pixel_values=enc['pixel_values'].to(device), pixel_mask=enc['pixel_mask'].to(device))
        sizes = torch.tensor([[p.height, p.width] for p in pil])
        res = processor.post_process_object_detection(out, threshold=threshold, target_sizes=sizes)
        for im, r in zip(chunk, res):
            for sc, lb, bx in zip(r['scores'].tolist(), r['labels'].tolist(), r['boxes'].tolist()):
                cid = lb + label_offset
                if label_map is not None:
                    if cid not in label_map:
                        continue
                    cid = label_map[cid]
                x0, y0, x1, y1 = bx
                dets.append({'image_id': im['id'], 'category_id': int(cid), 'score': float(sc),
                             'bbox': [x0, y0, x1 - x0, y1 - y0]})
    return dets


# ----------------------------------------------------------------------------- evaluación
_AREAS = {'all': 0, 'small': 1, 'medium': 2, 'large': 3}


def _ap(E, iou=None, area='all', md=500):
    p = E.eval['precision']                     # [T, R, K, A, M]
    m = list(E.params.maxDets).index(md)
    if iou is not None:
        p = p[[i for i, t in enumerate(E.params.iouThrs) if abs(t - iou) < 1e-6]]
    s = p[..., _AREAS[area], m]
    s = s[s > -1]
    return float(s.mean()) if s.size else float('nan')


def _ar(E, area='all', md=500):
    r = E.eval['recall']                        # [T, K, A, M]
    s = r[:, :, _AREAS[area], list(E.params.maxDets).index(md)]
    s = s[s > -1]
    return float(s.mean()) if s.size else float('nan')


def coco_eval(gt_json, dets, img_ids=None, cat_ids=None, max_dets=(1, 10, 100, 500)):
    """COCOeval con maxDets ampliado (D7). Con el valor por defecto (100) las variantes
    N=300/500 quedarían truncadas a 100 detecciones por imagen.
    Devuelve AP/AP50/AP75/AP_S/M/L (@maxDets=100, comparables con la literatura),
    AP@500 y AR@100 / AR@500 (+ AR por tamaño a 500)."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(gt_json)
        keys = ['AP', 'AP50', 'AP75', 'AP_S', 'AP_M', 'AP_L', 'AP@500', 'AR@100', 'AR@500',
                'AR_S@500', 'AR_M@500', 'AR_L@500']
        ids = list(img_ids) if img_ids is not None else gt.getImgIds()
        idset = set(ids)
        dets = [x for x in dets if x['image_id'] in idset]
        if not dets or not ids:
            return {k: 0.0 for k in keys} | {'n_img': len(ids)}
        E = COCOeval(gt, gt.loadRes(dets), 'bbox')
        E.params.maxDets = list(max_dets)
        E.params.imgIds = list(ids)
        if cat_ids is not None:
            E.params.catIds = list(cat_ids)
        E.evaluate(); E.accumulate()
    return {'AP': _ap(E, md=100), 'AP50': _ap(E, .5, md=100), 'AP75': _ap(E, .75, md=100),
            'AP_S': _ap(E, area='small', md=100), 'AP_M': _ap(E, area='medium', md=100),
            'AP_L': _ap(E, area='large', md=100), 'AP@500': _ap(E, md=500),
            'AR@100': _ar(E, md=100), 'AR@500': _ar(E, md=500),
            'AR_S@500': _ar(E, 'small'), 'AR_M@500': _ar(E, 'medium'), 'AR_L@500': _ar(E, 'large'),
            'n_img': len(ids)}


def evaluate_all(model, processor, img_dir, ann_json, groups=None, device='cuda', label_offset=1,
                 label_map=None, cat_ids=None, return_dets=False):
    """Predice una vez y evalúa en todo el conjunto ('overall') y en cada grupo de densidad."""
    dets = predict(model, processor, img_dir, ann_json, device=device,
                   label_offset=label_offset, label_map=label_map)
    res = {'overall': coco_eval(ann_json, dets, cat_ids=cat_ids)}
    for g, ids in (groups or {}).items():
        res[g] = coco_eval(ann_json, dets, img_ids=ids, cat_ids=cat_ids)
    return (res, dets) if return_dets else res


def _iou_matrix(a, b):
    """IoU entre cajas xywh: a (n,4), b (m,4) -> (n,m)."""
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2) - np.maximum(ax1[:, None], bx1), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2) - np.maximum(ay1[:, None], by1), 0, None)
    inter = iw * ih
    union = (a[:, 2] * a[:, 3])[:, None] + b[:, 2] * b[:, 3] - inter
    return inter / np.maximum(union, 1e-9)


def per_image_recall(ann_json, dets, iou_thr=0.5, score_thr=0.0, class_aware=True, cat_ids=None):
    """Recall por imagen (IoU >= iou_thr, emparejamiento greedy por score, 1 a 1).
    Sirve para comparar el recall medido con el techo min(1, N/n) imagen por imagen (H1).
    Devuelve una lista de dicts {image_id, n_gt, tp, recall}."""
    d = json.load(open(ann_json))
    keep_cat = (lambda c: True) if cat_ids is None else (lambda c: c in set(cat_ids))
    gts, pr = {}, {}
    for a in d['annotations']:
        if not a.get('iscrowd', 0) and keep_cat(a['category_id']):
            gts.setdefault(a['image_id'], []).append(a)
    for x in dets:
        if x['score'] >= score_thr and keep_cat(x['category_id']):
            pr.setdefault(x['image_id'], []).append(x)
    rows = []
    for im in d['images']:
        g, p = gts.get(im['id'], []), sorted(pr.get(im['id'], []), key=lambda x: -x['score'])
        tp = 0
        if g and p:
            M = _iou_matrix([x['bbox'] for x in p], [x['bbox'] for x in g])
            if class_aware:
                M = M * (np.array([x['category_id'] for x in p])[:, None]
                         == np.array([x['category_id'] for x in g])[None, :])
            used = np.zeros(len(g), bool)
            for i in range(len(p)):
                cand = np.where(~used & (M[i] >= iou_thr))[0]
                if cand.size:
                    used[cand[M[i, cand].argmax()]] = True
                    tp += 1
        rows.append({'image_id': im['id'], 'n_gt': len(g), 'tp': tp,
                     'recall': tp / len(g) if g else float('nan')})
    return rows


# ----------------------------------------------------------------------------- costo (Fase A / H3)
def flops_total(model, pixel_values, pixel_mask):
    """FLOPs totales de una pasada (convención de torch: FLOPs = 2 x MACs)."""
    import torch
    from torch.utils.flop_counter import FlopCounterMode
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(pixel_values=pixel_values, pixel_mask=pixel_mask)
    return fc.get_total_flops()


def latency_ms(model, pixel_values, pixel_mask, runs=30, warmup=10):
    """Latencia media en GPU (ms) con calentamiento y sincronización."""
    import torch
    assert pixel_values.is_cuda, 'medir latencia en GPU'
    with torch.no_grad():
        for _ in range(warmup):
            model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(runs):
            model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / runs * 1000
