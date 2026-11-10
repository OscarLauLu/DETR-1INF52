"""E3 · Fase B corregida: ajuste fino de DETR-R50 en VisDrone con la cabeza de clases inicializada
desde COCO (D19), backbone entrenable a lr 1e-5 (D20), 1500 imágenes y 20 épocas (D21).
La única diferencia entre variantes sigue siendo N (número de object queries).

Uso (lo lanza el notebook 04):
  python train_e3.py prep  --train_img ... --train_json ... --val_img ... --val_json ... --cache /tmp/e3cache
  python train_e3.py train --N 100 --cache /tmp/e3cache --val_json ... --out ../results/e3 [--zero_shot]

El preprocesamiento se hace una sola vez (D18) y se guarda en disco como un arreglo fp16 relleno a
512 x 853; los dos procesos (uno por GPU) lo leen con memmap, sin duplicar la RAM (D22).
Curso 1INF52 Deep Learning, PUCP 2026-2 (Grupo 3).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (CKPT, COCO_TO_VISDRONE, VISDRONE_CLASSES, coco_eval, density_groups,  # noqa: E402
                    per_image_recall, pretrained_queries, set_seed)

SHORT, LONG = 512, 853                                   # D4

# D19 · Fila de la cabeza de COCO con la que se inicializa cada clase de VisDrone (índice de etiqueta
# del modelo HF = id de categoría COCO; 91 = "no object"). Las 6 clases con equivalente usan su clase;
# las 4 restantes, la clase COCO más cercana en apariencia.
VISDRONE_TO_COCO_INIT = {
    'pedestrian': 1,        # person
    'people': 1,            # person (personas sentadas o agachadas)
    'bicycle': 2,           # bicycle
    'car': 3,               # car
    'van': 3,               # car
    'truck': 8,             # truck
    'tricycle': 4,          # motorcycle
    'awning-tricycle': 4,   # motorcycle
    'bus': 6,               # bus
    'motor': 4,             # motorcycle
}
COCO_NO_OBJECT = 91


# ============================================================================ caché en disco
def prep(args):
    import torch
    from transformers import DetrImageProcessor
    from common import CocoDetrDataset
    processor = DetrImageProcessor.from_pretrained(args.ckpt, size={'shortest_edge': SHORT, 'longest_edge': LONG})
    os.makedirs(args.cache, exist_ok=True)
    for split, img_dir, ann in (('train', args.train_img, args.train_json), ('val', args.val_img, args.val_json)):
        t = time.time()
        ds = CocoDetrDataset(img_dir, ann, processor)
        n = len(ds)
        arr = np.lib.format.open_memmap(f'{args.cache}/{split}_pv.npy', mode='w+', dtype=np.float16,
                                        shape=(n, 3, SHORT, LONG))
        hw, labels = np.zeros((n, 2), np.int32), []
        dl = torch.utils.data.DataLoader(ds, batch_size=None, shuffle=False, num_workers=os.cpu_count())
        for i, (pv, lab) in enumerate(dl):
            _, h, w = pv.shape
            if h > SHORT or w > LONG:                     # imagen vertical: no se espera en VisDrone
                raise ValueError(f'imagen de {h}x{w} no cabe en la caché de {SHORT}x{LONG}')
            arr[i, :, :h, :w] = pv.numpy().astype(np.float16)
            hw[i] = (h, w)
            labels.append({k: (v.clone() if hasattr(v, 'clone') else v) for k, v in lab.items()})
        arr.flush(); del arr
        np.save(f'{args.cache}/{split}_hw.npy', hw)
        torch.save(labels, f'{args.cache}/{split}_labels.pt')
        print(f'{split}: {n} imágenes en {time.time() - t:.0f} s')


class Cached:
    """Lee la caché fp16 con memmap; recorta al tamaño real de cada imagen."""

    def __init__(self, cache, split):
        import torch
        self.pv = np.load(f'{cache}/{split}_pv.npy', mmap_mode='r')
        self.hw = np.load(f'{cache}/{split}_hw.npy')
        self.labels = torch.load(f'{cache}/{split}_labels.pt', weights_only=False)

    def __len__(self):
        return len(self.hw)

    def __getitem__(self, i):
        import torch
        h, w = self.hw[i]
        return torch.from_numpy(np.array(self.pv[i, :, :h, :w], dtype=np.float32)), self.labels[i]


def pad_batch(pvs):
    """Equivalente a DetrImageProcessor.pad: relleno con ceros abajo/derecha y máscara de píxeles."""
    import torch
    H, W = max(p.shape[1] for p in pvs), max(p.shape[2] for p in pvs)
    out, mask = torch.zeros(len(pvs), 3, H, W), torch.zeros(len(pvs), H, W, dtype=torch.long)
    for i, p in enumerate(pvs):
        out[i, :, :p.shape[1], :p.shape[2]] = p
        mask[i, :p.shape[1], :p.shape[2]] = 1
    return out, mask


# ============================================================================ modelo
def build_model_e3(N, ckpt=CKPT, query_noise=0.1, seed=42, init_from_coco=True):
    """DETR-R50 de COCO con N queries y 10 clases de VisDrone.
    - Queries: misma inicialización que en el E2 (D6), idéntica para todo N.
    - Cabeza de clases (D19): fila k <- fila COCO de VISDRONE_TO_COCO_INIT; 'no object' <- 'no object' de COCO.
    - Backbone (D20): se deja la configuración de DETR (stem y layer1 congelados, layer2-4 entrenables)."""
    import contextlib, io
    import torch
    from transformers import DetrForObjectDetection
    id2label = dict(enumerate(VISDRONE_CLASSES))
    with contextlib.redirect_stderr(io.StringIO()):
        model = DetrForObjectDetection.from_pretrained(
            ckpt, num_queries=N, num_labels=10, id2label=id2label,
            label2id={v: k for k, v in id2label.items()}, ignore_mismatched_sizes=True)
        coco = DetrForObjectDetection.from_pretrained(ckpt)
    Q = coco.model.query_position_embeddings.weight.detach()
    g = torch.Generator().manual_seed(seed)
    W = Q[torch.arange(N) % Q.shape[0]] + query_noise * Q.std() * torch.randn(N, Q.shape[1], generator=g)
    with torch.no_grad():
        model.model.query_position_embeddings.weight.copy_(W)
        if init_from_coco:
            src_w, src_b = coco.class_labels_classifier.weight, coco.class_labels_classifier.bias
            rows = [VISDRONE_TO_COCO_INIT[c] for c in VISDRONE_CLASSES] + [COCO_NO_OBJECT]
            model.class_labels_classifier.weight.copy_(src_w[rows])
            model.class_labels_classifier.bias.copy_(src_b[rows])
        # comprobación: el resto de pesos (cajas, transformer, backbone) viene de COCO sin cambios
        for k in ('bbox_predictor.layers.2.weight', 'model.decoder.layers.5.fc2.weight'):
            assert torch.equal(model.state_dict()[k], coco.state_dict()[k]), k
    del coco
    return model


def param_groups(model, lr, lr_backbone, wd, lr_head=None):
    """Sin lr_head, los mismos dos grupos del E3. Con lr_head (D25), la cabeza de clases va en un tercer grupo."""
    head = lambda n: lr_head is not None and n.startswith('class_labels_classifier')
    bb = [p for n, p in model.named_parameters() if p.requires_grad and 'backbone' in n]
    rest = [p for n, p in model.named_parameters() if p.requires_grad and 'backbone' not in n and not head(n)]
    hd = [p for n, p in model.named_parameters() if p.requires_grad and head(n)]
    groups = [{'params': rest, 'lr': lr, 'weight_decay': wd},
              {'params': bb, 'lr': lr_backbone, 'weight_decay': wd}]
    if hd:
        groups.append({'params': hd, 'lr': lr_head, 'weight_decay': wd})
    return groups, sum(p.numel() for p in bb), sum(p.numel() for p in rest + hd)


# ============================================================================ predicción y evaluación
def predict(model, val, val_json, device, label_offset=1, label_map=None, bs=4, max_imgs=None):
    import torch
    from transformers import DetrImageProcessor
    post = DetrImageProcessor()                          # solo para post_process_object_detection
    imgs = sorted(json.load(open(val_json))['images'], key=lambda x: x['id'])[:max_imgs]
    model.to(device).eval()
    dets = []
    for s in range(0, len(imgs), bs):
        chunk = imgs[s:s + bs]
        pv, mask = pad_batch([val[i][0] for i in range(s, s + len(chunk))])
        with torch.no_grad():
            out = model(pixel_values=pv.to(device), pixel_mask=mask.to(device))
        sizes = torch.tensor([[im['height'], im['width']] for im in chunk])
        for im, r in zip(chunk, post.post_process_object_detection(out, threshold=0.0, target_sizes=sizes)):
            for sc, lb, bx in zip(r['scores'].tolist(), r['labels'].tolist(), r['boxes'].tolist()):
                cid = lb + label_offset
                if label_map is not None:
                    if cid not in label_map:
                        continue
                    cid = label_map[cid]
                dets.append({'image_id': im['id'], 'category_id': int(cid), 'score': float(sc),
                             'bbox': [bx[0], bx[1], bx[2] - bx[0], bx[3] - bx[1]]})
    return dets


def full_eval(val_json, dets, groups, cat_ids=None):
    MAPPED = sorted(set(COCO_TO_VISDRONE.values()))
    res = {'overall': coco_eval(val_json, dets, cat_ids=cat_ids)}
    for g, ids in groups.items():
        res[g] = coco_eval(val_json, dets, img_ids=ids, cat_ids=cat_ids)
    if cat_ids is None:
        res['mapped6'] = coco_eval(val_json, dets, cat_ids=MAPPED)
        res['por_clase'] = {VISDRONE_CLASSES[c - 1]: coco_eval(val_json, dets, cat_ids=[c])
                            for c in range(1, 11)}
    # recall por imagen a IoU 0.5 (para el techo min(1, N/n)) y distribución de clases predichas
    R = per_image_recall(val_json, dets, cat_ids=cat_ids)
    res['recall_img'] = [{k: r[k] for k in ('image_id', 'n_gt', 'tp')} for r in R]
    cls = np.bincount([d['category_id'] for d in dets], minlength=11)[1:]
    res['dets_por_clase'] = {VISDRONE_CLASSES[i]: int(cls[i]) for i in range(10)}
    return res


# ============================================================================ entrenamiento
def train(args):
    import torch
    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.out, exist_ok=True)
    tag = f'N{args.N}'
    res_path = f'{args.out}/{tag}.json'
    groups, _ = density_groups(args.val_json)
    tr, val = Cached(args.cache, 'train'), Cached(args.cache, 'val')
    print(f'[{tag}] {device} | train {len(tr)} | val {len(val)}', flush=True)

    if args.zero_shot:                                   # punto de partida bajo la misma tubería
        from transformers import DetrForObjectDetection
        zs = DetrForObjectDetection.from_pretrained(args.ckpt)
        d = predict(zs, val, args.val_json, device, label_offset=0, label_map=COCO_TO_VISDRONE,
                    max_imgs=args.max_val)
        json.dump(full_eval(args.val_json, d, groups, cat_ids=sorted(set(COCO_TO_VISDRONE.values()))),
                  open(f'{args.out}/zero_shot.json', 'w'))
        del zs; torch.cuda.empty_cache()
        print(f'[{tag}] zero-shot listo', flush=True)

    model = build_model_e3(args.N, ckpt=args.ckpt, init_from_coco=not args.no_coco_head).to(device)
    groups_opt, n_bb, n_rest = param_groups(model, args.lr, args.lr_backbone, args.wd, args.lr_head)
    opt = torch.optim.AdamW(groups_opt)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=args.lr_drop, gamma=0.1)
    print(f'[{tag}] entrenables: backbone {n_bb / 1e6:.2f} M, resto {n_rest / 1e6:.2f} M', flush=True)

    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)   # D24: sin --amp, idéntico al E3
    g = torch.Generator(); g.manual_seed(args.seed)
    loader = torch.utils.data.DataLoader(
        tr, batch_size=args.bs, shuffle=True, generator=g, num_workers=2,
        collate_fn=lambda b: (*pad_batch([x[0] for x in b]), [x[1] for x in b]))
    log = {'N': args.N, 'config': vars(args), 'loss': [], 'eval': {}}
    eval_at = {int(e) for e in args.eval_at.split(',') if e} | {args.epochs}
    t_start = time.time()
    for ep in range(1, args.epochs + 1):
        model.train(); t0, tot, comps, nb = time.time(), 0.0, {}, 0
        for it, (pv, mask, labels) in enumerate(loader):
            if args.max_iters and it >= args.max_iters:
                break
            labels = [{k: v.to(device) for k, v in t.items()} for t in labels]
            with torch.autocast('cuda', dtype=torch.float16, enabled=args.amp):
                out = model(pixel_values=pv.to(device), pixel_mask=mask.to(device), labels=labels)
            opt.zero_grad(set_to_none=True)
            scaler.scale(out.loss).backward()
            scaler.unscale_(opt)                          # antes del clip: recorta gradientes reales
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(opt)                              # omite el paso si hubo gradientes no finitos
            scaler.update()
            if not torch.isfinite(out.loss):
                raise RuntimeError(f'pérdida no finita en ep {ep} it {it}')
            tot += out.loss.item(); nb += 1
            for k, v in (out.loss_dict or {}).items():
                comps[k] = comps.get(k, 0.0) + float(v)
        sched.step()
        rec = {'epoch': ep, 'loss': tot / max(1, nb), 'seg': round(time.time() - t0, 1),
               **{k: v / max(1, nb) for k, v in comps.items()}}
        log['loss'].append(rec)
        print(f'[{tag}] época {ep}/{args.epochs}: loss {rec["loss"]:.4f} ({rec["seg"]} s)', flush=True)
        if ep in eval_at:
            t1 = time.time()
            dets = predict(model, val, args.val_json, device, max_imgs=args.max_val)
            ev = full_eval(args.val_json, dets, groups)
            log['eval'][ep] = ev
            print(f'[{tag}] eval ep {ep}: AP {ev["overall"]["AP"]:.4f} | AP mapped6 {ev["mapped6"]["AP"]:.4f}'
                  f' | AR@500 gt100 {ev["gt100"]["AR@500"]:.4f} ({time.time() - t1:.0f} s)', flush=True)
            if ep == args.epochs:
                json.dump(dets, open(f'{args.out}/dets_{tag}.json', 'w'))
        log['minutos'] = round((time.time() - t_start) / 60, 1)
        json.dump(log, open(res_path, 'w'))             # se guarda en cada época: resultados parciales
    print(f'[{tag}] FIN en {log["minutos"]} min', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['prep', 'train'])
    ap.add_argument('--ckpt', default=CKPT)
    ap.add_argument('--train_img'); ap.add_argument('--train_json')
    ap.add_argument('--val_img'); ap.add_argument('--val_json')
    ap.add_argument('--cache', default='/tmp/e3cache')
    ap.add_argument('--out', default='../results/e3')
    ap.add_argument('--N', type=int, default=100)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--lr_drop', type=int, default=14)
    ap.add_argument('--eval_at', default='8,14')
    ap.add_argument('--bs', type=int, default=2)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--lr_backbone', type=float, default=1e-5)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--clip', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--zero_shot', action='store_true')
    ap.add_argument('--no_coco_head', action='store_true')
    ap.add_argument('--lr_head', type=float, default=None)   # D25: lr propio de la cabeza de clases
    ap.add_argument('--amp', action='store_true')            # D24: precisión mixta fp16 en el entrenamiento
    ap.add_argument('--max_iters', type=int, default=0)      # solo para pruebas
    ap.add_argument('--max_val', type=int, default=None)     # solo para pruebas
    a = ap.parse_args()
    prep(a) if a.mode == 'prep' else train(a)
