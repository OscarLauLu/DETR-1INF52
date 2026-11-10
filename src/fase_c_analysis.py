"""Fase C · Diagnóstico sin GPU a partir de detecciones guardadas.

Separa el efecto de la densidad de escena del de la escala de los objetos (en VisDrone ambas covarían):
recall por objeto (IoU >= 0.5, emparejamiento greedy 1 a 1 por score) estratificado por
  - tamaño del objeto: lado equivalente sqrt(área) en píxeles de la imagen original,
  - densidad de la imagen: objetos por imagen.
También compara el recall con y sin clase (class-aware vs class-agnostic) para separar los errores
de localización de los de clasificación.

Uso: python fase_c_analysis.py --val ../data/val_sub.json --dets ../results/e3/dets_N100.json --tag E3_N100 --out ../results/fase_c
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from common import _iou_matrix

SIZE_BINS = [0, 8, 16, 32, 64, 1e9]           # lado equivalente en px (COCO: small < 32)
SIZE_LBL = ['<8', '8-16', '16-32', '32-64', '>=64']
DENS_BINS = [0, 50, 100, 1000]
DENS_LBL = ['n<=50', '50<n<=100', 'n>100']


def match_objects(val, dets, iou_thr=0.5, class_aware=True):
    """Devuelve un DataFrame con una fila por objeto anotado: tamaño, densidad de su imagen y si fue recuperado."""
    gts, pr = {}, {}
    for a in val['annotations']:
        if not a.get('iscrowd', 0):
            gts.setdefault(a['image_id'], []).append(a)
    for x in dets:
        pr.setdefault(x['image_id'], []).append(x)
    rows = []
    for im in val['images']:
        g = gts.get(im['id'], [])
        if not g:
            continue
        p = sorted(pr.get(im['id'], []), key=lambda x: -x['score'])
        hit = np.zeros(len(g), bool)
        if p:
            M = _iou_matrix([x['bbox'] for x in p], [x['bbox'] for x in g])
            if class_aware:
                M = M * (np.array([x['category_id'] for x in p])[:, None]
                         == np.array([x['category_id'] for x in g])[None, :])
            for i in range(len(p)):
                cand = np.where(~hit & (M[i] >= iou_thr))[0]
                if cand.size:
                    hit[cand[M[i, cand].argmax()]] = True
        for a, h in zip(g, hit):
            rows.append({'image_id': im['id'], 'n_img': len(g), 'cat': a['category_id'],
                         'lado': float(np.sqrt(a['bbox'][2] * a['bbox'][3])), 'hit': bool(h)})
    D = pd.DataFrame(rows)
    D['tam'] = pd.cut(D.lado, SIZE_BINS, labels=SIZE_LBL, right=False)
    D['dens'] = pd.cut(D.n_img, DENS_BINS, labels=DENS_LBL)
    return D


def tables(D):
    rec = D.pivot_table(index='tam', columns='dens', values='hit', aggfunc='mean', observed=False)
    cnt = D.pivot_table(index='tam', columns='dens', values='hit', aggfunc='size', observed=False)
    mix = (cnt / cnt.sum()).round(3)                   # composición por tamaño de cada grupo de densidad
    # recall esperado de cada grupo de densidad si solo importara el tamaño (estandarización directa):
    # recall_por_tamaño(global) ponderado por la mezcla de tamaños del grupo
    r_size = D.groupby('tam', observed=False).hit.mean()
    expected = (mix.mul(r_size, axis=0)).sum()
    observed = D.groupby('dens', observed=False).hit.mean()
    summ = pd.DataFrame({'recall_observado': observed, 'recall_esperado_solo_tamano': expected,
                         'lado_mediano_px': D.groupby('dens', observed=False).lado.median(),
                         'objetos': D.groupby('dens', observed=False).size()})
    return rec, cnt, mix, summ


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--val', required=True)
    ap.add_argument('--dets', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--out', default='../results/fase_c')
    ap.add_argument('--cats', default='')               # p. ej. 1,3,4,6,9,10 para el zero-shot
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    val = json.load(open(a.val))
    dets = json.load(open(a.dets))
    if a.cats:
        keep = {int(c) for c in a.cats.split(',')}
        val = {**val, 'annotations': [x for x in val['annotations'] if x['category_id'] in keep]}
        dets = [x for x in dets if x['category_id'] in keep]
    out = {}
    for mode, ca in (('con_clase', True), ('sin_clase', False)):
        D = match_objects(val, dets, class_aware=ca)
        rec, cnt, mix, summ = tables(D)
        rec.to_csv(f'{a.out}/{a.tag}_recall_tam_x_dens_{mode}.csv')
        summ.to_csv(f'{a.out}/{a.tag}_resumen_{mode}.csv')
        out[mode] = {'recall_global': float(D.hit.mean()), 'recall_tam_x_dens': rec.round(4).to_dict(),
                     'resumen': summ.round(4).to_dict()}
        print(f'== {a.tag} · {mode} · recall global {D.hit.mean():.3f}')
        print('recall por tamaño (filas) y densidad (columnas):\n', rec.round(3))
        print(summ.round(3))
    cnt.to_csv(f'{a.out}/{a.tag}_conteos_tam_x_dens.csv')
    print('objetos por celda:\n', cnt)
    json.dump(out, open(f'{a.out}/{a.tag}.json', 'w'), indent=1)
