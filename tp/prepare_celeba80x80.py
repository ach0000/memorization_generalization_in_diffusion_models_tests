'''
Rebuilds an approximation of the authors' datasets/img_align_celeba/train80x80_no_repeats.pt
(written here to tp/donnees/img_align_celeba/train80x80_no_repeats.pt)
(not distributed by the authors) from a HuggingFace mirror of CelebA (img_align_celeba).

Same preprocessing as dataloader_func.load_CelebA_dataset + prep_celeba:
crop 218x178 -> 160x160 (im[29:-29, 9:-9]), downsample to 80x80 (or another size),
average RGB channels, float range [0, 1], shape (N, 1, 80, 80).

Differences with the authors' file:
 - image order is not the authors' one (they used os.listdir order), so
   the training sets data[0:N] / data[-N:] do NOT match the images the
   released denoisers were trained on;
 - near-duplicate removal (quality_metrics_func.remove_repeats) is skipped,
   its N x N correlation matrix does not fit in a free Colab session.
'''
import io
import os
import sys
import urllib.request

import numpy as np
import torch
from PIL import Image

HF_URL = 'https://huggingface.co/datasets/nielsr/CelebA-faces/resolve/main/data/train-0000{}-of-00003.parquet'
N_SHARDS = 3


def preprocess(img, taille=80):
    im = img.convert('RGB')
    if im.size == (178, 218):
        im = im.crop((9, 29, 169, 189))  # (left, top, right, bottom) = im[29:-29, 9:-9]
    else:  # fallback: center crop to a square
        w, h = im.size
        s = min(w, h)
        im = im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    im = im.resize((taille, taille), Image.BOX)
    return np.asarray(im, dtype=np.float32).mean(axis=2) / 255.


def build_celeba(out_path, n_images=2 * 99325, taille=80, cache_dir=None):
    import pyarrow.parquet as pq

    cache_dir = cache_dir or os.environ.get('CELEBA_CACHE', '/tmp/celeba_parquet')

    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    data = torch.empty(n_images, 1, taille, taille, dtype=torch.float32)
    i = 0
    for s in range(N_SHARDS):
        shard = os.path.join(cache_dir, 'shard{}.parquet'.format(s))
        if not os.path.exists(shard):
            print('downloading shard', s + 1, '/', N_SHARDS)
            urllib.request.urlretrieve(HF_URL.format(s), shard)
        pf = pq.ParquetFile(shard)
        for batch in pf.iter_batches(batch_size=2000, columns=['image']):
            for rec in batch.column(0).to_pylist():
                if i >= n_images:
                    break
                data[i, 0] = torch.from_numpy(preprocess(Image.open(io.BytesIO(rec['bytes'])), taille))
                i += 1
            print('\r', i, 'images', end='')
            if i >= n_images:
                break
        if i >= n_images:
            break
    print()
    if i < n_images:
        data = data[:i].clone()
    torch.save(data, out_path)
    print('saved', tuple(data.shape), 'to', out_path)


def build_celeba80x80(out_path, n_images=2 * 99325, cache_dir=None):
    build_celeba(out_path, n_images, 80, cache_dir)


if __name__ == '__main__':
    defaut = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'donnees', 'img_align_celeba',
                          'train80x80_no_repeats.pt')
    build_celeba80x80(sys.argv[1] if len(sys.argv) > 1 else defaut)
