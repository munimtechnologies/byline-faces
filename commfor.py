#!/usr/bin/env python3
# Copyright 2026 Munim, Inc. (MV2). Licensed under the Apache License, Version 2.0 (Apache-2.0).
# Made with AI agents working for MV2 of Munim, Inc.
"""A general AI-image detector for byline-faces: Community Forensics (Park and Owens, CVPR 2025).

The eye-placement test in byline_faces.py only sees StyleGAN-family faces. Community
Forensics is a ViT-Small classifier trained end to end on images from 4,803 generators,
diffusion models included (arXiv:2411.04125). Its authors publish the weights on Hugging
Face (OwensLab/commfor-model-384, MIT licence, 21.8M parameters), so it runs on a CPU.

Evaluation recipe, as in the authors' dataloader.get_transform(mode="test"):
resize the shorter side to 440 (bilinear), centre-crop 384 x 384, scale to [0, 1],
normalise with the ImageNet mean and standard deviation; one logit, sigmoid = P(generated).

The recipe arithmetic is standard-library only; CommFor needs torch, timm, safetensors,
numpy and Pillow, imported lazily. reproduce() checks the loaded model against the five
scores printed in the authors' eval_using_huggingface.ipynb.
"""
import io
import math
import os

WEIGHTS_URL = ("https://huggingface.co/OwensLab/commfor-model-384/resolve/main/"
               "model.safetensors")
WEIGHTS_SIZE = 87262324
TIMM_ARCH = "vit_small_patch16_384"
RESIZE, CROP = 440, 384
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# Five DALL-E 2 images shipped with the authors' code, and the scores their notebook prints
# for the 384 model (in this order).
TEST_IMAGE_BASE = ("https://raw.githubusercontent.com/JeongsooP/Community-Forensics/main/"
                   "test_images/")
TEST_IMAGES = [("00000274.png", 0.9988), ("00000420.png", 0.9878), ("00000845.png", 0.9569),
               ("00000916.png", 0.9516), ("00000989.png", 0.7860)]
REPRO_TOLERANCE = 0.02


def resize_crop_box(width, height, resize=RESIZE, crop=CROP):
    """torchvision Resize(int) then CenterCrop(int), as arithmetic.

    Returns ((new_w, new_h), (left, top, right, bottom)) for PIL's resize and crop."""
    if width <= 0 or height <= 0:
        raise ValueError("image size must be positive")
    if width <= height:
        new_w, new_h = resize, int(resize * height / width)
    else:
        new_w, new_h = int(resize * width / height), resize
    top = int(round((new_h - crop) / 2.0))
    left = int(round((new_w - crop) / 2.0))
    return (new_w, new_h), (left, top, left + crop, top + crop)


def compare_repro(got, reference=TEST_IMAGES, tol=REPRO_TOLERANCE):
    """Compare {name: score or None} with the reference scores."""
    diffs = {}
    for name, ref in reference:
        v = got.get(name)
        diffs[name] = None if v is None else abs(v - ref)
    known = [d for d in diffs.values() if d is not None]
    return {"reference": dict(reference), "got": dict(got), "abs_diff": diffs,
            "max_abs_diff": max(known) if known else None,
            "ok": bool(known) and len(known) == len(reference) and max(known) <= tol,
            "tolerance": tol}


def strip_prefix(state_dict, prefix="vit."):
    """The authors' ViTClassifier keeps the timm model as .vit; timm wants bare keys."""
    return {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in state_dict.items()}


class CommFor:
    """Community Forensics ViT-S/16 at 384 px. prob(bytes) -> P(generated) in [0, 1]."""

    def __init__(self, weights_path, threads=2):
        import threading
        import numpy
        import timm
        import torch
        from safetensors.torch import load_file
        torch.set_num_threads(max(1, int(threads)))
        self.torch, self.np = torch, numpy
        self.lock = threading.Lock()
        model = timm.create_model(TIMM_ARCH, pretrained=False, num_classes=1)
        state = strip_prefix(load_file(weights_path))
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError("weights do not match %s: missing %s, unexpected %s"
                               % (TIMM_ARCH, list(missing)[:5], list(unexpected)[:5]))
        model.eval()
        self.model = model
        self.mean = numpy.array(MEAN, dtype=numpy.float32).reshape(1, 1, 3)
        self.std = numpy.array(STD, dtype=numpy.float32).reshape(1, 1, 3)
        self.name = "commfor-model-384"

    def tensor(self, data):
        from PIL import Image
        im = Image.open(io.BytesIO(data))
        im.load()
        im = im.convert("RGB")
        (nw, nh), box = resize_crop_box(im.size[0], im.size[1])
        im = im.resize((nw, nh), Image.BILINEAR).crop(box)
        a = self.np.asarray(im, dtype=self.np.float32) / 255.0
        a = (a - self.mean) / self.std
        return self.torch.from_numpy(a.transpose(2, 0, 1).copy()).unsqueeze(0)

    def prob(self, data):
        x = self.tensor(data)
        with self.lock, self.torch.inference_mode():
            logit = self.model(x)
        p = float(self.torch.sigmoid(logit.float())[0, 0])
        return p if math.isfinite(p) else None


def ensure_weights(path, fetch_file):
    """Download the weights once (fetch_file(url, path, expect_size) -> bool)."""
    if os.path.exists(path) and os.path.getsize(path) == WEIGHTS_SIZE:
        return True
    return fetch_file(WEIGHTS_URL, path, WEIGHTS_SIZE)


def reproduce(model, fetch):
    """Score the authors' five test images; fetch(url) -> (status, final, ctype, body)."""
    got = {}
    for name, _ref in TEST_IMAGES:
        try:
            st, _f, _ct, body = fetch(TEST_IMAGE_BASE + name)
            got[name] = model.prob(body) if st == 200 and body else None
        except Exception:
            got[name] = None
    return compare_repro(got)
