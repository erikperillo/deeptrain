"""
The MIT License (MIT)

Copyright (c) 2017 Erik Perillo <erik.perillo@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.
"""

import numpy as np
import json
import random

def load(filepath, **kwargs):
    """
    Interface to load X, y data.
    Returns X and y
    """
    data = np.load(filepath)
    x, y = data["x"], data["y"]
    if "x_dtype" in kwargs:
        x = x.astype(kwargs["x_dtype"])
    if "y_dtype" in kwargs:
        y = y.astype(kwargs["y_dtype"])
    return x, y

def pre_proc(batch, **kwargs):
    """
    Pre-processing of batch.
    Assumes format (n_samples, n_channels, height, width).
    """
    #opening stats file
    #with open(kwargs["stats_fp"]) as f:
    #    stats = json.load(f)

    #converting keys from str to int
    #stats = {int(k): v for k, v in stats.items()}

    #normalizing per channel
    #for k, s in stats.items():
    for k in range(4):
        #batch[:, k, :, :] = (batch[:, k, :, :] - s["mean"])/s["std"]
        u = random.uniform(-10, 10)
        sd = random.uniform(0.1, 10)
        #batch[:, k, :, :] = (batch[:, k, :, :] - u)/sd
        batch[k, :, :] = (batch[k, :, :] - u)/sd

    return batch

def rot(arr, reps=1):
    """
    Rotating of batch, in reps*90 degrees.
    Assumes format (n_samples, n_channels, height, width).
    """
    for __ in range(reps%4):
        arr = arr.swapaxes(-2, -1)[..., ::-1]
    return arr

def alter_brightness(img, prob, rng=0.5):
    if np.random.uniform(0, 1) <= prob:
        img = img*(1 + np.random.uniform(-1, 1)*rng)
        img = img.clip(0, 255)
    return img

def _augment(batch, **kwargs):
    """
    Data augmentation of batch.
    Assumes format (n_samples, n_channels, height, width).
    """
    if not kwargs["augment"]:
        return batch

    augm = batch

    #rotated
    for i in range(1, 4):
        if kwargs["rot{}".format(i*90)]:
            augm = np.append(augm, rot(batch, i), axis=0)

    #horizontal mirrror
    hmirr = batch[..., ::-1]
    if kwargs["hmirr"]:
        augm = np.append(augm, hmirr, axis=0)

    #horizontal mirrror rotated
    for i in range(1, 4):
        if kwargs["hmirr_rot{}".format(i*90)]:
            augm = np.append(augm, rot(hmirr, i), axis=0)

    #random brightness modifications
    if "alter_brightness_prob" in kwargs:
        for i in range(augm.shape[0]):
            augm[i] = alter_brightness(augm[i], kwargs["alter_brightness_prob"],
                kwargs["alter_brightness_rng"])

    return augm

def augment(x, y, **kwargs):
    x = _augment(x, **kwargs)
    del kwargs["alter_brightness_prob"]
    y = _augment(y, **kwargs)

    if kwargs["shuffle"]:
        indexes = np.array(range(len(x)))
        np.random.shuffle(indexes)
        x = x[indexes]
        y = y[indexes]

    return x, y
