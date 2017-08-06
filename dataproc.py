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
    with open(kwargs["stats_fp"]) as f:
        stats = json.load(f)

    #converting keys from str to int
    stats = {int(k): v for k, v in stats.items()}

    #normalizing per channel
    for k, s in stats.items():
        batch[:, k, :, :] = (batch[:, k, :, :] - s["mean"])/s["std"]

    return batch
