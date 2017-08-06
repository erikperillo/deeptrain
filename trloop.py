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
import time
from collections import defaultdict
import threading
import queue

import dataproc

def _str_fmt_time(seconds):
    int_seconds = int(seconds)
    hours = int_seconds//3600
    minutes = (int_seconds%3600)//60
    seconds = int_seconds%60 + (seconds - int_seconds)
    return "%.2dh:%.2dm:%.3fs" % (hours, minutes, seconds)

def _silence(*args, **kwargs):
    pass

def _inf_gen():
    n = 0
    while True:
        yield n
        n += 1

def _str_fmt_dct(dct):
    """Assumes dict mapping str to float."""
    return " | ".join("%s: %.4g" % (k, v) for k, v in dct.items())

def _str(obj):
    return str(obj) if obj is not None else "?"

def batches_gen(X, y, batch_size, shuffle=False):
    n_samples = len(y)
    indices = np.arange(n_samples)
    if shuffle:
        np.random.shuffle(indices)
    for start_idx in range(0, max(n_samples-batch_size+1, 1), batch_size):
        excerpt = indices[start_idx:start_idx+batch_size]
        yield X[excerpt], y[excerpt]

def load_data(filepaths, q, stop, chunk_size, load_args_dict={}):
    i = 0
    while i <= len(filepaths) - chunk_size:
        if stop.is_set():
            break

        if q.empty():
            xs = []
            ys = []

            for j in range(i, i+chunk_size):
                x, y = dataproc.load(filepaths[j], **load_args_dict)
                xs.append(x)
                ys.append(y)

            xs_mtx = np.stack(xs, axis=0)
            ys_mtx = np.stack(ys, axis=0)

            q.put((xs_mtx, ys_mtx))
            i += chunk_size

        time.sleep(0.1)

    stop.set()

def batches_gen_async(filepaths, batch_size, shuffle=False, print_f=_silence,
        load_chunk_size=1000, load_args_dict={},
        pre_proc_args_dict={}):
    if shuffle:
        np.random.shuffle(filepaths)

    q = queue.Queue(maxsize=1)
    stop = threading.Event()
    data_loader = threading.Thread(target=load_data,
        args=(filepaths, q, stop, load_chunk_size, load_args_dict))
    data_loader.start()

    try:
        for i in _inf_gen():
            X, y = q.get()

            #normalization of data
            X = dataproc.pre_proc(X, **pre_proc_args_dict)

            for batch_X, batch_y in batches_gen(X, y, batch_size, shuffle):
                yield i, (batch_X, batch_y)

            if q.empty() and stop.is_set():
                break
    except:
        raise
    finally:
        stop.set()
        data_loader.join()

def run_epoch(
    data, func,
    batch_size=1,
    info=_silence, warn=_silence,
    shuf_data=True,
    load_chunk_size=1000, load_args_dict={},
    pre_proc_args_dict={}):

    vals_sum = defaultdict(float)
    n_its = 0

    for i, (bi, xy) in enumerate(batches_gen_async(data, batch_size,
        shuf_data, info, load_chunk_size, load_args_dict,
        pre_proc_args_dict)):
        vals = func(*xy)

        for k, v in vals.items():
            vals_sum[k] += v

        n_its += 1
        info("    [batch %d, data part %d/%d]" %\
            (i, bi+1, len(data)//load_chunk_size),
            _str_fmt_dct(vals), 32*" ", end="\r")

    vals_mean = {k: v/n_its for k, v in vals_sum.items()}
    return vals_mean

def train_loop(
    tr_set, tr_f,
    n_epochs=10, batch_size=1,
    val_set=None, val_f=None, val_f_val_tol=None,
    load_chunk_size=1000, load_args_dict={},
    pre_proc_args_dict={},
    save_model_after_epoch_f=None,
    verbose=2, print_f=print):
    """
    General Training loop.
    Parameters:
    tr_set: [str]
        list of str Training set containing X and y.
    tr_f : callable
        Training function giving a dict in format {"name": val, ...}
    n_epochs : int or None
        Number of epochs. If None, is infinite.
    batch_size : int
        Batch size.
    val_set: None or [str]
        Validation set containing X and y.
    val_f : callable or None
        Validation function giving a dict in format {"name": val, ...}
    verbose : int
        Prints nothing if 0, only warnings if 1 and everything if >= 2.
    """

    #info/warning functions
    info = print_f if verbose >= 2 else _silence
    warn = print_f if verbose >= 1 else _silence
    epoch_info = print if verbose >= 2 else _silence

    #checking if use validation
    validation = val_set is not None and val_f is not None

    #initial values for some variables
    start_time = time.time()

    #main loop
    info("[info] starting training loop...")
    for epoch in _inf_gen():
        if n_epochs is not None and epoch >= n_epochs:
            warn("\nWARNING: maximum number of epochs reached")
            end_reason = "n_epochs"
            return end_reason

        info("epoch %d/%s:" % (epoch+1, _str(n_epochs)))

        tr_vals = run_epoch(tr_set, tr_f, batch_size,
            info=epoch_info, warn=warn, shuf_data=True,
            load_chunk_size=load_chunk_size, load_args_dict=load_args_dict,
            pre_proc_args_dict=pre_proc_args_dict)
        info("    train values:", _str_fmt_dct(tr_vals), 32*" ")

        if validation:
            val_vals = run_epoch(val_set, val_f, batch_size,
                info=epoch_info, warn=warn, shuf_data=True,
                load_chunk_size=load_chunk_size,
                load_args_dict=load_args_dict,
                pre_proc_args_dict=pre_proc_args_dict)
            info("    val values:", _str_fmt_dct(val_vals), 32*" ")

        info("    time so far:", _str_fmt_time(time.time() - start_time))

        if save_model_after_epoch_f is not None:
            info("    saving model after epoch")
            save_model_after_epoch_f(epoch)
