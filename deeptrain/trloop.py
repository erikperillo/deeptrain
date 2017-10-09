#!/usr/bin/env python3

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

"""
Module for training models.
Contains routines for parallel data fetching/processing and training.
"""

import multiprocessing as mp
import numpy as np
import random
import time
import math
import queue
from collections import defaultdict

def _no_op(*args, **kwargs):
    pass

def _dummy_load(fp):
    return random.randint(0, 10), random.randint(0, 10)

def _dummy_augment(xy):
    return 8*[xy]

def _dummy_augment2(xy):
    return [xy]

def _dummy_pre_proc(xy):
    return xy

def _inf_gen():
    n = 0
    while True:
        yield n
        n += 1

def _pop_rand_elem(lst):
    if not lst:
        return None
    index = random.randint(0, len(lst)-1)
    lst[-1], lst[index] = lst[index], lst[-1]
    return lst.pop()

def _str_fmt_time(seconds):
    int_seconds = int(seconds)
    hours = int_seconds//3600
    minutes = (int_seconds%3600)//60
    seconds = int_seconds%60 + (seconds - int_seconds)
    return "{:02d}h:{:02d}m:{:.2f}s".format(hours, minutes, seconds)

def _str(obj):
    return str(obj) if obj is not None else "?"

def _str_fmt_dct(dct):
    """Assumes dict mapping str to float."""
    return " | ".join("{}: {:.4g}".format(k, v) for k, v in dct.items())

def fetch(fps, q, load_chunk_size, max_n_samples,
    load_fn, augment_fn, pre_proc_fn):
    """
    Thread to load, pre-process and augment data, putting it into queue q.
    """
    #list to store samples
    samples = []

    end = False
    while not end:
        if len(samples) < max_n_samples:
            #loading load_chunk_size files before putting into queue.
            #this is to better spread augmented samples
            for __ in range(load_chunk_size):
                if not fps:
                    end = True
                    break

                #getting filepath
                fp = fps.pop()
                #loading x and y
                xy = load_fn(fp)
                #augmentation
                augm_xy = augment_fn(xy)
                #pre-processing
                for i, xy in enumerate(augm_xy):
                    augm_xy[i] = pre_proc_fn(xy)
                #putting to samples list
                samples.extend(augm_xy)

        #putting random sample into queue
        if samples:
            q.put(_pop_rand_elem(samples))

    #putting remaining samples to queue
    while samples:
        q.put(_pop_rand_elem(samples))

def batch_gen(
        filepaths,
        batch_size=1,
        n_threads=1,
        max_n_samples=None,
        fetch_thr_load_chunk_size=1,
        fetch_thr_load_fn=_dummy_load,
        fetch_thr_augment_fn=_dummy_augment,
        fetch_thr_pre_proc_fn=_dummy_pre_proc,
        max_augm_factor=1,
        shuffle_fps=True):

    """
    Main thread for generation of batches.
    Spans other threads for data loading and processing, gathering the
        samples into batches and yielding them.
    """
    if max_n_samples is None:
        max_n_samples = len(filepaths)*max_augm_factor

    #shuffling filepaths
    if shuffle_fps:
        random.shuffle(filepaths)

    #threads and thread queues
    threads = []
    qs = []
    #maximum number of samples per thread
    max_n_samples_per_thr = math.ceil(
        max_n_samples/(n_threads*fetch_thr_load_chunk_size*max_augm_factor))
    #number of filepaths per thread
    n_fps_per_thr = math.ceil(len(filepaths)/n_threads)

    #initializing thread objects
    for i in range(n_threads):
        #getting a slice of filepaths for thread
        thr_fps = filepaths[i*n_fps_per_thr:(i+1)*n_fps_per_thr]
        #queue in which fetch thread will put its samples
        thr_q = mp.Queue(maxsize=1)
        #process object
        thr = mp.Process(
            target=fetch,
            args=(thr_fps, thr_q,
                fetch_thr_load_chunk_size, max_n_samples_per_thr,
                fetch_thr_load_fn, fetch_thr_augment_fn, fetch_thr_pre_proc_fn))

        threads.append(thr)
        qs.append(thr_q)

    #starting threads
    for thr in threads:
        thr.start()

    #true iff all threads are finished
    all_done = False
    #indexes for threads
    thr_ids = list(range(n_threads))
    #batch to be yielded
    batch = []

    while not all_done:
        all_done = True
        #shuffling indexes to fetch from threads in random order
        random.shuffle(thr_ids)

        for i in thr_ids:
            if threads[i].is_alive():
                all_done = False

            #trying to get sample from thread
            try:
                xy = qs[i].get(block=False)
            except queue.Empty:
                continue

            #if reached batch size, yields batch
            batch.append(xy)
            if len(batch) == batch_size:
                batch_x = np.stack([b[0] for b in batch], axis=0)
                batch_y = np.stack([b[1] for b in batch], axis=0)
                yield batch_x, batch_y
                batch = []

    #joining processes
    for thr in threads:
        thr.join()

def train_loop(
    train_set, train_fn,
    n_epochs=10,
    val_set=None, val_fn=None, val_every_its=None,
    save_model_fn=_no_op, save_every_its=None,
    verbose=2, print_fn=print,
    batch_gen_kw={}):
    """
    General Training loop.
    """

    #info/warning functions
    info = print_fn if verbose >= 2 else _no_op
    warn = print_fn if verbose >= 1 else _no_op
    _print = print if verbose >= 2 else _no_op

    #checking if using validation
    validation = val_set is not None and val_fn is not None
    #setting up validation batches_gen_kwargs
    val_batch_gen_kw = dict(batch_gen_kw)
    val_batch_gen_kw["max_augm_factor"] = 1
    val_batch_gen_kw["fetch_thr_augment_fn"] = _dummy_augment2

    #initial start time
    start_time = time.time()

    #main loop
    _print("starting training loop...")
    for epoch in _inf_gen():
        #checking stopping
        if n_epochs is not None and epoch >= n_epochs:
            warn("\nWARNING: maximum number of epochs reached")
            end_reason = "n_epochs"
            return end_reason

        info("on epoch {}/{} (time so far: {})".format(
            epoch+1, _str(n_epochs), _str_fmt_time(time.time() - start_time)))

        #dictionary in format metric: value
        loss_sum = 0
        #estimating number of batches for train phase
        n_batches = len(train_set)//batch_gen_kw["batch_size"]
        n_batches *= batch_gen_kw["max_augm_factor"]
        #main train loop
        for i, (bx, by) in enumerate(batch_gen(train_set, **batch_gen_kw)):
            loss = train_fn(bx, by)
            loss_sum += loss
            _print("    [batch {}/~{}] loss: {:.4g}".format(
                i+1, n_batches, loss), 16*" ", end="\r")

            #validation every #step
            if val_every_its is not None and (i+1)%val_every_its == 0:
                metrics = val_fn(bx, by)
                _print("    [batch {}] metrics: {}".format(
                    i+1, _str_fmt_dct(metrics)))

            #saving model
            if save_every_its is not None and (i+1)%save_every_its == 0:
                save_model_fn()

        #printing train loop metrics
        loss_mean = loss_sum/max(i+1, 1)
        info("    train set:", 32*" ")
        info("        processed {} batches".format(i+1))
        info("        mean loss: {}".format(loss_mean))

        #saving model after epoch
        save_model_fn()

        if not validation:
            continue

        #dictionary in format metric: value
        metrics_sum = defaultdict(float)
        #estimating number of batches for validation phase
        n_batches = len(val_set)//val_batch_gen_kw["batch_size"]
        #main validation loop
        for i, (bx, by) in enumerate(batch_gen(val_set, **val_batch_gen_kw)):
            metrics = val_fn(bx, by)

            for k, v in metrics.items():
                metrics_sum[k] += v

            _print("    [batch {}/~{}] {}".format(
                i+1, n_batches, _str_fmt_dct(metrics)), 16*" ", end="\r")

        metrics_mean = {k: v/max(i+1, 1) for k, v in metrics_sum.items()}
        info("    val set:", 40*" ")
        info("        processed {} batches".format(i+1))
        info("        mean metrics:", _str_fmt_dct(metrics_mean))

def test():
    import glob
    fps = glob.glob("/home/erik/random/gaiia/decision_tree/data/tensors/*.npz")
    fps.append(fps)
    fps.append(fps)
    fps.append(fps)
    fps.append(fps)

    batch_gen_kw = {
        "batch_size": 2,
        "n_threads": 3,
        "max_n_samples": 1000,
        "fetch_thr_load_chunk_size": 10,
        "fetch_thr_load_fn": _dummy_load,
        "fetch_thr_augment_fn": _dummy_augment,
        "fetch_thr_pre_proc_fn": _dummy_pre_proc,
        "max_augm_factor": 8
    }

    train_loop(
        train_set=fps[:-200],
        train_fn=lambda x, y: random.uniform(0, 1),
        n_epochs=10,
        val_set=fps[-200:],
        val_fn=lambda x, y: {"dummy1": random.uniform(0, 1), "dummy2": 0},
        val_every_its=64,
        batch_gen_kw=batch_gen_kw)

    #train_set, train_fn,
    #n_epochs=10,
    #val_set=None, val_fn=None, val_every_its=None,
    #save_model_fn=_no_op, save_every_its=None,
    #verbose=2, print_fn=print,
    #**batch_gen_kw):

if __name__ == "__main__":
    test()
