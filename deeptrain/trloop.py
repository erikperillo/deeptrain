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

import multiprocessing as mp
import random
import time
import math
import queue

import dataproc

def load(fp):
    return dataproc.load(fp, **{"x_dtype": "float32", "y_dtype": "float32"})

def augment(x, y):
    kw = {
        "augment": True,
        "rot90": True,
        "rot180": True,
        "rot270": True,
        "hmirr_rot90": True,
        "hmirr_rot180": True,
        "hmirr_rot270": True,
        "alter_brightness_prob": 0.4,
        "alter_brightness_rng": 0.5,
        "shuffle": False,
        "hmirr": True,
        "vmirr": True,
    }
    return dataproc.augment(x, y, **kw)

def pre_proc(x, y):
    return dataproc.pre_proc(x), y

def pop_rand_elem(lst):
    if not lst:
        return None
    index = random.randint(0, len(lst)-1)
    lst[-1], lst[index] = lst[index], lst[-1]
    return lst.pop()

def fetch(thr_id, fps, q, load_chunk_size, max_n_samples):
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
                x, y = load(fp)
                #pre-processing
                x, y = pre_proc(x, y)
                #augmentation
                augm_xy = augment(x, y)
                #putting to samples list
                samples.extend(augm_xy)

        #putting random sample into queue
        if samples:
            q.put(pop_rand_elem(samples))

    #putting remaining samples to queue
    while samples:
        q.put(pop_rand_elem(samples))

def batch_gen(
        filepaths,
        batch_size=1,
        n_threads=1,
        max_n_samples=None,
        fetch_thr_load_chunk_size=1,
        max_augm_factor=1):

    """
    Main thread for generation of batches.
    Spans other threads for data loading and processing, gathering the
        samples into batches and yielding them.
    """
    if max_n_samples is None:
        max_n_samples = len(filepaths)*max_augm_factor

    #shuffling filepaths
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
            args=(i, thr_fps, thr_q, fetch_thr_load_chunk_size,
                max_n_samples_per_thr))

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
                yield batch
                batch = []

    #joining processes
    for thr in threads:
        thr.join()

def test():
    import glob
    fps = glob.glob("/home/erik/random/gaiia/decision_tree/data/tensors/*.npz")
    fps.extend(fps)
    fps.extend(fps)
    gen = batch_gen(
        filepaths=fps,
        batch_size=4,
        n_threads=4,
        fetch_thr_load_chunk_size=4*4,
        max_augm_factor=8,
        max_n_samples=1000)

    for i, batch in enumerate(gen):
        print(i, len(batch), flush=True)
        #print(batch)

if __name__ == "__main__":
    test()
