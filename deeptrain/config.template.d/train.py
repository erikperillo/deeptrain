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

import os
import glob
import numpy as np

#if not None, uses weights of a pre-trained model from path
pre_trained_model_fp = "/home/erik/data/train/train_26/model_epoch_1.npz"
#pre_trained_model_fp = None

#directory where dir with train info/model will be stored
output_dir_basedir = "/home/erik/data/train"

#filepaths of train batches
dataset_train_filepaths = glob.glob(
        #"/home/erik/data/cuts/modis/train/*.npz")
        "/home/erik/data/cuts/cuts_1/train/*.npz")
#filepaths of validation batches, can be None
dataset_val_filepaths = glob.glob(
    #"/home/erik/data/cuts/modis/val/*.npz")
    "/home/erik/data/cuts/cuts_1/val/*.npz")

#number of epochs to use in train
n_epochs = 20
#0 for nothing, 1 for only warnings, 2 for everything
verbose = 2

#batch_size
batch_size = 8

batches_gen_kwargs = {
    #number of samples per batch
    "batch_size": batch_size,

    #number of threads to load and pre-proc data in parallel
    "n_load_threads": 3,

    #number of samples to be loaded/processed by each thread
    "load_thread_n_samples": 4*batch_size,

    #maximum number of samples to be loaded at a given time
    "max_n_samples_loaded": 512*(2*batch_size),

    #dictionary of arguments for dataproc.pre_proc method
    "pre_proc_kwargs": {
        #"stats_fp": "/home/erik/data/cuts/modis/train/stats.json",
        "stats_fp": "/home/erik/data/cuts/cuts_0/train/stats.json",
    },

    #dictionary of arguments for dataproc.augment method.
    "augment_kwargs": {
        #augment options
        #"operations": set()
        "rot90": True,
        "rot180": True,
        "rot270": True,
        "hmirr": True,
        "hmirr_rot90": True,
        "hmirr_rot180": True,
        "hmirr_rot270": True,
        #probability of altering one image's brightness
        "alter_brightness_prob": 0.2,
        #range of brightness alteration
        "alter_brightness_rng": 0.5,
        #shuffle augmented data
        "shuffle": True,
    },

    #dictionary of arguments for dataproc.load method
    "load_kwargs": {
        "x_dtype": "float32",
        "y_dtype": "float32",
    },
}
