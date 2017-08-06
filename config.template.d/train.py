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

_data_dir_path = "/home/erik/proj/att/att/deep/data"

#if not None, uses weights of a pre-trained model from path
pre_trained_model_fp = "/home/erik/proj/att/att/deep/data/train_696/model.npz"

#directory where dir with train info/model will be stored
output_dir_basedir = _data_dir_path

#filepaths of train batches
dataset_train_filepaths = glob.glob("/home/erik/imgs_train/*[^0123456789].npz")
#filepaths of validation batches, can be None
dataset_val_filepaths = glob.glob("/home/erik/imgs_val/*.npz")

#number of epochs to use in train
n_epochs = 8
#batch size
batch_size = 10
#0 for nothing, 1 for only warnings, 2 for everything
verbose = 2
#validation function value tolerance
val_f_val_tol = None
#data types
x_dtype = np.float32
y_dtype = np.float32

#chunk size to load data
load_chunk_size = 4000

#dictionary of arguments for dataproc.pre_proc method
pre_proc_args_dict = {
    "stats_fp": "/home/erik/imgs_train_stats.json"
}

#dictionary of arguments for dataproc.load method
load_args_dict = {
    "x_dtype": x_dtype,
    "y_dtype": y_dtype
}
