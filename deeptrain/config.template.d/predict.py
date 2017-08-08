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

from . import train

_data_dir_path = "/home/erik/data/"
#filepath of mode
model_filepath = "/home/erik/data/train/train_25/model_epoch_3.npz"

#filepath or filepaths of input. may be None, in this case reads from argv
input_filepaths = glob.glob("/home/erik/data/sao_martinho/farms/*/tensor.npz")

#maximum number of prediction images to save
max_n_preds_save = 300
#directory to save predictions
preds_save_dir = "/home/erik/data/preds"
#filepath to save predictions probabilities. if None, does not save anything
preds_csv_fp = "/home/erik/data/preds.csv"
#maximum number of prediction points to have in pred csv. can be None
max_pred_points = 10**7

#dictionary of arguments for dataproc.pre_proc method
pre_proc_kwargs = train.batches_gen_kwargs["pre_proc_kwargs"]

#dictionary of arguments for dataproc.load method
load_kwargs = train.batches_gen_kwargs["load_kwargs"]

#stride for stride prediction
pred_stride = 64
