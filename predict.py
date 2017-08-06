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

import pandas as pd
import sys
import random
import time
import theano
from theano import tensor as T
import itertools
import numpy as np
from skimage import io
import os

import dataproc
import config.model as model
import config.predict as cfg

def predict(x, pred_f):
    x = x.reshape((1, ) + x.shape)
    x = dataproc.pre_proc(x, **cfg.pre_proc_args_dict)

    start_time = time.time()
    y = pred_f(x)
    elapsed_time = time.time() - start_time

    y = y.reshape(y.shape[-2:])

    return y, elapsed_time

def stride_predict(img, shape, stride, pred_f):
    if isinstance(shape, int):
        shape = (shape, shape)
    h_shape, w_shape = shape
    if isinstance(stride, int):
        stride = (stride, stride)
    h_stride, w_stride = stride
    height, width = img.shape[-2:]

    pred = np.zeros(shape=img.shape[-2:], dtype="float32")
    mult = np.zeros(shape=img.shape[-2:], dtype="int")
    pred_counter = 0
    tot_pred_time = 0

    for i, j in itertools.product(range(0, height-h_shape+1, h_stride),
        range(0, width-w_shape+1, w_stride)):
        x = img[..., i:i+h_shape, j:j+w_shape].copy()
        y, pred_time = predict(x, pred_f)

        pred[i:i+h_shape, j:j+w_shape] += y
        mult[i:i+h_shape, j:j+w_shape] += 1

        tot_pred_time += pred_time
        pred_counter += 1

    mult[np.where(mult == 0)] = 1
    pred = pred/mult.astype("float32")
    avg_pred_time = tot_pred_time/max(1, pred_counter)

    return pred, tot_pred_time, avg_pred_time

def main():
    if cfg.input_filepaths is None:
        if len(sys.argv) < 2:
            print("usage: {} <filepath_or_dir_of_inputs>".format(sys.argv[0]))
            exit()
        else:
            input_fps = sys.argv[1]
    else:
        input_fps = cfg.input_filepaths

    if isinstance(input_fps, str):
        input_fps = [input_fps]
    input_fps = input_fps[:200]

    #input
    inp = T.tensor4("inp")
    #neural network model
    net_model = model.Model(inp, load_net_from=cfg.model_filepath)
    #making prediction function
    pred_f = theano.function([inp], net_model.test_pred)

    preds = np.array([], dtype="float32")
    trues = np.array([], dtype="float32")

    #creating dir if needed
    if not os.path.isdir(cfg.preds_save_dir):
        os.makedirs(cfg.preds_save_dir)

    #iterating over images doing predictions
    for i, fp in enumerate(input_fps):
        print("in image {}".format(fp))

        x, y = dataproc.load(fp, **cfg.load_args_dict)

        print("\tpredicting...")
        print("x shape:", x.shape)
        pred, tot_pred_time, avg_pred_time = stride_predict(x,
            model.Model.INPUT_SHAPE[-2:], cfg.pred_stride, pred_f)
        print("\tdone predicting. took %.6f seconds (%.6f avg)" %\
            (tot_pred_time, avg_pred_time))
        print("\tpred shape:", pred.shape)

        if cfg.preds_csv_fp is not None:
            preds = np.append(preds, pred.flatten())
            print("\tpreds csv shape:", preds.shape)
            trues = np.append(trues, y.flatten().astype("float32"))

        if i < cfg.max_n_preds_save:
            #saving prediction
            _pred = (255*pred).astype("uint8")
            pred_fp = os.path.join(cfg.preds_save_dir,
                "{}_prob.png".format(i))
            io.imsave(pred_fp, _pred)

            #saving ground-truth
            gt = (255*y).astype("uint8")
            gt = gt.reshape(gt.shape[-2:])
            gt_fp = os.path.join(cfg.preds_save_dir,
                "{}_gt.png".format(i))
            io.imsave(gt_fp, gt)

    #saving predictions
    if cfg.preds_csv_fp is not None:
        df = pd.DataFrame({"pred": preds, "true": trues})
        df.to_csv(cfg.preds_csv_fp, index=False)

if __name__ == '__main__':
    main()

