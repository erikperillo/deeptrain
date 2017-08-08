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

import theano.tensor as T
import theano
import numpy as np
import shutil
import sys
import os

import trloop
import util
import config.train as cfg
import config.model as model

def mk_output_dir(base_dir, pattern="train"):
    """
    Creates dir to store model.
    """
    #creating dir
    out_dir = util.uniq_filepath(base_dir, pattern)
    os.makedirs(out_dir)
    return out_dir

def populate_output_dir(out_dir):
    """
    Populates outout dir with info files.
    """
    #copying model generator file to dir
    shutil.copy(model.__file__, os.path.join(out_dir, "model.py"))
    #copying this file to dir
    shutil.copy(cfg.__file__, os.path.join(out_dir, "config.py"))
    #info file
    with open(os.path.join(out_dir, "info.txt"), "w") as f:
        print("date created (y-m-d):", util.date_str(), file=f)
        print("time created:", util.time_str(), file=f)
        print("git commit hash:", util.git_hash(), file=f)

def main():
    #theano variables for inputs and targets
    input_var = T.tensor4("inputs", dtype="floatX")
    target_var = T.tensor4("targets", dtype="floatX")

    out_dir = mk_output_dir(cfg.output_dir_basedir)
    print("created output dir '%s'..." % out_dir)
    populate_output_dir(out_dir)

    #neural network model
    print("building network...", flush=True)
    if cfg.pre_trained_model_fp is not None:
        print("loading pre-trained model from '%s'" % cfg.pre_trained_model_fp,
            flush=True)
    net_model = model.Model(input_var, target_var, cfg.pre_trained_model_fp)

    print("compiling functions...", flush=True)
    #compiling function performing a training step on a mini-batch (by giving
    #the updates dictionary) and returning the corresponding training loss
    train_fn = net_model.train_fn
    #second function computing the validation loss and accuracy:
    val_fn = net_model.val_fn

    #creating logging object
    log = util.Tee([sys.stdout, open(os.path.join(out_dir, "train.log"), "w")])

    #function to update learning rate
    def update_learning_rate(epoch):
        old_lr, new_lr = net_model.update_learning_rate()
        print("updated learning rate:", old_lr, "->", new_lr)

    #function to save model
    def save_model(epoch):
        model_fp = os.path.join(out_dir, "model_epoch_{}.npz".format(epoch))
        print("saving after-epoch model to '{}'".format(model_fp))
        net_model.save_net(model_fp)

    #main train loop
    print("calling train loop")
    try:
        trloop.train_loop(
            tr_set=cfg.dataset_train_filepaths,
            tr_f=train_fn,
            n_epochs=cfg.n_epochs,
            val_set=cfg.dataset_val_filepaths,
            val_f=val_fn,
            batches_gen_kwargs=cfg.batches_gen_kwargs,
            verbose=cfg.verbose,
            save_model_after_epoch_f=save_model,
            update_learning_rate_f=update_learning_rate,
            print_f=log.print)
    except KeyboardInterrupt:
        print("Keyboard Interrupt event.")

    model_path = os.path.join(out_dir, "model.npz")
    print("saving model to '{}'".format(model_path), flush=True)
    net_model.save_net(model_path)

    print("\ndone.", flush=True)

if __name__ == '__main__':
    main()
