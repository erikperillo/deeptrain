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

import tensorflow as tf
import os
import sys
import shutil

import model
import trloop
import config as conf
import util

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
    #copying config file to dir
    shutil.copy(conf.__file__, os.path.join(out_dir, "config.py"))
    #creating dir to store model weights
    os.makedirs(os.path.join(out_dir, "checkpoints"))
    #info file
    with open(os.path.join(out_dir, "info.txt"), "w") as f:
        print("date created (y-m-d):", util.date_str(), file=f)
        print("time created:", util.time_str(), file=f)
        print("git commit hash:", util.git_hash(), file=f)

def main():
    out_dir = mk_output_dir(conf.train["out_dir_basedir"])
    print("created output dir '{}'".format(out_dir))
    populate_output_dir(out_dir)

    #meta-model
    meta_model = model.MetaModel(conf.model["build_graph_fn"])

    #creating logging object
    log = util.Tee([sys.stdout, open(os.path.join(out_dir, "train.log"), "w")])

    #building graph
    if conf.train["pre_trained_model_path"] is None:
        log.print("[info] building graph for the first time")
        graph = meta_model.build_graph()
    else:
        graph = tf.Graph()

    #training session
    with tf.Session(graph=graph) as sess:
        #if first time training, creates graph collections for model params
        #else, loads model weights and params from collections
        if conf.train["pre_trained_model_path"] is None:
            sess.run(tf.global_variables_initializer())
            meta_model.mk_params_colls(graph=graph)
        else:
            log.print("[info] loading graph/weights from '{}'".format(
                conf.train["pre_trained_model_path"]))
            model.load(sess, conf.train["pre_trained_model_path"])
            meta_model.set_params_from_colls(graph=graph)

        #building functions
        #train function: cumputes loss
        train_fn = meta_model.get_train_fn(sess)

        #test function: returns a dict with pairs metric_name: metric_value
        _test_fn = meta_model.get_test_fn(sess)
        def test_fn(x, y_true):
            metrics_values = _test_fn(x, y_true)
            return {
                k: m for k, m in zip(
                    meta_model.params["metrics"].keys(), metrics_values)
            }

        #save model function: given epoch and iteration number, saves checkpoint
        def save_model_fn(epoch, it):
            path = os.path.join(out_dir, "checkpoints",
                "epoch-{}_it-{}".format(epoch, it))
            model.save(sess, path, overwrite=True)
            print("    saved checkpoint to '{}'".format(path))

        import random
        #main train loop
        print("calling train loop")
        try:
            trloop.train_loop(
                train_set=conf.train["train_set_fps"],
                train_fn=train_fn,
                #train_fn=lambda x, y: random.uniform(0, 1),
                n_epochs=conf.train["n_epochs"],
                val_set=conf.train["val_set_fps"],
                val_fn=test_fn,
                #val_fn=lambda x, y: {"acc": random.uniform(0, 1), "hue": 10},
                val_every_its=conf.train["val_every_its"],
                save_model_fn=save_model_fn,
                save_every_its=conf.train["save_every_its"],
                verbose=conf.train["verbose"],
                print_fn=log.print,
                batch_gen_kw=conf.train["batch_gen_kw"]
            )
        except KeyboardInterrupt:
            print("Keyboard Interrupt event.")

        #saving model on final state
        path = os.path.join(out_dir, "checkpoints", "final")
        print("saving checkpoint to '{}'...".format(path), flush=True)
        model.save(sess, path, overwrite=True)

    print("\ndone.", flush=True)

if __name__ == '__main__':
    main()
