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
from tensorflow.contrib import keras
import numpy as np
import glob

import augment

#data augmentation default operation sequence to be applied in about every op
_def_augm_ops = [
    ("blur", 0.15, {"rng": (0.5, 1.0)}),
    ("translation", 0.15, {"rng": (-0.1, 0.1)}),
    ("rotation", 0.15, {"rng": (-15, 15)}),
    ("shear", 0.15, {"rng": (-0.15, 0.15)}),
    ("add_noise", 0.15, {"rng": (-0.07, 0.07)}),
    ("mul_noise", 0.15, {"rng": (0.9, 1.1)}),
]

"""
Sequences of operations for data augmentation.
Each sequence spans a new image and applies operations randomly as defined by
    their probabilities.
Each sequence must contain operations in format (op_name, op_prob, op_kwargs).
"""
_augment_op_seqs = [
    [
        ("identity", 1.0, {}),
    ],

    #[
    #    ("rot90", 1.0, {"reps": 1}),
    #] + _def_augm_ops,

    #[
    #    ("rot90", 1.0, {"reps": 2}),
    #] + _def_augm_ops,

    #[
    #    ("rot90", 1.0, {"reps": 3}),
    #] + _def_augm_ops,

    #[
    #    ("hmirr", 1.0, {}),
    #] + _def_augm_ops,

    #[
    #    ("hmirr", 1.0, {}),
    #    ("rot90", 1.0, {"reps": 1}),
    #] + _def_augm_ops,

    #[
    #    ("hmirr", 1.0, {}),
    #    ("rot90", 1.0, {"reps": 2}),
    #] + _def_augm_ops,

    #[
    #    ("hmirr", 1.0, {}),
    #    ("rot90", 1.0, {"reps": 3}),
    #] + _def_augm_ops,
]

def _load(fp):
    xy = np.load(fp)
    return xy["x"], xy["y"]

def _pre_proc(xy):
    return xy

def _augment(xy):
    return augment.augment(xy, _augment_op_seqs, apply_on_y=False)

def _build_graph():
    params = {}
    #placeholders
    params["x"] = tf.placeholder("float32", shape=(None, 1, 28, 28), name="x")
    params["y_true"] = tf.placeholder("float32", shape=(None, 10),
        name="y_true")

    #building net
    #reshaping input
    net = tf.reshape(params["x"], (-1, 28, 28, 1))
    #convolution followed by pooling
    net = keras.layers.Conv2D(filters=24, kernel_size=(3, 3),
        padding="same", activation="relu")(net)
    net = keras.layers.MaxPooling2D(
        pool_size=(2, 2), strides=None)(net)
    #second conv layer
    net = keras.layers.Conv2D(filters=48, kernel_size=(3, 3),
        padding="same", activation="relu")(net)
    net = keras.layers.MaxPooling2D(
        pool_size=(2, 2), strides=None)(net)
    #fully-connected layer
    net = keras.layers.Flatten()(net)
    net = keras.layers.Dense(7*7*48, activation="relu")(net)
    #dropout layer
    net = keras.layers.Dropout(0.5)(net)
    #linear regression
    logits = keras.layers.Dense(10, activation=None)(net)
    params["y_pred"] = tf.nn.softmax(logits, name="y_pred")

    #loss function
    params["loss"] = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            labels=params["y_true"], logits=logits), name="loss")

    #update step
    params["update"] = tf.train.AdamOptimizer(1e-4, name="update").minimize(
        params["loss"])

    #metrics
    params["metrics"] = {
        "acc": tf.reduce_mean(
            tf.cast(
                tf.equal(
                    tf.argmax(params["y_pred"], 1),
                    tf.argmax(params["y_true"], 1)),
                "float32"),
            name="acc")
    }

    #learning phase
    params["learning_phase"] = keras.backend.learning_phase()

    return params

model = {
    "build_graph_fn": _build_graph,
}

train = {
    "out_dir_basedir": "/home/erik/random/traindata",

    "pre_trained_model_path": "/home/erik/random/traindata/train_12/checkpoints/epoch-1_it-0",

    "train_set_fps": glob.glob("/home/erik/random/mnist/train/*.npz")[:2000],

    "val_set_fps": glob.glob("/home/erik/random/mnist/val/*.npz"),

    "n_epochs": 3,

    "val_every_its": 200,

    "save_every_its": None,

    "verbose": 2,

    "batch_gen_kw": {
        "batch_size": 4,
        "n_threads": 3,
        "max_n_samples": 1000,
        "fetch_thr_load_chunk_size": 10,
        "fetch_thr_load_fn": _load,
        "fetch_thr_augment_fn": _augment,
        "fetch_thr_pre_proc_fn": _pre_proc,
        "max_augm_factor": len(_augment_op_seqs),
    },
}
