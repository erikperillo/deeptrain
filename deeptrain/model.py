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
from tensorflow.examples.tutorials.mnist import input_data

import shutil
import os

def get_param_from_coll(coll_key, index=0, max_coll_size=1, graph=None):
    graph = tf.get_default_graph() if graph is None else graph
    coll = graph.get_collection(coll_key)
    assert max_coll_size is None or len(coll) <= max_coll_size
    obj = coll[index]
    return obj

def save(sess, save_dir, overwrite=False, **builder_kwargs):
    if not "tags" in builder_kwargs:
        builder_kwargs["tags"] = []
    if os.path.isdir(save_dir):
        if overwrite:
            shutil.rmtree(save_dir)
        else:
            raise Exception("'{}' exists".format(save_dir))
    builder = tf.saved_model.builder.SavedModelBuilder(save_dir)
    builder.add_meta_graph_and_variables(sess, **builder_kwargs)
    builder.save()

def load(sess, save_dir, **builder_kwargs):
    if not "tags" in builder_kwargs:
        builder_kwargs["tags"] = []
    tf.saved_model.loader.load(sess, export_dir=save_dir, **builder_kwargs)

class MetaModel:
    NAME = "model"

    PARAMS_KEYS = {
        "x",
        "y_pred",
        "y_true",
        "loss",
        "update",
        "learning_phase",
        "metrics",
    }

    @staticmethod
    def get_coll_key_for_param(param_name):
        return "/".join([MetaModel.NAME, "params", param_name])

    def __init__(self, build_graph_fn):
        self.params = {}
        self.build_graph_fn = build_graph_fn

    def build_graph(self, pre_graph=tf.Graph()):
        graph = tf.get_default_graph() if pre_graph is None else pre_graph
        with graph.as_default():
            self.params = self.build_graph_fn()
        assert set(self.params.keys()) == MetaModel.PARAMS_KEYS
        return graph

    def mk_params_colls(self, graph=None):
        graph = tf.get_default_graph() if graph is None else graph
        coll_keys = set(graph.get_all_collection_keys())

        #creating param collections
        for k, p in self.params.items():
            if k == "metrics":
                continue
            coll_key = self.get_coll_key_for_param(k)
            assert coll_key not in coll_keys
            tf.add_to_collection(coll_key, p)

        #creating metrics collections
        for k, p in self.params["metrics"].items():
            coll_key = self.get_coll_key_for_param("metrics/{}".format(k))
            assert coll_key not in coll_keys
            tf.add_to_collection(coll_key, p)

    def set_params_from_colls(self, graph=None):
        graph = tf.get_default_graph() if graph is None else graph

        #getting default parameters
        for k in MetaModel.PARAMS_KEYS - {"metrics"}:
            coll_key = self.get_coll_key_for_param(k)
            self.params[k] = get_param_from_coll(coll_key, graph=graph)

        #getting metrics
        self.params["metrics"] = {}
        metrics_coll_key_pattern = self.get_coll_key_for_param("metrics")
        for k in graph.get_all_collection_keys():
            if k.startswith(metrics_coll_key_pattern):
                self.params["metrics"][k] = get_param_from_coll(k, graph=graph)

    def get_train_fn(self, sess):
        def train_fn(x, y_true):
            __, loss = sess.run([self.params["update"], self.params["loss"]],
                feed_dict={
                    self.params["x"]: x,
                    self.params["y_true"]: y_true,
                    self.params["learning_phase"]: True,
                })
            return loss
        return train_fn

    def get_test_fn(self, sess):
        def test_fn(x, y_true):
            metrics = sess.run(list(self.params["metrics"].values()),
                feed_dict={
                    self.params["x"]: x,
                    self.params["y_true"]: y_true,
                    self.params["learning_phase"]: False,
                })
            return metrics
        return test_fn

    def get_pred_fn(self, sess):
        def pred_fn(x):
            pred = sess.run(self.params["y_pred"],
                feed_dict={
                    self.params["x"]: x,
                    self.params["learning_phase"]: False,
                })
            return pred
        return pred_fn
