#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.contrib import keras
from tensorflow.examples.tutorials.mnist import input_data

from collections import OrderedDict
from matplotlib import pyplot as plt
import os
import shutil
import time

import model

def _build_graph():
    params = {}
    #placeholders
    params["x"] = tf.placeholder("float32", shape=(None, 28*28), name="x")
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

def conv_net_metamodel(save_to, load_from=None):
    #getting data
    mnist = input_data.read_data_sets("./data", one_hot=True)

    meta_model = model.MetaModel(_build_graph)

    graph = meta_model.build_graph() if load_from is None else tf.Graph()

    #optimizing
    with tf.Session(graph=graph) as sess:
        if load_from is None:
            print("built model graph for first time")
            sess.run(tf.global_variables_initializer())
        else:
            print("loading graph from", load_from)
            model.load(sess, load_from)
            print(sess.graph.get_all_collection_keys())
            meta_model.set_params_from_colls()

        train_fn = meta_model.get_train_fn(sess)
        test_fn = meta_model.get_test_fn(sess)
        pred_fn = meta_model.get_pred_fn(sess)

        loss = None
        n_iters = 120
        start_time = time.time()
        for i in range(n_iters):
            print("on iter {}/{} (loss = {})".format(i+1, n_iters, loss),
                end="    \r", flush=True)
            #getting next train batch
            batch = mnist.train.next_batch(50)

            #loggin result every 100 steps
            if i % 100 == 0:
                metrics = test_fn(batch[0], batch[1])
                print("\nstep %d, metrics:" % i, metrics)

            #updating parameters
            loss = train_fn(batch[0], batch[1])
        end_time = time.time() - start_time
        print("elapsed training time:", end_time)

        #printing result
        print("\nfinal metrics:",
            test_fn(mnist.test.images, mnist.test.labels))

        #making a prediction
        img = mnist.validation.images[77]
        plt.imshow(img.reshape((28, 28)), cmap="Greys")
        plt.show()
        pred = pred_fn(img.reshape((1, 28*28)))
        print(pred, "\n", sess.run(tf.argmax(pred, 1)))

        if load_from is None:
            meta_model.mk_params_colls()
        model.save(sess, save_to, overwrite=True)

def predict_metamodel(load_from):
    #getting data
    mnist = input_data.read_data_sets("./data", one_hot=True)

    meta_model = model.MetaModel(_build_graph)

    with tf.Session() as sess:
        model.load(sess, load_from)
        meta_model.set_params_from_colls()

        for k, v in meta_model.params.items():
            print(k, "->", v)

        test_fn = meta_model.get_test_fn(sess)
        #test_fn = get_test_fn(sess, meta_model.params)
        print("\nmetrics:", test_fn(mnist.test.images, mnist.test.labels))
        exit()

        #making a prediction
        img = mnist.validation.images[88]
        plt.imshow(img.reshape((28, 28)), cmap="Greys")
        plt.show()
        pred_fn = meta_model.get_train_fn(sess)
        pred = pred_fn(img.reshape((1, 28*28)))
        print(pred, "\n", sess.run(tf.argmax(pred, 1)))

def main():
    #conv_net()
    #conv_net_keras()
    #conv_net_metamodel("./eyb0ss", "./eyb0ss")
    #linear_classifier()
    #predict()
    predict_metamodel("./eyb0ss")
    #conv_net_retrain_metamodel()
    pass

if __name__ == "__main__":
    main()
