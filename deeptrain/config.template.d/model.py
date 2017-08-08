import numpy as np

import theano
from theano import tensor as T

import lasagne
from lasagne.layers import InputLayer
from lasagne.layers import ConcatLayer
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer

import os
import sys
sys.path.append(os.path.join(".."))
import objectives as obj

def set_layer_as_immutable(layer):
    """Sets layer parameters so as not to be modified in training steps."""
    for k in layer.params.keys():
        layer.params[k] -= {"regularizable", "trainable"}

def acc(pred, tgt, thr=0.5):
    """Accuracy."""
    return T.mean(T.eq((pred >= thr), tgt))

def precision(pred, tgt, thr=0.5):
    """Mean-absolute-error."""
    tp = T.sum((pred >= thr)*tgt)
    fp = T.sum((pred >= thr)*(1 - tgt))
    return T.switch(T.eq(tp + fp, 0), 0, tp/(tp + fp))

def recall(pred, tgt, thr=0.5):
    """Mean-absolute-error."""
    tp = T.sum((pred >= thr)*tgt)
    p = T.sum(tgt)
    return T.switch(T.eq(p, 0), 0, tp/p)

def bn_conv(input_layer, **kwargs):
    layer = lasagne.layers.batch_norm(input_layer, epsilon=1e-7)
    layer = lasagne.layers.Conv2DLayer(layer, **kwargs)
    return layer

def bn_deconv(input_layer, **kwargs):
    layer = lasagne.layers.batch_norm(input_layer, epsilon=1e-7)
    layer = lasagne.layers.Deconv2DLayer(layer, **kwargs)
    return layer

class Model:
    #in format depth, rows, cols
    INPUT_SHAPE = (3, 224, 224)
    OUTPUT_SHAPE = (1, 224, 224)

    def __init__(self, input_var=None, target_var=None, load_net_from=None):
        self.input_var = T.tensor4('inps') if input_var is None else input_var
        self.target_var = T.matrix('tgts') if target_var is None else target_var

        #the network lasagne model
        self.net = self.get_net_model(input_var)
        if load_net_from is not None:
            self.load_net(load_net_from)

        #prediction train/test symbolic functions
        self.train_pred = lasagne.layers.get_output(self.net["output"],
            deterministic=False)
        self.test_pred = lasagne.layers.get_output(self.net["output"],
            #deterministic=True)
            deterministic=True)

        #loss train/test symb. functionS
        self.train_loss = obj.bin_cross_entr(self.train_pred, self.target_var)
        #optional regularization term
        #reg = lasagne.regularization.regularize_network_params(
        #    self.net["output"],
        #    lasagne.regularization.l2)
        #self.train_loss += reg*0.00003
        self.test_loss = obj.bin_cross_entr(self.test_pred, self.target_var)

        #updates symb. function for gradient descent
        self.params = lasagne.layers.get_all_params(self.net["output"],
            trainable=True)

        #updates parameters
        self.learning_rate = theano.shared(np.array(0.002,
            dtype=theano.config.floatX))
        self.lr_decay = 0.95
        self.min_lr = 1e-6
        self.updates = lasagne.updates.nesterov_momentum(
            self.train_loss, self.params, learning_rate=self.learning_rate,
            momentum=0.9)

        #train function
        self.train_fn = theano.function(
            inputs=[self.input_var, self.target_var],
            outputs={
                "acc": acc(self.train_pred, self.target_var),
                "xe": self.train_loss,
                "mse": obj.mse(self.train_pred, self.target_var),
            },
            updates=self.updates
        )
        #val function
        self.val_fn = theano.function(
            inputs=[self.input_var, self.target_var],
            outputs={
                "xe": self.test_loss,
                "acc": acc(self.test_pred, self.target_var),
                "prec": precision(self.test_pred, self.target_var),
                "recall": recall(self.test_pred, self.target_var),
                #"mse": obj.mse(self.test_pred, self.target_var),
            }
        )

    def update_learning_rate(self):
        """
        Updates learning rate.
        """
        if self.lr_decay is None:
            return
        old_lr = self.learning_rate.get_value()
        new_lr = max(old_lr*self.lr_decay, self.min_lr)
        self.learning_rate.set_value(np.array(new_lr,
            dtype=theano.config.floatX))
        return old_lr, new_lr

    def get_net_model(self, input_var=None, inp_shp=None):
        """
        Builds network.
        """
        if inp_shp is None:
            inp_shp = (None,) + Model.INPUT_SHAPE

        net = {}

        #input: 112x112
        net["input"] = lasagne.layers.InputLayer(shape=inp_shp,
            input_var=input_var)

        #224x224
        net["conv0.1"] = lasagne.layers.Conv2DLayer(net["input"],
            num_filters=32, filter_size=(3, 3), stride=1, flip_filters=False,
            nonlinearity=lasagne.nonlinearities.rectify,
            pad="same")
        net["conv0.2"] = bn_conv(net["conv0.1"],
            num_filters=32, filter_size=(3, 3), stride=1, flip_filters=False,
            nonlinearity=lasagne.nonlinearities.rectify,
            pad="same")
        net["conv0.3"] = bn_conv(net["conv0.2"],
            num_filters=32, filter_size=(3, 3), stride=1, flip_filters=False,
            nonlinearity=lasagne.nonlinearities.rectify,
            pad="same")
        net["pool0"] = lasagne.layers.MaxPool2DLayer(net["conv0.3"],
            pool_size=(2, 2))

        #112x112
        net["conv1.1"] = bn_conv(net["pool0"],
            num_filters=48, filter_size=(3, 3), stride=1, flip_filters=False,
            nonlinearity=lasagne.nonlinearities.rectify,
            pad="same")
        net["conv1.2"] = bn_conv(net["conv1.1"],
            num_filters=48, filter_size=(3, 3), stride=1, flip_filters=False,
            nonlinearity=lasagne.nonlinearities.rectify,
            pad="same")
        net["conv1.3"] = bn_conv(net["conv1.2"],
            num_filters=48, filter_size=(3, 3), stride=1, flip_filters=False,
            nonlinearity=lasagne.nonlinearities.rectify,
            pad="same")
        net["pool1"] = lasagne.layers.MaxPool2DLayer(net["conv1.3"],
            pool_size=(2, 2))

        #56x56
        net["conv2.1"] = bn_conv(net["pool1"],
            num_filters=64, filter_size=(3, 3), stride=1, flip_filters=False,
            nonlinearity=lasagne.nonlinearities.rectify,
            pad="same")
        net["conv2.2"] = bn_conv(net["conv2.1"],
            num_filters=64, filter_size=(3, 3), stride=1, flip_filters=False,
            nonlinearity=lasagne.nonlinearities.rectify,
            pad="same")
        net["conv2.3"] = bn_conv(net["conv2.2"],
            num_filters=64, filter_size=(3, 3), stride=1, flip_filters=False,
            nonlinearity=lasagne.nonlinearities.rectify,
            pad="same")
        net["pool2"] = lasagne.layers.MaxPool2DLayer(net["conv2.3"],
            pool_size=(2, 2))

        #28x28
        net["conv3.1"] = bn_conv(net["pool2"],
            num_filters=96, filter_size=(3, 3), stride=1, flip_filters=False,
            nonlinearity=lasagne.nonlinearities.rectify,
            pad="same")
        net["conv3.2"] = bn_conv(net["conv3.1"],
            num_filters=96, filter_size=(3, 3), stride=1, flip_filters=False,
            nonlinearity=lasagne.nonlinearities.rectify,
            pad="same")
        net["conv3.3"] = bn_conv(net["conv3.2"],
            num_filters=96, filter_size=(3, 3), stride=1, flip_filters=False,
            nonlinearity=lasagne.nonlinearities.rectify,
            pad="same")
        net["pool3"] = lasagne.layers.MaxPool2DLayer(net["conv3.3"],
            pool_size=(2, 2))

        #14x14
        net["conv4.1"] = bn_conv(net["pool3"],
            num_filters=112, filter_size=(3, 3), stride=1, flip_filters=False,
            nonlinearity=lasagne.nonlinearities.rectify,
            pad="same")
        net["conv4.2"] = bn_conv(net["conv4.1"],
            num_filters=112, filter_size=(3, 3), stride=1, flip_filters=False,
            nonlinearity=lasagne.nonlinearities.rectify,
            pad="same")
        net["conv4.3"] = bn_conv(net["conv4.2"],
            num_filters=112, filter_size=(3, 3), stride=1, flip_filters=False,
            nonlinearity=lasagne.nonlinearities.rectify,
            pad="same")
        net["pool4"] = lasagne.layers.MaxPool2DLayer(net["conv4.3"],
            pool_size=(2, 2))

        #7x7
        net["conv5.1"] = bn_conv(net["pool4"],
            num_filters=128, filter_size=(3, 3), stride=1, flip_filters=False,
            nonlinearity=lasagne.nonlinearities.rectify,
            pad="same")
        net["conv5.2"] = bn_conv(net["conv5.1"],
            num_filters=128, filter_size=(3, 3), stride=1, flip_filters=False,
            nonlinearity=lasagne.nonlinearities.rectify,
            pad="same")
        net["conv5.3"] = bn_conv(net["conv5.2"],
            num_filters=128, filter_size=(3, 3), stride=1, flip_filters=False,
            nonlinearity=lasagne.nonlinearities.rectify,
            pad="same")
        net["deconv5"] = bn_deconv(net["conv5.3"],
            num_filters=128, filter_size=3, stride=2, crop=1,
            output_size=(14, 14))

        #14x14
        net["concat6"] = lasagne.layers.ConcatLayer([
            net["deconv5"],
            net["conv4.2"]
        ])
        net["conv6.1"] = bn_conv(net["concat6"],
            num_filters=128, filter_size=(3, 3), stride=1, flip_filters=False,
            nonlinearity=lasagne.nonlinearities.rectify,
            pad="same")
        net["conv6.2"] = bn_conv(net["conv6.1"],
            num_filters=112, filter_size=(3, 3), stride=1, flip_filters=False,
            nonlinearity=lasagne.nonlinearities.rectify,
            pad="same")
        net["deconv6"] = bn_deconv(net["conv6.2"],
            num_filters=112, filter_size=3, stride=2, crop=1,
            output_size=(28, 28))

        #28x28
        net["concat7"] = lasagne.layers.ConcatLayer([
            net["deconv6"],
            net["conv3.2"]
        ])
        net["conv7.1"] = bn_conv(net["concat7"],
            num_filters=112, filter_size=(3, 3), stride=1, flip_filters=False,
            nonlinearity=lasagne.nonlinearities.rectify,
            pad="same")
        net["conv7.2"] = bn_conv(net["conv7.1"],
            num_filters=96, filter_size=(3, 3), stride=1, flip_filters=False,
            nonlinearity=lasagne.nonlinearities.rectify,
            pad="same")
        net["deconv7"] = bn_deconv(net["conv7.2"],
            num_filters=96, filter_size=3, stride=2, crop=1,
            output_size=(56, 56))

        #56x56
        net["concat8"] = lasagne.layers.ConcatLayer([
            net["deconv7"],
            net["conv2.2"]
        ])
        net["conv8.1"] = bn_conv(net["concat8"],
            num_filters=96, filter_size=(3, 3), stride=1, flip_filters=False,
            nonlinearity=lasagne.nonlinearities.rectify,
            pad="same")
        net["conv8.2"] = bn_conv(net["conv8.1"],
            num_filters=64, filter_size=(3, 3), stride=1, flip_filters=False,
            nonlinearity=lasagne.nonlinearities.rectify,
            pad="same")
        net["deconv8"] = bn_deconv(net["conv8.2"],
            num_filters=64, filter_size=3, stride=2, crop=1,
            output_size=(112, 112))

        #112x112
        net["concat9"] = lasagne.layers.ConcatLayer([
            net["deconv8"],
            net["conv1.2"]
        ])
        net["conv9.1"] = bn_conv(net["concat9"],
            num_filters=64, filter_size=(3, 3), stride=1, flip_filters=False,
            nonlinearity=lasagne.nonlinearities.rectify,
            pad="same")
        net["conv9.2"] = bn_conv(net["conv9.1"],
            num_filters=48, filter_size=(3, 3), stride=1, flip_filters=False,
            nonlinearity=lasagne.nonlinearities.rectify,
            pad="same")
        net["deconv9"] = bn_deconv(net["conv9.2"],
            num_filters=48, filter_size=3, stride=2, crop=1,
            output_size=(224, 224))

        #224x224
        net["concat10"] = lasagne.layers.ConcatLayer([
            net["deconv9"],
            net["conv0.2"]
        ])
        net["conv10.1"] = bn_conv(net["concat10"],
            num_filters=48, filter_size=(3, 3), stride=1, flip_filters=False,
            nonlinearity=lasagne.nonlinearities.rectify,
            pad="same")
        net["conv10.2"] = bn_conv(net["conv10.1"],
            num_filters=32, filter_size=(3, 3), stride=1, flip_filters=False,
            nonlinearity=lasagne.nonlinearities.rectify,
            pad="same")

        #output: 224x224
        net["output"] = lasagne.layers.Conv2DLayer(net["conv10.2"],
            num_filters=1, filter_size=(1, 1),
            nonlinearity=lasagne.nonlinearities.sigmoid)

        print("# params:", lasagne.layers.count_params(net["output"]))
        print("shape output:", lasagne.layers.get_output_shape(net["output"]))

        return net

    def save_net(self, filepath):
        """
        Saves net weights.
        """
        np.savez(filepath, *lasagne.layers.get_all_param_values(
            self.net["output"]))

    def load_net(self, filepath):
        """
        Loads net weights.
        """
        with np.load(filepath) as f:
            param_values = [f["arr_%d" % i] for i in range(len(f.files))]
            lasagne.layers.set_all_param_values(self.net["output"],
                param_values)
