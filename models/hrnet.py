#%%

import tensorflow as tf
from tensorflow import keras as tk
import numpy as np

import utils.util as util
from pathlib import Path

#%%

class HRNet :

    def __init__(self, configs) :

        self.configs = configs
        self.image_size = configs["image_size"]
        self.batch_size = configs["batch_size"]

        # self.input_gt = tk.Input(shape=self.image_size, batch_size=self.batch_size, name="input_gt", dtype=tf.int32)

        # tensorflow.python.framework.ops.disable_eager_mode
        # tf.compat.v1.disable_eager_execution()

        self.input_image = tk.Input(shape=(self.image_size+[3]), name="input_image", dtype=tf.float32)
        self.model = self.build_model()

        # self.model.summary()

        # multi gpu code
        # if len(configs["gpu_indices"]) > 1 :
        #     self.model = tk.utils.multi_gpu_model(self.model, gpus=len(configs["gpu_indices"]))

        self.build_loss_and_op(self.model)

        self.load_weight(configs)

        self.tmp = tf.Variable(tf.zeros((configs["num_classes"], 2)))


    def cbr (self, net, channels, name="", i=0) :
        net = tk.layers.Conv2D(filters=channels, kernel_size=3, strides=1, padding="SAME")(net)
        net = tk.layers.BatchNormalization()(net)
        net = tk.layers.ReLU(6, name=name + "_1_" + str(i))(net)
        return net
    
    def cb (self, net, channels) :
        net = tk.layers.Conv2D(filters=channels, kernel_size=3, strides=1, padding="SAME")(net)
        net = tk.layers.BatchNormalization()(net)
        return net

    def stage (self, input_layer, channels, name="") :
        net = input_layer
        for i in range(4) :
            residual = net
            net = self.cbr(net, channels, name, i)
            net = self.cb(net, channels)
            net = tk.layers.Add()([net, residual])
            net = tk.layers.ReLU(6, name=name + "_2_" + str(i))(net)
        return net

    def downsample (self, input_layer, downsize, channels) :

        net = tk.layers.Conv2D(filters=channels, kernel_size=3, strides=downsize, padding="SAME")(input_layer)
        net = tk.layers.BatchNormalization()(net)
        net = tk.layers.ReLU(6)(net)
        
        return net

    def upsample (self, input_layer, upsize, channels) :

        net = tk.layers.Conv2D(channels, 1, 1, padding="SAME")(input_layer)
        net = tk.layers.BatchNormalization()(net)
        net = tk.layers.ReLU(6)(net)
        # net = tk.layers.UpSampling2D(size=upsize, interpolation="bilinear")(net)
        net = tk.layers.Lambda(
            lambda x: tf.compat.v1.image.resize_bilinear(x, [x.shape[1]*upsize, x.shape[2]*upsize], align_corners=True),
            output_shape=(net.shape[1]*upsize, net.shape[2]*upsize)
            )(net)
        net = tk.layers.BatchNormalization()(net)
        net = tk.layers.ReLU(6)(net)

        return net
    
    def build_model (self) :

        c = self.configs["model"]["c"]

        # Introducing stem
        stem1 = self.downsample(self.input_image, 2, int(c/2))
        stem2 = self.downsample(stem1, 2, int(c/2))

        after_stem2 = self.cbr(stem2, c, "after_stem2")
        stage1 = self.stage(after_stem2, c, "stage1")

        fused1_1 = self.cbr(stage1, c, "fused1_1")
        fused1_2 = self.downsample(stage1, 2, c*2)

        stage2_r1 = self.stage(fused1_1, c, "stage2_r1")
        stage2_r2 = self.stage(fused1_2, c*2, "stage2_r2")

        fused2_1 = tk.layers.add([
            self.cbr(stage2_r1, c, "fused2_1"),
            self.upsample(stage2_r2, 2, c)
        ])
        fused2_2 = tk.layers.add([
            self.downsample(stage2_r1, 2, c*2),
            self.cbr(stage2_r2, c*2, "fused2_2")
        ])
        fused2_3 = tk.layers.add([
            self.downsample(stage2_r1, 4, c*4),
            self.downsample(stage2_r2, 2, c*4)
        ])

        stage3_r1 = self.stage(fused2_1, c, "stage3_r1")
        stage3_r2 = self.stage(fused2_2, c*2, "stage3_r2")
        stage3_r3 = self.stage(fused2_3, c*4, "stage3_r3")

        fused3_1 = tk.layers.add([
            self.cbr(stage3_r1, c, "fused3_1"),
            self.upsample(stage3_r2, 2, c),
            self.upsample(stage3_r3, 4, c)
        ])
        fused3_2 = tk.layers.add([
            self.downsample(stage3_r1, 2, c*2),
            self.cbr(stage3_r2, c*2, "fused3_2"),
            self.upsample(stage3_r3, 2, c*2)
        ])
        fused3_3 = tk.layers.add([
            self.downsample(stage3_r1, 4, c*4),
            self.downsample(stage3_r2, 2, c*4),
            self.cbr(stage3_r3, c*4, "fused3_3")
        ])
        fused3_4 = tk.layers.add([
            self.downsample(stage3_r1, 8, c*8),
            self.downsample(stage3_r2, 4, c*8),
            self.downsample(stage3_r3, 2, c*8),
        ])

        stage4_r1 = self.stage(fused3_1, c, "stage4_r1")
        stage4_r2 = self.stage(fused3_2, c*2, "stage4_r2")
        stage4_r3 = self.stage(fused3_3, c*4, "stage4_r3")
        stage4_r4 = self.stage(fused3_4, c*8, "stage4_r4")


        upsampled_output = tk.layers.concatenate([
            stage4_r1,
            self.upsample(stage4_r2, 2, c*2),
            self.upsample(stage4_r3, 4, c*4),
            self.upsample(stage4_r4, 8, c*8)
        ])

        num_classes = self.configs["num_classes"]
        logits = tk.layers.Conv2D(num_classes, 1, 1, padding="SAME")(upsampled_output)
        
        # print(logits.shape)
        # restore the size
        # logits = tk.layers.UpSampling2D(size=2, interpolation="bilinear")(logits)
        # logits = tk.layers.UpSampling2D(size=2, interpolation="bilinear")(logits)
        logits = tk.layers.Lambda(
            lambda x: tf.compat.v1.image.resize_bilinear(x, [x.shape[1]*2, x.shape[2]*2], align_corners=True),
            output_shape=(logits.shape[1]*2, logits.shape[2]*2)
            )(logits)
        logits = tk.layers.Lambda(
            lambda x: tf.compat.v1.image.resize_bilinear(x, [x.shape[1]*2, x.shape[2]*2], align_corners=True),
            output_shape=(logits.shape[1]*2, logits.shape[2]*2)
            )(logits)
        # logits = tk.layers.Lambda(lambda x: tf.compat.v1.image.resize_bilinear(x, [x.shape[2]*2, x.shape[2]*2], align_corners=True))(logits)
        # logits = tk.layers.Lambda(lambda x: tf.compat.v1.image.resize_bilinear(x, [x.shape[2]*2, x.shape[2]*2], align_corners=True))(logits)

        self.logits = logits
        # self.output = tk.layers.Softmax(axis=3)(logits)
        self.output = logits

        model = tk.Model(inputs=self.input_image, outputs=self.output)

        return model


    def wce_loss (self, y_true, y_pred) :

        y_true = self.rgb_to_label_tf(y_true, self.configs)

        # tmptmp = tf.nn.softmax_cross_entropy_with_logits(y_true, y_pred, axis=3)

        class_weights_mask = tf.constant(self.configs["class_weight"])
        wce = tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, self.configs["wce_weight"])
        wce = tf.reduce_mean(wce, axis=[1, 2]) * class_weights_mask
        wce = tf.reduce_mean(wce)

        return wce

    def sce_loss (self, y_true, y_pred) :

        y_true = self.rgb_to_label_tf(y_true, self.configs)

        class_weights = tf.reduce_sum(tf.constant([self.configs["class_weight"]]) * y_true, axis=3)
        sce = tf.nn.softmax_cross_entropy_with_logits(y_true, y_pred)
        sce = tf.reduce_mean(sce * class_weights)

        return sce
    
    def bce_loss (self, y_true, y_pred) :

        y_true = self.rgb_to_label_tf(y_true, self.configs)

        # class_weights = tf.reduce_sum(tf.constant([self.configs["class_weight"]]) * y_true, axis=3)
        bce = tf.nn.sigmoid_cross_entropy_with_logits(y_true, y_pred)
        bce = tf.reduce_mean(bce)

        return bce


    def pixel_accuracy (self, y_true, y_pred) :
        y_true = tf.cast(tf.reduce_mean(y_true, axis=-1), dtype=tf.int32)
        y_pred = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
        tmp = tf.where(condition=tf.equal(y_true, y_pred), x=1, y=0)
        return tf.reduce_mean(tf.cast(tmp, dtype=tf.float32))

    def build_loss_and_op (self, model) :

        optim = tk.optimizers.Adam(learning_rate=self.configs["lr"])
        model.compile(optim, loss=self.bce_loss, metrics=[self.pixel_accuracy])

    def rgb_to_label_tf (self, y_true, configs) :
        
        label_true = tf.one_hot(tf.cast(tf.reduce_mean(y_true, axis=-1), tf.int32), configs["num_classes"], axis=-1)
        label_true = tf.cast(label_true, tf.float32)
        return label_true

    def load_weight (self, configs) :

        # if configs["present_epoch"] != 0 :
        if configs["mode"] == 0 :
            pass
        elif configs["mode"] == 1 or (configs["mode"] == 2 and not configs["test"]["best"]) :
            weight_path = Path(configs["save_path"])/(f"model_{str(configs['present_epoch'])}.h5")
            custom_objs = {
                # "sce_loss" : self.sce_loss,
                "wce_loss" : self.wce_loss,
                "pixel_accuracy" : self.pixel_accuracy
            }
            self.model.load_weights(str(weight_path))
        elif configs["mode"] == 2 and configs["test"]["best"] :
            # weight_path = Path(configs["save_path"])/"best.h5"
            weight_path = Path(configs["save_path"])/configs["test"]["best_file_name"]
            # print(weight_path)
            custom_objs = {
                # "sce_loss" : self.sce_loss,
                "wce_loss" : self.wce_loss,
                "pixel_accuracy" : self.pixel_accuracy
            }
            self.model.load_weights(str(weight_path))

