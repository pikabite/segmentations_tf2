import tensorflow as tf
import tensorflow.keras as tk

from tensorflow.keras.layers import *
from tensorflow.keras.applications import *
from tensorflow.keras.applications.xception import preprocess_input

from pathlib import Path

class Bisenet :

    def __init__(self, configs) :

        self.configs = configs
        self.image_size = configs["image_size"]
        self.batch_size = configs["batch_size"]

        self.input_image = tk.Input(shape=(self.image_size+[3]), name="input_image", dtype=tf.float32)
        self.model = self.build_model()

        self.build_loss_and_op(self.model)

        self.load_weight(configs)
        
        self.ignore_index = 255


    def ConvAndBatch(self, x, n_filters=64, kernel=(2, 2), strides=(1, 1), padding='valid', activation='relu'):
        filters = n_filters

        conv_ = Conv2D(filters=filters,
                    kernel_size=kernel,
                    strides=strides,
                    padding=padding)

        batch_norm = BatchNormalization()

        activation = Activation(activation)

        x = conv_(x)
        x = batch_norm(x)
        x = activation(x)

        return x


    def ConvAndAct(self, x, n_filters, kernel=(1, 1), activation='relu', pooling=False):
        poolingLayer = AveragePooling2D(pool_size=(1, 1), padding='same')
        convLayer = Conv2D(filters=n_filters,
                        kernel_size=kernel,
                        strides=1)

        if activation != None :
            activation = Activation(activation)

        if pooling:
            x = poolingLayer(x)

        x = convLayer(x)
        if activation != None :
            x = activation(x)

        return x


    def AttentionRefinmentModule(self, inputs, n_filters):
        filters = n_filters

        poolingLayer = AveragePooling2D(pool_size=(1, 1), padding='same')

        x = poolingLayer(inputs)
        x = self.ConvAndBatch(x, kernel=(1, 1), n_filters=filters, activation='sigmoid')

        return multiply([inputs, x])


    def FeatureFusionModule(self, input_f, input_s, n_filters):
        concatenate = Concatenate(axis=-1)([input_f, input_s])

        branch0 = self.ConvAndBatch(concatenate, n_filters=n_filters, kernel=(3, 3), padding='same')
        branch_1 = self.ConvAndAct(branch0, n_filters=n_filters, pooling=True, activation='relu')
        # branch_1 = self.ConvAndAct(branch_1, n_filters=n_filters, pooling=False, activation='sigmoid')
        branch_1 = self.ConvAndAct(branch_1, n_filters=n_filters, pooling=False, activation=None)
        
        x = multiply([branch0, branch_1])
        return Add()([branch0, x])


    def ContextPath(self, layer_13, layer_14):
        globalmax = GlobalAveragePooling2D()

        block1 = self.AttentionRefinmentModule(layer_13, n_filters=1024)
        block2 = self.AttentionRefinmentModule(layer_14, n_filters=2048)

        global_channels = globalmax(block2)
        block2_scaled = multiply([global_channels, block2])

        block1 = UpSampling2D(size=(4, 4), interpolation='bilinear')(block1)
        block2_scaled = UpSampling2D(size=(4, 4), interpolation='bilinear')(block2_scaled)

        cnc = Concatenate(axis=-1)([block1, block2_scaled])

        return cnc


    def build_model(self):
        
        # x = Lambda(lambda image: preprocess_input(image))(inputs)
        x = self.input_image

        xception = Xception(weights='imagenet', input_tensor=x, include_top=False)

        tail_prev = xception.get_layer('block13_pool').output
        tail = xception.output

        layer_13, layer_14 = tail_prev, tail

        x = self.ConvAndBatch(x, 32, strides=2)
        x = self.ConvAndBatch(x, 64, strides=2)
        x = self.ConvAndBatch(x, 156, strides=2)

        # context path
        cp = self.ContextPath(layer_13, layer_14)
        # fusion = self.FeatureFusionModule(cp, x, 32)
        fusion = self.FeatureFusionModule(cp, x, self.configs["num_classes"])
        ans = UpSampling2D(size=(8, 8), interpolation='bilinear')(fusion)

        return tk.Model(inputs=self.input_image, outputs=ans)

        # output = self.FinalModel(x, tail_prev, tail)
        # return inputs, xception.input, output
    

    def sce_loss (self, y_true, y_pred) :

        ignore_mask = tf.where(tf.equal(y_true, self.ignore_index), x=0., y=1.)
        y_true = self.rgb_to_label_tf(y_true, self.configs)

        class_weights = tf.constant([self.configs["class_weight"]])
        if len(class_weights[0]) != 0 :
            weights_processed = tf.reduce_sum(class_weights * y_true, axis=-1)

        sce = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred, axis=-1)

        sce = ignore_mask * sce
        if len(class_weights[0]) != 0 :
            sce = weights_processed * sce
        sce = tf.reduce_mean(sce)

        return sce

    def miou (self, y_true, y_pred) :

        y_true = tf.argmax(self.rgb_to_label_tf(y_true, self.configs), axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)

        return self.miou_op(y_true, y_pred)


    def pixel_accuracy (self, y_true, y_pred) :
    
        ignore_mask = tf.where(tf.equal(y_true, self.ignore_index), x=0, y=1)

        y_true = tf.cast(y_true, dtype=tf.int32) * ignore_mask
        y_pred = tf.argmax(y_pred, axis=-1, output_type=tf.int32) * ignore_mask
        tmp = tf.where(condition=tf.equal(y_true, y_pred), x=1, y=0)

        return tf.reduce_mean(tf.cast(tmp, dtype=tf.float32))

    def build_loss_and_op (self, model) :

        self.miou_op = tk.metrics.MeanIoU(num_classes=self.configs["num_classes"])

        self.optim = tk.optimizers.Adam(learning_rate=self.configs["lr"])
        # model.compile(self.optim, loss=self.sce_loss, metrics=[self.iou0, self.iou1])
        # model.compile(self.optim, loss=self.sce_loss, metrics=[self.pixel_accuracy, self.miou])


    def rgb_to_label_tf (self, y_true, configs) :
        
        y_true = tf.cast(y_true, dtype=tf.int32)
        label_true = tf.one_hot(y_true, configs["num_classes"], axis=-1)
        return label_true


    def load_weight (self, configs) :

        # if configs["present_epoch"] != 0 :
        if configs["mode"] == 0 :
            pass
        elif configs["mode"] == 1 or (configs["mode"] == 2 and not configs["test"]["best"]) :
            weight_path = Path(configs["save_path"])/(f"model_{str(configs['present_epoch'])}.h5")
            custom_objs = {
                "sce_loss" : self.sce_loss,
                "pixel_accuracy" : self.pixel_accuracy,
            }
            self.model.load_weights(str(weight_path))
        elif configs["mode"] == 2 and configs["test"]["best"] :
            # weight_path = Path(configs["save_path"])/"best.h5"
            weight_path = Path(configs["save_path"])/configs["test"]["best_file_name"]
            # print(weight_path)
            custom_objs = {
                "sce_loss" : self.sce_loss,
                "pixel_accuracy" : self.pixel_accuracy,
            }
            self.model.load_weights(str(weight_path))
