#%%
import tensorflow as tf
from tensorflow import keras as tk
# import keras as tk

from pathlib import Path

class Vggunet :

    def __init__(self, configs) :

        self.configs = configs
        self.image_size = configs["image_size"]
        self.batch_size = configs["batch_size"]

        self.model = self.build_model(self.image_size, one_hot_label=True)

        # optimizer=Adam(lr=1e-5)#RMSprop(lr=0.0001, rho=0.9, epsilon=None, decay=0.995)#Adam(lr=1e-4)
        optimizer = tk.optimizers.Adam(lr=1e-5)

        self.smoothing = 100
        ##--- Model compile and training
        self.model.compile(optimizer=optimizer, loss=self.sce_loss, metrics=[self.iou0, self.iou1])
        # self.model.compile(optimizer=optimizer, loss=self.jaccard_loss, metrics=[self.iou0, self.iou1])

        self.load_weight(configs)


    def build_model(self, input_shape, one_hot_label):

        vgg16_model = tk.applications.vgg16.VGG16(input_shape=input_shape+[3], weights='imagenet', include_top=False)

        # vgg16_model.trainable = True
        # for layer in vgg16_model.layers:
        #     layer.trainable = True

        block4_pool = vgg16_model.get_layer('block4_pool').output
        block5_conv1 = tk.layers.Conv2D(1024, kernel_size=3, strides=1, activation='relu', padding='same', kernel_initializer='he_normal')(block4_pool)
        block5_conv2 = tk.layers.Conv2D(1024, kernel_size=3, strides=1, activation='relu', padding='same', kernel_initializer='he_normal')(block5_conv1)
        block5_drop = tk.layers.Dropout(0.5)(block5_conv2)

        block6_up = tk.layers.Conv2D(512, kernel_size=2, strides=1, activation='relu', padding='same', kernel_initializer='he_normal')(
            tk.layers.UpSampling2D(size=(2, 2))(block5_drop))
        block6_merge = tk.layers.Concatenate(axis=3)([vgg16_model.get_layer('block4_conv3').output, block6_up])
        block6_conv1 = tk.layers.Conv2D(512, kernel_size=3, strides=1, activation='relu', padding='same', kernel_initializer='he_normal')(block6_merge)
        block6_conv2 = tk.layers.Conv2D(512, kernel_size=3, strides=1, activation='relu', padding='same', kernel_initializer='he_normal')(block6_conv1)
        block6_conv3 = tk.layers.Conv2D(512, kernel_size=3, strides=1, activation='relu', padding='same', kernel_initializer='he_normal')(block6_conv2)

        block7_up = tk.layers.Conv2D(256, kernel_size=2, strides=1, activation='relu', padding='same', kernel_initializer='he_normal')(
            tk.layers.UpSampling2D(size=(2, 2))(block6_conv3))
        block7_merge = tk.layers.Concatenate(axis=3)([vgg16_model.get_layer('block3_conv3').output, block7_up])
        block7_conv1 = tk.layers.Conv2D(256, kernel_size=3, strides=1, activation='relu', padding='same', kernel_initializer='he_normal')(block7_merge)
        block7_conv2 = tk.layers.Conv2D(256, kernel_size=3, strides=1, activation='relu', padding='same', kernel_initializer='he_normal')(block7_conv1)
        block7_conv3 = tk.layers.Conv2D(256, kernel_size=3, strides=1, activation='relu', padding='same', kernel_initializer='he_normal')(block7_conv2)

        block8_up = tk.layers.Conv2D(128, kernel_size=2, strides=1, activation='relu', padding='same', kernel_initializer='he_normal')(
            tk.layers.UpSampling2D(size=(2, 2))(block7_conv3))
        block8_merge = tk.layers.Concatenate(axis=3)([vgg16_model.get_layer('block2_conv2').output, block8_up])
        block8_conv1 = tk.layers.Conv2D(128, kernel_size=3, strides=1, activation='relu', padding='same', kernel_initializer='he_normal')(block8_merge)
        block8_conv2 = tk.layers.Conv2D(128, kernel_size=3, strides=1, activation='relu', padding='same', kernel_initializer='he_normal')(block8_conv1)

        block9_up = tk.layers.Conv2D(64, kernel_size=2, strides=1, activation='relu', padding='same', kernel_initializer='he_normal')(
            tk.layers.UpSampling2D(size=(2, 2))(block8_conv2))
        block9_merge = tk.layers.Concatenate(axis=3)([vgg16_model.get_layer('block1_conv2').output, block9_up])
        block9_conv1 = tk.layers.Conv2D(64, kernel_size=3, strides=1, activation='relu', padding='same', kernel_initializer='he_normal')(block9_merge)
        block9_conv2 = tk.layers.Conv2D(64, kernel_size=3, strides=1, activation='relu', padding='same', kernel_initializer='he_normal')(block9_conv1)

        net = tk.layers.Conv2D(64, kernel_size=3, strides=1, activation='relu', padding='same', kernel_initializer='he_normal')(block9_conv2)
        
        if one_hot_label:
            net = tk.layers.Conv2D(2, kernel_size=1, strides=1, activation='relu', padding='same')(net)
            # net = tk.layers.Reshape((2,input_shape[0]*input_shape[1]))(net)
            # net = tk.layers.Permute((2,1))(net)
            net = tk.layers.Softmax(axis=3)(net)
        else:
            net = tk.layers.Conv2D(1, kernel_size=1, strides=1, activation='sigmoid')(net)
            
    #     block10_conv2 = tk.layers.Conv2D(1, kernel_size=1, strides=1, activation='sigmoid')(block10_conv1)

        model = tk.Model(inputs=vgg16_model.input, outputs=net)

        return model

    def iou_calculation (self, y_true, y_pred, class_number) :

        i = class_number
        y_true = self.rgb_to_label_tf(y_true, self.configs)
        y_pred = tf.argmax(y_pred, axis=3)

        y_pred_for_i = tf.where(condition=tf.equal(y_pred, i), x=1, y=0)
        y_pred_for_i = tf.cast(y_pred_for_i, dtype=tf.float32)

        # print(y_true[:, :, :, i])
        # print(y_pred_for_i)
        sumsum = y_true[:, :, :, i] + y_pred_for_i


        inters = tf.reduce_sum(tf.where(condition=tf.equal(sumsum, 2), x=1, y=0))
        union = tf.reduce_sum(tf.where(condition=tf.greater_equal(sumsum, 1), x=1, y=0))
        
        iou = inters/union

        return iou

    def iou0 (self, y_true, y_pred) :
        return self.iou_calculation(y_true, y_pred, 0)
    
    def iou1 (self, y_true, y_pred) :
        return self.iou_calculation(y_true, y_pred, 1)

    def rgb_to_label_tf (self, y_true, configs) :

        label_true = []
        for i in range(len(configs["class_color_map"])) :
            label = tf.where(condition=tf.equal(y_true, configs["class_color_map"][i]), x=1, y=0)
            label_true.append(label)

        label_true = tf.concat(label_true, axis=-1)
        label_true = tf.cast(label_true, tf.float32)

        # print(label_true)

        return label_true
    
    def bce_loss (self, y_true, y_pred) :

        y_true = self.rgb_to_label_tf(y_true, self.configs)

        return tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)

    def sce_loss (self, y_true, y_pred) :

        y_true = self.rgb_to_label_tf(y_true, self.configs)

        class_weights = tf.reduce_sum(tf.constant([self.configs["class_weight"]]) * y_true, axis=3)
        sce = tf.nn.softmax_cross_entropy_with_logits(y_true, y_pred)
        sce = tf.reduce_mean(sce * class_weights)

        return sce

    def jaccard_loss (self, y_true, y_pred) :

        # j0 = tf.math.subtract(1., self.iou0(y_true=y_true, y_pred=y_pred))*self.smoothing
        # j1 = tf.math.subtract(1., self.iou1(y_true=y_true, y_pred=y_pred))*self.smoothing

        i = 1
        y_true = self.rgb_to_label_tf(y_true, self.configs)
        # y_pred = tf.argmax(y_pred, axis=3)

        # y_pred_for_i = tf.where(condition=tf.equal(y_pred, i), x=1, y=0)
        # y_pred_for_i = tf.cast(y_pred_for_i, dtype=tf.float32)

        # print(y_true[:, :, :, i])
        # print(y_pred_for_i)
        # sumsum = y_true[:, :, :, i] + y_pred_for_i

        # inters = tf.reduce_sum(tf.where(condition=tf.equal(sumsum, 2), x=1, y=0))
        # union = tf.reduce_sum(tf.where(condition=tf.greater_equal(sumsum, 1), x=1, y=0))

        # jacc = (1. - (inters+self.smoothing)/(union+self.smoothing))*self.smoothing
        intersection = tf.reduce_sum(tf.abs(y_true * y_pred), axis=-1)
        sum_ = tf.reduce_sum(tf.abs(y_true) + tf.abs(y_pred), axis=-1)
        jac = (intersection + self.smoothing) / (sum_ - intersection + self.smoothing)

        return (1 - jac) * self.smoothing

        # j0 = (1. - self.iou0(y_true=y_true, y_pred=y_pred))*self.smoothing
        # j1 = (1. - self.iou1(y_true=y_true, y_pred=y_pred))*self.smoothing

        # return j0 + j1


    def load_weight (self, configs) :

        # if configs["present_epoch"] != 0 :
        if configs["mode"] == 0 :
            pass
        elif configs["mode"] == 1 or (configs["mode"] == 2 and not configs["test"]["best"]) :
            weight_path = Path(configs["save_path"])/(f"model_{str(configs['present_epoch'])}.h5")
            custom_objs = {
                "sce_loss" : self.sce_loss,
                "iou0" : self.iou0,
                "iou1" : self.iou1
            }
            self.model.load_weights(str(weight_path))
        elif configs["mode"] == 2 and configs["test"]["best"] :
            # weight_path = Path(configs["save_path"])/"best.h5"
            weight_path = Path(configs["save_path"])/configs["test"]["best_file_name"]
            # print(weight_path)
            custom_objs = {
                "sce_loss" : self.sce_loss,
                "iou0" : self.iou0,
                "iou1" : self.iou1
            }
            self.model.load_weights(str(weight_path))

