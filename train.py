#%%
import numpy as np
import argparse
import yaml
from pathlib import Path
import os

import tensorflow as tf
from tensorflow import keras as tk

tf.compat.v1.disable_eager_execution()

from models.hrnet import HRNet
from models.vggunet import Vggunet
from models.subject4 import Subject4
from models.bisenet import Bisenet

from models.callback import Custom_Callback
from dataparser.inria import Inria, Inria_v
from dataparser.ade20k import Ade20k, Ade20k_v

#%%


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = yaml.load("".join(Path(args.config).open("r").readlines()), Loader=yaml.FullLoader)

    print("=====================config=====================")
    for v in config.keys() :
        print("%s : %s" %(v, config[v]))
    print("================================================")

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in config["gpu_indices"]])

    if not (config["mode"] == 0 or config["mode"] == 1) :
        print("Config mode is not for training!")
        quit()

    # for multi gpu
    # tf.compat.v1.disable_eager_execution()

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for i in range(len(config["gpu_indices"])) :
                tf.config.experimental.set_memory_growth(gpus[i], True)
        except RuntimeError as e:
            print(e)


    if config["dataset_name"] == "inria" :
        data_parser = Inria(config)
        data_parserv = Inria_v(config)
    if config["dataset_name"] == "ade20k" :
        data_parser = Ade20k(config)
        data_parserv = Ade20k_v(config)


    repeat = config["epoch"]*data_parser.steps
    repeatv = config["epoch"]*data_parserv.steps
    dataset = tf.data.Dataset.from_generator(
        data_parser.generator,
        (tf.float32, tf.float32),
        (tf.TensorShape([config["image_size"][0], config["image_size"][1], 3]), tf.TensorShape([config["image_size"][0], config["image_size"][1], 3]))
    ).batch(config["batch_size"], drop_remainder=True)
    datasetv = tf.data.Dataset.from_generator(
        data_parserv.generator,
        (tf.float32, tf.float32),
        (tf.TensorShape([config["image_size"][0], config["image_size"][1], 3]), tf.TensorShape([config["image_size"][0], config["image_size"][1], 3]))
    ).batch(config["batch_size"], drop_remainder=True)

    logger = tk.callbacks.CSVLogger(config["logger_file"], append=True)
    model_cb = Custom_Callback(config)

    # lr_scheduler = tk.callbacks.ReduceLROnPlateau(monitor="val_loss")
    def lr_sched(epoch) :
        if epoch < 50 :
            return config["lr"]
        elif epoch < 100 :
            return 0.5 * config["lr"]
        else :
            return 0.25 * config["lr"]
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_sched)

    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        if config["model_name"] == "hrnet" : 
            model = HRNet(configs=config)
        elif config["model_name"] == "vggunet" :
            model = Vggunet(configs=config)
        elif config["model_name"] == "subject4" : 
            model = Subject4(configs=config)
        elif config["model_name"] == "bisenet" : 
            model = Bisenet(configs=config)

        print(model.model)

    print("model loaded")

    # print(data_parser.steps)
    model.model.fit(
        dataset,
        epochs=config["epoch"],
        callbacks=[model_cb, logger, lr_scheduler],
        # callbacks=[model_cb, logger],
        validation_data=datasetv,
        validation_freq=1,
        # steps_per_epoch=data_parser.steps,
        # class_weight=config["class_weight"],
        initial_epoch=config["present_epoch"]
        )

    model.model.save(str(Path(config["save_path"])/f"model_{config['epoch']}.h5"))


