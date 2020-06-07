#%%
import numpy as np
import argparse
import yaml
from pathlib import Path
import os

import tensorflow as tf
from tensorflow import keras as tk

from models.hrnet import HRNet
from models.vggunet import Vggunet

from models.callback import Custom_Callback
from dataparser.inria import Inria, Inria_v


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


    repeat = config["epoch"]*data_parser.steps
    repeatv = config["epoch"]*data_parserv.steps
    dataset = tf.data.Dataset.from_generator(
        data_parser.generator,
        (tf.float32, tf.float32),
        (tf.TensorShape([config["image_size"][0], config["image_size"][1], 3]), tf.TensorShape([config["image_size"][0], config["image_size"][1], 1]))
    ).batch(config["batch_size"], drop_remainder=True)
    datasetv = tf.data.Dataset.from_generator(
        data_parserv.generator,
        (tf.float32, tf.float32),
        (tf.TensorShape([config["image_size"][0], config["image_size"][1], 3]), tf.TensorShape([config["image_size"][0], config["image_size"][1], 1]))
    ).batch(config["batch_size"], drop_remainder=True)

    # model checkpoint has bug in the 2.0 version so wait until 2.1 is release and use custom callback
    # saver = tk.callbacks.ModelCheckpoint(config["save_path"])
    logger = tk.callbacks.CSVLogger(config["logger_file"], append=True)
    model_cb = Custom_Callback(config)


    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        if config["model_name"] == "hrnet" : 
            the_model = HRNet(configs=config)
        elif config["model_name"] == "vggunet" :
            the_model = Vggunet(configs=config)

        print(the_model.model)


    # print(data_parser.steps)
    the_model.model.fit(
        dataset,
        epochs=config["epoch"],
        callbacks=[model_cb, logger],
        validation_data=datasetv,
        validation_freq=1,
        # steps_per_epoch=data_parser.steps,
        # class_weight=config["class_weight"],
        initial_epoch=config["present_epoch"]
        )

    the_model.model.save(str(Path(config["save_path"])/f"model_{config['epoch']}.h5"))

