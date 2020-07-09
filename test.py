#%%
import numpy as np
import argparse
import yaml
from pathlib import Path
import os
from tqdm import tqdm

import tensorflow as tf
from tensorflow import keras as tk

from models.hrnet import HRNet
from models.vggunet import Vggunet
from models.subject4 import Subject4

from dataparser.inria import Inria, Inria_v

from PIL import Image


#%%

def softmax (a) : 
    c = np.max(a, axis=2, keepdims=True)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a, axis=2, keepdims=True)
    y = exp_a / sum_exp_a
    # print(y.shape)
    return y


# %%

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

    if not config["mode"] == 2 :
        print("Config mode is not for testing!")
        quit()

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus :
        try :
            for i in range(len(config["gpu_indices"])) :
                tf.config.experimental.set_memory_growth(gpus[i], True)
        except RuntimeError as e :
            print(e)


    if config["dataset_name"] == "inria" :
        data_parserv = Inria_v(config)

    repeatv = config["epoch"]*data_parserv.steps
    datasetv = tf.data.Dataset.from_generator(
        data_parserv.generator,
        (tf.float32, tf.float32),
        (tf.TensorShape([config["image_size"][0], config["image_size"][1], 3]), tf.TensorShape([config["image_size"][0], config["image_size"][1], 1]))
    ).batch(config["batch_size"], drop_remainder=False)

    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope() :
        if config["model_name"] == "hrnet" : 
            the_model = HRNet(configs=config)
        elif config["model_name"] == "vggunet" :
            the_model = Vggunet(configs=config)
        elif config["model_name"] == "subject4" :
            the_model = Subject4(configs=config)
        # hrnet = HRNet(configs=config)

        print(the_model.model)

    if config["test"]["eval"] : 

        loss, iou0, iou1 = the_model.model.evaluate(datasetv)

        print(f"loss : {loss}")
        print(f"iou for background : {iou0}")
        print(f"iou for foreground : {iou1}")

    else : 

        saving_folder = Path(config["test"]["output_folder"])
        if not saving_folder.is_dir() :
            saving_folder.mkdir(parents=True)

        i = 0
        for x_data, y_data in tqdm(datasetv) :
            output = the_model.model.predict_on_batch(x_data)

            for ii in range(output.shape[0]) :

                predicted = np.tile(np.expand_dims(((np.argmax(output[ii], axis=2))*255), axis=-1), (1, 1, 3))
                image_name = f"{str(i)}_{str(ii)}.png"

                # image_name = data_parserv.image_list[data_parserv.index_list[config["batch_size"]*i+ii]].name
                # Image.fromarray(((softmax(output[ii])[:, :, 1] > 0.9)*255).astype(np.uint8)).save(str(saving_folder/image_name))
                # predicted = np.tile(np.expand_dims(((np.argmax(output[ii], axis=2))*255), axis=-1), (1, 1, 3))
                gt = np.tile(y_data[ii], (1, 1, 3))
                merged_img = np.concatenate([x_data[ii]*255, gt, predicted], axis=1).astype(np.uint8)
                Image.fromarray(merged_img).save(str(saving_folder/image_name))

            i += 1


#%%
