#%%
import numpy as np
import argparse
import yaml
from pathlib import Path
import os
from tqdm import tqdm

import tensorflow as tf
from tensorflow import keras as tk
import tensorflow.keras.backend as K

# tf.compat.v1.disable_eager_execution()

from models.hrnet import HRNet
from models.vggunet import Vggunet
from models.subject4 import Subject4
from models.bisenet import Bisenet

from models.callback import Custom_Callback
from dataparser.inria import Inria, Inria_v
from dataparser.ade20k import Ade20k, Ade20k_v
from dataparser.cityscape import Cityscape, Cityscape_v

#%%


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = yaml.load("".join(Path(args.config).open("r").readlines()), Loader=yaml.FullLoader)
    # config = yaml.load("".join(Path("configs/ade20k_hrnet.yaml").open("r").readlines()), Loader=yaml.FullLoader)

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
<<<<<<< HEAD
    if config["dataset_name"] == "cityscape" :
        data_parser = Cityscape(config)
        data_parserv = Cityscape_v(config)
=======
>>>>>>> 88c8c7dee5e8c4244b53826235183fd203071791

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

    with mirrored_strategy.scope():
        dataset = tf.data.Dataset.from_generator(
            data_parser.generator,
            (tf.float32, tf.float32),
            # (tf.TensorShape([config["image_size"][0], config["image_size"][1], 3]), tf.TensorShape([config["image_size"][0], config["image_size"][1], 3]))
<<<<<<< HEAD
            (tf.TensorShape([None, None, 3]), tf.TensorShape([None, None]))
=======
            (tf.TensorShape([None, None, 3]), tf.TensorShape([None, None, 3]))
>>>>>>> 88c8c7dee5e8c4244b53826235183fd203071791
        ).batch(config["batch_size"], drop_remainder=True)

        datasetv = tf.data.Dataset.from_generator(
            data_parserv.generator,
            (tf.float32, tf.float32),
            # (tf.TensorShape([config["image_size"][0], config["image_size"][1], 3]), tf.TensorShape([config["image_size"][0], config["image_size"][1], 3]))
<<<<<<< HEAD
            (tf.TensorShape([None, None, 3]), tf.TensorShape([None, None]))
        ).batch(config["batch_size"], drop_remainder=True)

=======
            (tf.TensorShape([None, None, 3]), tf.TensorShape([None, None, 3]))
        ).batch(config["batch_size"], drop_remainder=True)


>>>>>>> 88c8c7dee5e8c4244b53826235183fd203071791
        dist_dataset = mirrored_strategy.experimental_distribute_dataset(dataset)
        dist_datasetv = mirrored_strategy.experimental_distribute_dataset(datasetv)

# %%

    @tf.function
    def train_step(dist_inputs) :
        def train_fn(inputs) :
            x, y = inputs
            with tf.GradientTape() as tape :
                output = model.model(x, training=True)
                loss = model.sce_loss(y, output)
            grads = tape.gradient(loss, model.model.trainable_variables)
            model.optim.apply_gradients(list(zip(grads, model.model.trainable_variables)))
            return loss

        per_example_losses = mirrored_strategy.experimental_run_v2(train_fn, args=(dist_inputs,))
        mean_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN, per_example_losses, axis=None)
        return mean_loss

    @tf.function
    def test_step(dist_inputs) :
        def test_fn(inputs) :
            x, y = inputs

            output = model.model(x, training=False)
            accu = model.pixel_accuracy(y, output)
            miou = model.miou(y, output)
            return accu, miou

        pe_accu, pe_miou = mirrored_strategy.experimental_run_v2(test_fn, args=(dist_inputs,))
        mean_accu = mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN, pe_accu, axis=None)
        mean_miou = mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN, pe_miou, axis=None)
        return mean_accu, mean_miou

    def set_lr (e) :

        updated_lr = config["lr"] * ((1/2)**(e // 15))
        # if e > 20 :
        #     updated_lr = config["lr"]/2
        # elif e > 40 :
        #     updated_lr = config["lr"]/4
        # elif e > 60 :
        #     updated_lr = config["lr"]/8
        # elif e > 80 :
        #     updated_lr = config["lr"]/16
        # elif e > 100 :
        #     updated_lr = config["lr"]/32
        # elif e > 120 :
        #     updated_lr = config["lr"]/64
        # else :
        #     updated_lr = config["lr"]
        return updated_lr


    def train () :

        logfile = Path(config["logger_file"])

        if not logfile.exists() :
            tmpf = logfile.open("w+")
            top_text = "epoch,loss,pixel_acc,miou\n"
            tmpf.write(top_text)
            tmpf.close()

        # model.model.summary()

        for e in range(config["present_epoch"], config["epoch"]) :

<<<<<<< HEAD
            # K.set_value(model.optim.learning_rate, set_lr(e))
=======
            K.set_value(model.optim.learning_rate, set_lr(e))
>>>>>>> 88c8c7dee5e8c4244b53826235183fd203071791

            losses = []
            # tqdm_s = tqdm(range(data_parser.steps))
            tqdm_s = tqdm(dist_dataset)
            for s in tqdm_s :
                # mean_loss = train_step(data_parser.get_batch())
                mean_loss = train_step(s)
                losses.append(mean_loss.numpy())
                tqdm_s.set_description_str(f"Loss : {str(np.mean(losses))}")
<<<<<<< HEAD
            print(f"Epoch : {str(e)}, Loss : {str(np.mean(losses))}")
=======
            print(f"Epoch : {str(e)}, Loss : {str(mean_loss.numpy())}")
>>>>>>> 88c8c7dee5e8c4244b53826235183fd203071791

            accus, mious = [], []
            # tqdm_sv = tqdm(range(data_parserv.steps))
            tqdm_sv = tqdm(dist_datasetv)
            for s in tqdm_sv :
                # accu, miou = test_step(data_parserv.get_batch())
                accu, miou = test_step(s)
                accus.append(accu), mious.append(miou)
                tqdm_sv.set_description_str(f"accu : {str(np.mean(accus))}, miou : {str(np.mean(mious))}")
            print(f"Epoch : {str(e)}, Accu : {str(np.mean(accus))}, miou : {str(np.mean(mious))}")
            model.miou_op.reset_states()

            tmpf = logfile.open("a+")
            tmpf.write(",".join([str(e), str(np.mean(losses)), str(np.mean(accus)), str(np.mean(mious))]) + "\n")
            tmpf.close()

            data_parser.on_epoch_end()
            data_parserv.on_epoch_end()

            if not Path(config['save_path']).exists() :
                Path(config['save_path']).mkdir(parents=True)
            # break
            if e % config["saving_interval"] == config["saving_interval"]-1 :
                model.model.save(f"{config['save_path']}/model_{str(e+1)}.h5")


    with mirrored_strategy.scope():
        train()


    # repeat = config["epoch"]*data_parser.steps
    # repeatv = config["epoch"]*data_parserv.steps
<<<<<<< HEAD
    # with mirrored_strategy.scope():
    #     dataset = tf.data.Dataset.from_generator(
    #         data_parser.generator,
    #         (tf.float32, tf.float32),
    #         (tf.TensorShape([config["image_size"][0], config["image_size"][1], 3]), tf.TensorShape([config["image_size"][0], config["image_size"][1], 3]))
    #     ).batch(config["batch_size"], drop_remainder=True)
    #     datasetv = tf.data.Dataset.from_generator(
    #         data_parserv.generator,
    #         (tf.float32, tf.float32),
    #         (tf.TensorShape([config["image_size"][0], config["image_size"][1], 3]), tf.TensorShape([config["image_size"][0], config["image_size"][1], 3]))
    #     ).batch(config["batch_size"], drop_remainder=True)
=======
    # dataset = tf.data.Dataset.from_generator(
    #     data_parser.generator,
    #     (tf.float32, tf.float32),
    #     (tf.TensorShape([config["image_size"][0], config["image_size"][1], 3]), tf.TensorShape([config["image_size"][0], config["image_size"][1], 3]))
    # ).batch(config["batch_size"], drop_remainder=True)
    # datasetv = tf.data.Dataset.from_generator(
    #     data_parserv.generator,
    #     (tf.float32, tf.float32),
    #     (tf.TensorShape([config["image_size"][0], config["image_size"][1], 3]), tf.TensorShape([config["image_size"][0], config["image_size"][1], 3]))
    # ).batch(config["batch_size"], drop_remainder=True)
>>>>>>> 88c8c7dee5e8c4244b53826235183fd203071791

    # logger = tk.callbacks.CSVLogger(config["logger_file"], append=True)
    # model_cb = Custom_Callback(config)

    # # lr_scheduler = tk.callbacks.ReduceLROnPlateau(monitor="val_loss")
    # def lr_sched(epoch) :
    #     if epoch < 50 :
    #         return config["lr"]
    #     elif epoch < 100 :
    #         return 0.5 * config["lr"]
    #     else :
    #         return 0.25 * config["lr"]
    # lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_sched)

<<<<<<< HEAD
    # print(model.model)
=======
    #     print(model.model)
>>>>>>> 88c8c7dee5e8c4244b53826235183fd203071791

    # print("model loaded")

    # # print(data_parser.steps)
    # model.model.fit(
    #     dataset,
    #     epochs=config["epoch"],
    #     callbacks=[model_cb, logger, lr_scheduler],
    #     # callbacks=[model_cb, logger],
    #     validation_data=datasetv,
    #     validation_freq=1,
    #     # steps_per_epoch=data_parser.steps,
    #     # class_weight=config["class_weight"],
    #     initial_epoch=config["present_epoch"]
    #     )

    # model.model.save(str(Path(config["save_path"])/f"model_{config['epoch']}.h5"))


# %%
if False :

    tmpx, tmpy = data_parser.get_batch()

    with tf.GradientTape() as tape :
        output = model.model(tmpx[:4, :, :, :], training=True)
        loss = model.sce_loss(tmpy[:4, :, :, :], output)
    grads = tape.gradient(loss, model.model.trainable_variables)

    grads


    # %%

    for v in grads :
        print(v.shape)
