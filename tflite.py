from pathlib import Path
import os

import tensorflow as tf
from tensorflow import keras as tk
import yaml

from models.bisenet import Bisenet


if __name__ == "__main__":

    config = yaml.load("".join(Path("./configs/ade20k_bisenet.yaml").open("r").readlines()), Loader=yaml.FullLoader)

    print("=====================config=====================")
    for v in config.keys() :
        print("%s : %s" %(v, config[v]))
    print("================================================")

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in config["gpu_indices"]])

    config["batch_size"] = 1

    if not config["mode"] == 2 :
        print("Config mode is not for testing!")
        quit()

    tf.keras.backend.set_learning_phase(0)
    the_model = Bisenet(configs=config)


    saved_model_pb = Path("./weights")/"ade20k_bisenet"/"pbfiles"

    @tf.function()
    def make_signature(inputinput):
        # inputinput = tf.reshape(inputinput, [1, 256, 256, 3])//255
        prediction = the_model.model(inputinput)
        # prediction_sm = tk.layers.Activation("softmax")(prediction)
        prediction_sm = tf.argmax(prediction, axis=-1)
        # prediction_sm = tf.cast(prediction_sm, tf.float32)
        # prediction_sm = tf.cast(tf.reshape(prediction_sm, [256 * 256])*-16777216/150, tf.int32)
        prediction_sm = tf.cast(prediction_sm, tf.int32)
        return {"output": prediction_sm}

    new_signatures = make_signature.get_concrete_function(
        inputinput=tf.TensorSpec([None, config["image_size"][0], config["image_size"][0], 3], dtype=tf.dtypes.float32, name="input")
    )

    tf.saved_model.save(the_model.model, export_dir=str(saved_model_pb), signatures={"predict":new_signatures})


    saved_model_dir = Path("./weights")/"ade20k_bisenet"/"pbfiles"

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(0) # running the demo server in gpu 1

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir=str(saved_model_dir))
    tflite_model = converter.convert()

    saving_model_dir = Path("./weights")/"tflite"/"ade20k_bisenet.tflite"
    open(str(saving_model_dir), "wb").write(tflite_model)
