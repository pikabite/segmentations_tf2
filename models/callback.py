import tensorflow as tf
from tensorflow import keras as tk
from tensorflow.keras.callbacks import Callback

from pathlib import Path

class Custom_Callback (Callback):

    def __init__(self, configs):
        self.configs = configs
        self.saving_file = str(Path(self.configs["save_path"])/"model.h5")
        self.best_iou = 0
        self.save_best = configs["save_best"]

        if not Path(self.configs["save_path"]).is_dir() :
            Path(self.configs["save_path"]).mkdir(parents=True)

        super().__init__()

    def on_epoch_end (self, epoch, logs={}):
        if epoch%self.configs["saving_interval"] == 0 :
            self.model.save(self.saving_file.replace("model.h5", "model_%s.h5"%(str(epoch))))
        
        if self.best_iou < logs["moiu"] and self.save_best : 
            print("Save Best!")
            self.model.save(self.saving_file.replace("model.h5", "best.h5"))
            self.best_iou = logs["moiu"]
