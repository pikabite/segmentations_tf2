#%%
import numpy as np
from pathlib import Path
from tqdm import tqdm
import tensorflow as tf
from PIL import Image

import albumentations as album

import cv2

# import utils.util as util

# %%

class Cityscape () :

    def __init__ (self, configs : dict) :

        self.image_list = []
        self.mask_list = []
        self.index_list = []

        self.aug = True
        self.x_path, self.y_path, root_path = self.x_y_root_paths(configs)

        for i in range(len(self.x_path)) :
            v = Path(self.x_path[i].strip())
            vv = Path(self.y_path[i].strip())

            v = Path(str(v).replace("/datasets/outsourced_dataset/cityscapes", "/datasets/hdd/dataset/cityscapes"))
            vv = Path(str(vv).replace("/datasets/outsourced_dataset/cityscapes", "/datasets/hdd/dataset/cityscapes"))

            assert v.exists() and vv.exists(), "data is wrong"

            self.image_list.append(v)
            self.mask_list.append(vv)
            self.index_list.append(i)
        self.tot_len = i

        self.shuffle = True
        if self.shuffle :
            np.random.shuffle(self.index_list)

        self.configs = configs

        self.offset = 0

        ignore_label = 255
        self.label_mapping = {0: ignore_label, 
                              1: ignore_label, 2: ignore_label, 
                              3: ignore_label, 4: ignore_label, 
                              5: ignore_label, 6: ignore_label, 
                              7: 0, 8: 1, 9: ignore_label, 
                              10: ignore_label, 11: 2, 12: 3, 
                              13: 4, 14: ignore_label, 15: ignore_label, 
                              16: ignore_label, 17: 5, 18: ignore_label, 
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
                              25: 12, 26: 13, 27: 14, 28: 15, 
                              29: ignore_label, 30: ignore_label, 
                              31: 16, 32: 17, 33: 18}
        self.class_weights = np.array([0.8373, 0.918, 0.866, 1.0345, 
                                        1.0166, 0.9969, 0.9754, 1.0489,
                                        0.8786, 1.0023, 0.9539, 0.9843, 
                                        1.1116, 0.9037, 1.0865, 1.0955, 
                                        1.0865, 1.1529, 1.0507])
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])


    def convert_label(self, label, inverse=False):
        temp = label.copy()
        if inverse:
            for v, k in self.label_mapping.items():
                label[temp == k] = v
        else:
            for k, v in self.label_mapping.items():
                label[temp == k] = v
        return label
    
    @property
    def steps (self) :
        # return self.x_gen.__len__() - 1
        return int(len(self.image_list)/self.configs["batch_size"])
    
    def center_crop (self, img, mask) :
        mh = img.shape[0] - min(img.shape[0], img.shape[1])
        mw = img.shape[1] - min(img.shape[0], img.shape[1])
        msize = min(img.shape[0], img.shape[1])
        img = img[mh//2:mh//2+msize, mw//2:mw//2+msize, :]
        mask = mask[mh//2:mh//2+msize, mw//2:mw//2+msize, :]
        return img, mask
    
    def random_crop (self, img, mask, crop_size) :
        mh = img.shape[0] - crop_size[0]
        mw = img.shape[1] - crop_size[1]
        if mh == 0 and mw == 0 : return img, mask
        rmh = np.random.randint(0, mh)
        rmw = np.random.randint(0, mw)
        img = img[rmh:rmh+crop_size[0], rmw:rmw+crop_size[1], :]
        mask = mask[rmh:rmh+crop_size[0], rmw:rmw+crop_size[1], :]
        return img, mask
    
    def random_flip (self, img, mask) :

        # if np.random.random() > 0.5 :
        #     img = img[::-1, :, :]
        #     mask = mask[::-1, :, :]
        if np.random.random() > 0.5 :
            img = img[:, ::-1, :]
            mask = mask[:, ::-1, :]

        return img, mask

    def norm (self, img, mask) :

        img = img / 255.0
        img = img - self.mean
        img = img / self.std

        return img, mask

    def get_one_set (self, i) :

        imgi = self.index_list[i]

        img = Image.open(self.image_list[imgi]).convert("RGB")
        mask = Image.open(self.mask_list[imgi]).convert("RGB")

        img, mask = self.additional_op(img, mask)

        return img, mask


    def generator (self) :

        for i in range(len(self.index_list)) :
            img, mask = self.get_one_set(i)
            yield img, mask

        self.on_epoch_end()

    def get_batch (self) :

        xs, ys = [], []

        for b in range(self.configs["batch_size"]) :
            x, y = self.get_one_set(self.offset + b)
            xs.append(x), ys.append(y)

        self.offset += self.configs["batch_size"]
        return np.array(xs), np.array(ys)

    def x_y_root_paths (self, configs) :
        return (Path(configs["train_image_path"]).open("r").readlines(), 
                Path(configs["train_mask_path"]).open("r").readlines(), 
                Path(configs["train_image_path"]).parent)

    def additional_op (self, x, y) :

        x, y = np.asarray(x), np.asarray(y)
        if self.aug :
            x, y = self.random_flip(x, y)
        
        x, y = self.random_crop(x, y, self.configs["image_size"])
        img_sz = (self.configs["image_size"][1], self.configs["image_size"][0])
        x, y = cv2.resize(x, img_sz, img_sz, interpolation=cv2.INTER_LINEAR), cv2.resize(y, img_sz, img_sz, interpolation=cv2.INTER_NEAREST)
        x, y = self.norm(x, y)
        y = self.convert_label(y)[:, :, 0]

        return x.astype(np.float32), y.astype(np.float32)

    def on_epoch_end (self) :

        if self.shuffle :
            np.random.shuffle(self.index_list)
        self.offset = 0

class Cityscape_v (Cityscape) :
    
    def x_y_root_paths(self, configs):
        self.aug = False
        return (Path(configs["valid_image_path"]).open("r").readlines(), 
                Path(configs["valid_mask_path"]).open("r").readlines(), 
                Path(configs["valid_image_path"]).parent)



# %%

if False :

    # %%
    import yaml

    tmp = "/dy/configs/cityscape_hrnet.yaml"
    config = yaml.load("".join(Path(tmp).open("r").readlines()), Loader=yaml.FullLoader)

    cityscape = Cityscape(config)
    cityscapev = Cityscape_v(config)


    # %%
    # tmpgen = ade20k.generator()
    # tmpgen = cityscape.generator()
    tmpgen = cityscapev.generator()

    tmptmp = np.zeros((34))

    for v in tmpgen :
        a, b = v

        print(a.shape)
        print(b.shape)
        print(b)

        break
        for i in range(34) :
            tmptmp[i] += np.sum((b == i)*1)
        

    # %%

    tmptmp

    # %%

    datasetv = tf.data.Dataset.from_generator(
        cityscapev.generator,
        (tf.float32, tf.float32),
        (tf.TensorShape([config["image_size"][0], config["image_size"][1], 3]), tf.TensorShape([config["image_size"][0], config["image_size"][1], 3]))
    ).batch(config["batch_size"], drop_remainder=True)
    dataset = tf.data.Dataset.from_generator(
        cityscape.generator,
        (tf.float32, tf.float32),
        (tf.TensorShape([config["image_size"][0], config["image_size"][1], 3]), tf.TensorShape([config["image_size"][0], config["image_size"][1], 3]))
    ).batch(config["batch_size"], drop_remainder=True)

# %%

    a, b = cityscape.get_one_set(0)

# %%

    # Image.fromarray((a*255).astype(np.uint8))
    print(b.shape)
    Image.fromarray((b*10).astype(np.uint8))

# %%
