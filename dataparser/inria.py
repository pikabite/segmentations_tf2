#%%
import numpy as np
from pathlib import Path
from tqdm import tqdm
import tensorflow as tf
from PIL import Image

import albumentations as album

import utils.util as util

READ_CROPPED_IMAGES = True

class Inria () :
    
    def __init__ (self, configs : dict) :

        self.x_path, self.y_path, root_path = self.x_y_root_paths(configs)

        self.image_list = []
        self.mask_list = []
        self.index_list = []
        ii = 0
        self.loaded_in_memory = False

        self.image_size = configs["image_size"]

        # cropped image per image
        self.cpi = 1521
        self.sqrt_cpi = int(np.sqrt(self.cpi))
        self.img_full_size = [5000, 5000, 3]

        self.albu = album.Compose([
            album.VerticalFlip(),
            album.HorizontalFlip(),
            album.RGBShift(),
            album.RandomBrightness(),
            album.RandomResizedCrop(height=self.image_size[0], width=self.image_size[1], scale=(0.7, 0.7))
        ], p=0.8)
        
        for i in range(len(self.x_path)) :
            v = root_path/self.x_path[i].strip()
            vv = root_path/self.y_path[i].strip()

            assert v.exists() and vv.exists(), "dataset is not correct"

            self.image_list.append(v)
            self.mask_list.append(vv)
            for iii in range(self.cpi) :
                self.index_list.append(ii*self.cpi+iii)
            ii += 1
        self.tot_len = ii

        self.shuffle = True
        # if self.shuffle :
        #     np.random.shuffle(self.index_list)

        self.configs = configs

        if not READ_CROPPED_IMAGES :
            self.load_all_in_memory()

    @property
    def steps (self) :
        # return self.x_gen.__len__() - 1
        return int(len(self.image_list)/self.configs["batch_size"])
    
    def load_all_in_memory (self) :

        tmp_img_list = []
        tmp_mask_list = []

        for i in tqdm(range(len(self.image_list))) :
            img = np.asarray(Image.open(str(self.image_list[i])).convert("RGB"), dtype=np.uint8)
            mask = np.asarray(Image.open(str(self.mask_list[i])).convert("L"), dtype=np.uint8)
            mask = np.expand_dims(mask, axis=-1)
            
            tmp_img_list.append(img)
            tmp_mask_list.append(mask)
        
        self.image_list = tmp_img_list
        self.mask_list = tmp_mask_list

    def get_one_set (self, i) :

        imgi = self.index_list[i]//self.cpi
        cropi = self.index_list[i]%self.cpi

        # print(self.index_list[i])
        # print(imgi)
        # print(cropi)

        if READ_CROPPED_IMAGES :
            cropped_img_path = str(self.image_list[imgi]).replace("/train/", "/train_cropped/").replace(".tif", "_" + str(cropi) + ".png")
            cropped_mask_path = str(self.mask_list[imgi]).replace("/train/", "/train_cropped/").replace(".tif", "_" + str(cropi) + ".png")
            img = np.asarray(Image.open(cropped_img_path).convert("RGB"), dtype=np.uint8)
            mask = np.asarray(Image.open(cropped_mask_path).convert("L"), dtype=np.uint8)
            mask = np.expand_dims(mask, axis=-1)
        else :
            img = self.image_list[imgi]
            mask = self.mask_list[imgi]

        img, mask = self.additional_op(img, mask, crop_index=cropi)

        if img.shape != (self.configs["image_size"][0], self.configs["image_size"][1], 3) or mask.shape != (self.configs["image_size"][0], self.configs["image_size"][1], 1) :
            # print(img.shape)
            # print("passed!?")
            img, mask = self.get_one_set(np.random.choice(self.index_list))

        return img, mask


    def generator (self) :

        for i in range(len(self.index_list)) :

            img, mask = self.get_one_set(i)

            yield (img, mask)

        self.on_epoch_end()

    def x_y_root_paths (self, configs) :
        return (Path(configs["train_image_path"]).open("r").readlines(), 
                Path(configs["train_mask_path"]).open("r").readlines(), 
                Path(configs["train_image_path"]).parent)

    def additional_op (self, x, y, crop_index) :

        # x, y = self.random_crop_inria(x, y, crop_index)

        augmented = self.albu(image=x, mask=y)

        a_image = (augmented["image"]).astype(np.float32)/255
        a_mask = (augmented["mask"]).astype(np.float32)

        return a_image, a_mask

        # options = {
        #     "hflip" : True,
        #     "vflip" : True,
        #     "brightness_range" : [0.2, 1.2],
        #     "noise" : True,
        #     "scale" : True
        # }
        # return util.preprocessing(x, y, options)
    
    def random_crop_inria (self, x, y, crop_index) :

        h_ran_start = np.random.randint(0, self.img_full_size[0] - self.configs["image_size"][0])
        w_ran_start = np.random.randint(0, self.img_full_size[1] - self.configs["image_size"][1])

        x = x[h_ran_start:h_ran_start+self.configs["image_size"][0], w_ran_start:w_ran_start+self.configs["image_size"][1], :]
        y = y[h_ran_start:h_ran_start+self.configs["image_size"][0], w_ran_start:w_ran_start+self.configs["image_size"][1], :]


        return x, y
    
    def crop_index_inria (self, x, y, crop_index) :

        h_indx = crop_index%self.sqrt_cpi
        w_indx = crop_index//self.sqrt_cpi

        x = x[h_indx:h_indx+self.configs["image_size"][0], w_indx:w_indx+self.configs["image_size"][1], :]
        y = y[h_indx:h_indx+self.configs["image_size"][0], w_indx:w_indx+self.configs["image_size"][1], :]

        return x, y

    def on_epoch_end (self) :

        # self.x_gen.on_epoch_end()
        # self.y_gen.on_epoch_end()
        if self.shuffle :
            # print("shuffled!")
            np.random.shuffle(self.index_list)


class Inria_v (Inria) :
    
    def x_y_root_paths(self, configs):
        self.shuffle = False
        return (Path(configs["valid_image_path"]).open("r").readlines(), 
                Path(configs["valid_mask_path"]).open("r").readlines(), 
                Path(configs["valid_image_path"]).parent)

    def additional_op(self, x, y, crop_index):

        # x, y = self.crop_index_inria(x, y, crop_index)
        # options = {
        #     "scale" : True
        # }
        # return util.preprocessing(x, y, options)

        non_a_image = (x).astype(np.float32)/255
        non_a_mask = y.astype(np.float32)
        return non_a_image, non_a_mask



# %%

# import yaml

# tmp = "configs/inria_subject5.yaml"
# config = yaml.load("".join(Path(tmp).open("r").readlines()), Loader=yaml.FullLoader)

# inria = Inria(config)
# inriav = Inria_v(config)


# #%%

# tmpgen = inria.generator()

# len(inria.image_list)*inria.cpi

# #%%

# ii = 0
# for i in range(len(inria.index_list)) :

#     # len(inria.index_list)*inria.cpi/120

#     a, b = next(tmpgen)

#     print(ii)
#     ii += 1


# %%

# Image.fromarray(np.clip(a*255, 0, 255).astype(np.uint8))
# Image.fromarray(np.clip(b*255, 0, 255)[:, :, 0].astype(np.uint8))

#%%
