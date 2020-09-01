#%%
import numpy as np
from pathlib import Path
from tqdm import tqdm
import tensorflow as tf
from PIL import Image

import albumentations as album

import cv2

import utils.util as util

class Ade20k () :

    def __init__ (self, configs : dict) :

        self.image_list = []
        self.mask_list = []
        self.index_list = []
        self.loaded_in_memory = False

        self.aug = True
        self.x_path, self.y_path, root_path = self.x_y_root_paths(configs)

        for i in range(len(self.x_path)) :
            # v = Path(self.x_path[i].strip().replace("outsourced_dataset", "open_dataset"))
            # vv = Path(self.y_path[i].strip().replace("outsourced_dataset", "open_dataset"))
            v = Path(self.x_path[i].strip())
            vv = Path(self.y_path[i].strip())

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
    
    def random_crop (self, img, mask) :
        mh = img.shape[0] - min(img.shape[0], img.shape[1])
        mw = img.shape[1] - min(img.shape[0], img.shape[1])
        msize = min(img.shape[0], img.shape[1])
        rmh = np.random.randint(0, mh//2)
        rmw = np.random.randint(0, mw//2)
        img = img[rmh:rmh+msize, rmw:rmw+msize, :]
        mask = mask[rmh:rmh+msize, rmw:rmw+msize, :]
        return img, mask
    
    def random_flip (self, img, mask) :

        if np.random.random() > 0.5 :
            img = img[::-1, :, :]
            mask = mask[::-1, :, :]
        elif np.random.random() > 0.5 :
            img = img[:, ::-1, :]
            mask = mask[:, ::-1, :]

        return img, mask

    def norm (self, img, mask) :
        return img/255, mask

    def get_one_set (self, i) :

        imgi = self.index_list[i]

        img = Image.open(self.image_list[imgi]).convert("RGB")
        mask = Image.open(self.mask_list[imgi]).convert("RGB")

        img, mask = self.additional_op(img, mask)

        img, mask = self.center_crop(img, mask)

        # if img.shape != (self.configs["image_size"][0], self.configs["image_size"][1], 3) or mask.shape != (self.configs["image_size"][0], self.configs["image_size"][1], 3) :
            # print(img.shape)
            # exit()
            # img, mask = self.get_one_set(np.random.choice(self.index_list))

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
        x, y = self.random_flip(x, y)
        x, y = self.norm(x, y)
        img_sz = (self.configs["image_size"][0], self.configs["image_size"][1])
        x, y = cv2.resize(x, img_sz, img_sz, interpolation=cv2.INTER_LINEAR), cv2.resize(y, img_sz, img_sz, interpolation=cv2.INTER_NEAREST)

        a_image = (augmented["image"]).astype(np.float32)
        # a_image = a_image/255 - np.array([0.40760392, 0.45795686, 0.48501961])
        a_image = a_image/255
        a_mask = (augmented["mask"]).astype(np.int32)

        return a_image, a_mask

    def on_epoch_end (self) :

        if self.shuffle :
            np.random.shuffle(self.index_list)
        self.offset = 0

class Ade20k_v (Ade20k) :
    
    def x_y_root_paths(self, configs):
        self.aug = False
        return (Path(configs["valid_image_path"]).open("r").readlines(), 
                Path(configs["valid_mask_path"]).open("r").readlines(), 
                Path(configs["valid_image_path"]).parent)



# %%

if False :

    # %%
    import yaml

    tmp = "/dy/configs/ade20k_hrnet.yaml"
    config = yaml.load("".join(Path(tmp).open("r").readlines()), Loader=yaml.FullLoader)

    ade20k = Ade20k(config)
    ade20kv = Ade20k_v(config)


    # %%
    # tmpgen = ade20k.generator()
    tmpgen = ade20kv.generator()

    len(ade20kv.image_list)

    # %%

    datasetv = tf.data.Dataset.from_generator(
        ade20kv.generator,
        (tf.float32, tf.float32),
        (tf.TensorShape([config["image_size"][0], config["image_size"][1], 3]), tf.TensorShape([config["image_size"][0], config["image_size"][1], 3]))
    ).batch(config["batch_size"], drop_remainder=True)
    dataset = tf.data.Dataset.from_generator(
        ade20k.generator,
        (tf.float32, tf.float32),
        (tf.TensorShape([config["image_size"][0], config["image_size"][1], 3]), tf.TensorShape([config["image_size"][0], config["image_size"][1], 3]))
    ).batch(config["batch_size"], drop_remainder=True)

    # %%

    next(datasetv)

    # %%

    for i in range(2001) :
        a, b = next(tmpgen)
        if a.shape != (480, 480, 3) :
            print(a.shape)
        if b.shape != (480, 480, 3) :
            print(b.shape)

        break

    # %%

    for v in dataset.as_numpy_iterator() :
        print(v[0].shape)

    # %%


    tmp = a + np.array([0.40760392, 0.45795686, 0.48501961])

    tmpgt = (np.clip(b, 0, 255) == 6).astype(np.uint8)*255
    Image.fromarray(tmpgt)
    
    Image.fromarray(np.clip(tmp*255, 0, 255).astype(np.uint8))

    # %%



    # %%


    from pathlib import Path

    tmp1 = Path("/datasets/outsourced_dataset/ade20k/ADEChallengeData2016/images")
    tmp2 = Path("/datasets/outsourced_dataset/ade20k/ADEChallengeData2016/annotations")
    tmp3 = Path("/datasets/outsourced_dataset/ade20k/ADEChallengeData2016")

    # %%
    training_datasets = []
    training_labels = []
    for v in (tmp1/"training").iterdir() :
        training_datasets.append(str(v))
        training_labels.append(str(v).replace("images", "annotations").replace(".jpg", ".png"))
    # for v in (tmp2/"training").iterdir() :

    valid_datasets = []
    valid_labels = []
    for v in (tmp1/"validation").iterdir() :
        valid_datasets.append(str(v))
        valid_labels.append(str(v).replace("images", "annotations").replace(".jpg", ".png"))
    # for v in (tmp2/"validation").iterdir() :

    # %%

    outout_tr = "\n".join(training_datasets)
    open(str(tmp3/"train_images.txt"), "w+").write(outout_tr)
    outout_tr2 = "\n".join(training_labels)
    open(str(tmp3/"train_gts.txt"), "w+").write(outout_tr2)
    outout_tr3 = "\n".join(valid_datasets)
    open(str(tmp3/"valid_images.txt"), "w+").write(outout_tr3)
    outout_tr4 = "\n".join(valid_labels)
    open(str(tmp3/"valid_gts.txt"), "w+").write(outout_tr4)


    # %%

    from scipy import io

    mat_file = io.loadmat("color150.mat")

    print(mat_file["colors"])

# %%
    mat_file2 = io.loadmat("/datasets/outsourced_dataset/ade20k/ADE20K_2016_07_26/index_ade20k.mat")
    mat_file2


# %%
    mat_file2.keys()
    mat_file2["index"]

# %%

    from PIL import Image
    tmp_img1 = Image.open(str(tmp1/"validation"/"ADE_val_00001311.jpg"))
    tmp_img2 = Image.open("/datasets/outsourced_dataset/ade20k/ADEChallengeData2016/annotations/validation/ADE_val_00001311.png")

# %%
    import numpy as np

    Image.fromarray(((np.asarray(tmp_img2) == 20)*255).astype(np.uint8))
    # (tmp_img2 == 20)
