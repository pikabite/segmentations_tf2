#%%
import numpy as np
from pathlib import Path
from tqdm import tqdm
import tensorflow as tf
from PIL import Image

import albumentations as album

import utils.util as util

class Ade20k () :

    def __init__ (self, configs : dict) :

        self.image_list = []
        self.mask_list = []
        self.index_list = []
        self.loaded_in_memory = False

        self.albu = album.Compose([
            album.VerticalFlip(),
            album.HorizontalFlip(),
            album.RGBShift(),
            album.RandomBrightness(),
            # album.Resize(configs["image_size"][0], configs["image_size"][1], always_apply=True, p=1),
            album.RandomResizedCrop(height=configs["image_size"][0], width=configs["image_size"][1], scale=(0.7, 0.7)),
            album.Normalize(mean=(0.40760392, 0.45795686, 0.48501961))
            # album.CenterCrop(configs["image_size"][0], configs["image_size"][1], always_apply=True, p=1),
        ], p=0.5)

        self.x_path, self.y_path, root_path = self.x_y_root_paths(configs)

        for i in range(len(self.x_path)) :
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

    @property
    def steps (self) :
        # return self.x_gen.__len__() - 1
        return int(len(self.image_list)/self.configs["batch_size"])
    
    def get_one_set (self, i) :

        imgi = self.index_list[i]

        img = np.asarray(Image.open(self.image_list[imgi]).convert("RGB"))
        mask = np.asarray(Image.open(self.mask_list[imgi]).convert("RGB"))

        mh = img.shape[0] - min(img.shape[0], img.shape[1])
        mw = img.shape[1] - min(img.shape[0], img.shape[1])
        msize = min(img.shape[0], img.shape[1])
        img = img[mh//2:mh//2+msize, mw//2:mw//2+msize, :]
        mask = mask[mh//2:mh//2+msize, mw//2:mw//2+msize, :]

        img, mask = self.additional_op(img, mask)

        if img.shape != (self.configs["image_size"][0], self.configs["image_size"][1], 3) or mask.shape != (self.configs["image_size"][0], self.configs["image_size"][1], 3) :
            # print(img.shape)
            # exit()
            img, mask = self.get_one_set(np.random.choice(self.index_list))

        return img, mask


    def generator (self) :

        for i in range(len(self.index_list)) :
            img, mask = self.get_one_set(i)
            yield img, mask

        self.on_epoch_end()

    def x_y_root_paths (self, configs) :
        return (Path(configs["train_image_path"]).open("r").readlines(), 
                Path(configs["train_mask_path"]).open("r").readlines(), 
                Path(configs["train_image_path"]).parent)

    def additional_op (self, x, y) :

        augmented = self.albu(image=x, mask=y)

        a_image = (augmented["image"]).astype(np.float32)/255
        a_mask = (augmented["mask"]).astype(np.float32)

        return a_image, a_mask

    def on_epoch_end (self) :

        if self.shuffle :
            np.random.shuffle(self.index_list)


class Ade20k_v (Ade20k) :
    
    def x_y_root_paths(self, configs):

        self.albu = album.Compose([
            album.Resize(configs["image_size"][0], configs["image_size"][1], always_apply=True, p=1),
        ], p=1.0)

        return (Path(configs["valid_image_path"]).open("r").readlines(), 
                Path(configs["valid_mask_path"]).open("r").readlines(), 
                Path(configs["valid_image_path"]).parent)



# %%

if False :

    # %%
    import yaml

    tmp = "configs/ade20k_bisenet.yaml"
    config = yaml.load("".join(Path(tmp).open("r").readlines()), Loader=yaml.FullLoader)

    ade20k = Ade20k(config)
    ade20kv = Ade20k_v(config)


    # %%
    tmpgen = ade20k.generator()
    tmpgen = ade20kv.generator()

    len(ade20k.image_list)

    # %%

    dataset = tf.data.Dataset.from_generator(
        ade20k.generator,
        (tf.float32, tf.float32),
        (tf.TensorShape([config["image_size"][0], config["image_size"][1], 3]), tf.TensorShape([config["image_size"][0], config["image_size"][1], 3]))
    ).batch(config["batch_size"], drop_remainder=True)

    # %%

    next(dataset)

    # %%

    for i in range(20000) :
        a, b = next(tmpgen)
        if a.shape != (256, 256, 3) :
            print(a.shape)
    # %%

    a, b = next(tmpgen)

    # %%

    Image.fromarray(np.clip(a*255, 0, 255).astype(np.uint8))
    Image.fromarray(np.clip(b, 0, 255)[:, :, :].astype(np.uint8))

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
    