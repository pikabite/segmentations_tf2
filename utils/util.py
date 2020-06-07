#%%
import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow import keras as tk

#%%

def preprocessing (image, mask, options) :

    if "hflip" in options and options["hflip"] :
        if np.random.randint(0, 2) == 1 :
            image = image[::-1, :, :]
            mask = mask[::-1, :, :]
    if "vflip" in options and options["vflip"] :
        if np.random.randint(0, 2) == 1 :
            image = image[:, ::-1, :]
            mask = mask[:, ::-1, :]
    if "brightness_range" in options and options["brightness_range"] is not None :
        assert len(options["brightness_range"]) == 2, "brightness should be formed [min, max]"
        b_number = np.random.uniform(options["brightness_range"][0], options["brightness_range"][1])
        image = image*b_number
    if "scale" in options and options["scale"] :
        image /= 255
    if "noise" in options and options["noise"] :
        # print("noise ahn hae tta")
        pass

    # print(np.sum((mask == 255)*1))

    return image, mask


#%%

def calculate_mask_iou (masks1, masks2, class_number) :
    # tmp1 = (mask1 == class_number)*1
    # tmp2 = (mask2 == class_number)*1

    mask_sum = masks1 + masks2
    i = np.sum((mask_sum == class_number*2)*1, axis=(1, 2, 3))
    u = np.sum((mask_sum >= class_number)*1, axis=(1, 2, 3))

    return np.divide(i, u, where=u!=0)
    # return np.clip(i / u, 0, 1.0)



#%%

