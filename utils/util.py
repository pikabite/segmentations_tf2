#%%
import numpy as np
from PIL import Image


#%%

def label_to_color (imgs, colormap) :

    newimg = np.zeros((imgs.shape[0], imgs.shape[1], imgs.shape[2]))
    imgs = np.mean(imgs, axis=-1)

    for i in range(len(colormap)) :
        newimg[imgs == i] = colormap[i]
    
    return newimg.astype(np.uint8)


# %%
def unnorm (imgs, mean, std) :
    return (((imgs * std) + mean)*255).astype(np.uint8)