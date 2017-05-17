import numpy as np
from random import shuffle
import os
import cv2


def image_random_batch(dirname, batchsize, imagesize, array_type):
    
    cwd = os.getcwd()
    os.chdir(dirname)
    
    files = [x for x in os.listdir(dirname)]
    idx = [x for x in range(len(files))]
    shuffle(idx)
    
    batch = [files[b] for b in idx[0:batchsize]]
    batch_images = []

    for fname in batch:
        img = cv2.imread(fname)
        img_resized = cv2.resize(img, (imagesize[0],imagesize[1]))
        img_RGB = cv2.cvtColor(img_resized,cv2.COLOR_BGR2RGB)
        img_resized_np = np.asarray( img_RGB )
        inputs = (img_resized_np/255.0)*2.0-1.0

        batch_images.append(np.array(inputs))
    
    batch_images = np.stack(batch_images)
    
    os.chdir(cwd)
    
    return batch_images