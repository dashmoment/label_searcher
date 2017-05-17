import tensorflow as tf
import utility as ut
import numpy as np
import os
import pandas as pd
import net_factory as nf





var_list = [
            ['conv1w',[3,3,6,16]],
            ['conv1b',[16]],
            ['conv2w',[3,3,16,32]],
            ['conv2b',[32]],
            ['conv3w',[3,3,32,64]],
            ['conv3b',[64]],
            ['conv4w',[3,3,64,128]],
            ['conv4b',[128]],
        
            ['fc10w',[4608,256]],
            ['fc10b',[256]],
            ['fc11w',[256,128]],
            ['fc11b',[128]],
            ['fc12w',[128,2]],
            ['fc12b',[2]]]


def list_image_sets():
    """
    List all the image sets from Pascal VOC. Don't bother computing
    this on the fly, just remember it. It's faster.
    """
    return [
        'aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train',
        'tvmonitor']
    
    
batch_file = "/home/dashmoment/workspace/dataset/VOCdevkit/VOC2012/JPEGImages"
label_file = "/home/dashmoment/workspace/dataset/VOCdevkit/VOC2012/ImageSets/Main"

class_list = list_image_sets()

def create_label_set(class_list, label_file, ftype):
    
    image_dict = {}
    for cls in class_list:
    
        filename = os.path.join(label_file, cls+'_'+ftype+'.txt')    
        
        df = pd.read_csv(
                filename,
                delim_whitespace=True,
                header=None,
                dtype={0: str},
                names=['filename', 'true'])   
            
        df = df[df['true'] == 1]   
        filelist = list(df['filename'].values) 
        image_dict[cls] = filelist
        
    return image_dict
    

image_dict = create_label_set(class_list, label_file, 'train')

for i in range(len(image_dict['aeroplane'])-1):
    
    path = image_dict['aeroplane'][i]
    src = os.path.join(batch_file, path+'.jpg')
    path = image_dict['aeroplane'][i +1]
    src2 = os.path.join(batch_file, path+'.jpg')



batchsize = 1

inputs = tf.placeholder(tf.float32, (None, 448, 448, 6), name='input')
label = tf.placeholder(tf.float32, (None, 2), name = 'label')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

scopename = 'variable'
var_dict = nf.create_variable(scopename, var_list)
logit = nf.model_vanilla('vanilla', scopename, var_dict, inputs, var_dict[scopename], keep_prob)


#src = ut.image_random_batch(batch_file, batchsize, (448,448,3), np.float32)







































