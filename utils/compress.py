
from skimage import io
from PIL import Image
from torchvision.transforms import transforms
import torch
import numpy as np
import pdb

import imgaug.augmenters as iaa


def compress_image_skimage(input_tensor,quality=75):
    ## input_tensor shape  n * c w h
    n,c,w,h = input_tensor.shape
    #try:
    images = [transforms.ToPILImage()(input_tensor[i]) for i in range(n)]
    #except:
    comp_arr_list=[]
    for image in images:
        new_file_name="test.jpeg"
        io.imsave(new_file_name,np.array(image),quality=quality)
        input_comp_arr = io.imread(new_file_name)
        pi = Image.fromarray(input_comp_arr)

        to_tensor_transform = transforms.Compose([transforms.ToTensor()])
        tensor_compress = to_tensor_transform(pi).reshape([1, c, w, h])
        comp_arr_list.append(tensor_compress)

    comp_tensor = torch.cat(comp_arr_list,dim=0)

    return comp_tensor
def compress_image_iaa(input_tensor,quality=70):
    ## input_tensor shape  n * c w h
    n,c,w,h = input_tensor.shape
    images = [transforms.ToPILImage()(input_tensor[i]) for i in range(n)]
    comp_arr_list=[]
    compress_transform = iaa.JpegCompression(compression=(100 - quality, 100 - quality))
    to_tensor_transform = transforms.Compose([transforms.ToTensor()])
    for image in images:
        image_comp = compress_transform.augment_images([np.array(image)])[0]
        pi = Image.fromarray(image_comp)

        tensor_compress = to_tensor_transform(pi).reshape([1, c, w, h])
        comp_arr_list.append(tensor_compress)

    comp_tensor = torch.cat(comp_arr_list,dim=0)

    return comp_tensor
