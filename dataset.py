import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import utils
import cv2 
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter
from skimage import exposure

ALLMASKTYPES = ['single_bbox', 'bbox', 'free_form']

def read_image(filePath):
    # Depending on the extension
    # Shape : h, w
    if filePath.endswith(".dcm"):
        image = sitk.ReadImage(filePath)
        image = sitk.GetArrayFromImage(image).astype("int16")
        image = image[0,:,:]

    elif filePath.endswith(".png"):
        image = cv2.imread(filePath, 0)
        image = np.array(image, dtype = "int16")

    elif filePath.endswith(".npy"):
        image = np.load(filePath).astype("int16")
        image = np.flipud(image)
        
    return image


class InpaintDataset(Dataset):
    def __init__(self, opt):
        assert opt.mask_type in ALLMASKTYPES
        self.opt = opt
        self.namelist = utils.get_names(opt.baseroot)

    def __len__(self):  
        return len(self.namelist)

    def __getitem__(self, index):
        # image
        imgname = self.namelist[index]
        path = os.path.join(self.opt.baseroot+ imgname)
        array = read_image(path)
        array = cv2.resize(array, (self.opt.imgsize, self.opt.imgsize))

        """
        DICOM Preprocessing
        """
        array = exposure.rescale_intensity(array)
        array = (array-np.min(array))/(np.max(array)-np.min(array)) #(0, 1)
        array = 2*array-1                                           #(-1, 1)
        array = cv2.merge([array, array, array])


        # mask
        if self.opt.mask_type == 'free_form':
            maskname = os.path.splitext(imgname)[0] + '.png'
            maskpath = os.path.join(self.opt.maskroot+ maskname)
            
            mask = cv2.imread(maskpath, 0)
            mask = cv2.resize(mask, (self.opt.imgsize, self.opt.imgsize))
            _, mask = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY) 
            
            # edge mask
            mask_edge = gaussian_filter(mask*255, sigma=5)
            mask_edge[ mask_edge > 200 ] = 0
            mask_edge[ mask_edge < 100 ] = 0
            _, mask_edge = cv2.threshold(mask_edge, 0, 1, cv2.THRESH_BINARY) 

        # the outputs are entire image and mask, respectively
        image = torch.from_numpy(array.astype(np.float32)).permute(2, 0, 1).contiguous() 
        mask = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).contiguous()
        mask_edge = torch.from_numpy(mask_edge.astype(np.float32)).unsqueeze(0).contiguous()
        return image, mask, mask_edge

    

class ValidationSet_with_Known_Mask(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.namelist = utils.get_names(opt.baseroot)

    def __len__(self):
        return len(self.namelist)

    def __getitem__(self, index):
        # image
        imgname = self.namelist[index]
        path = os.path.join(self.opt.baseroot+ imgname)
        array = read_image(path)
        array = cv2.resize(array, (self.opt.imgsize, self.opt.imgsize))

        min_val = np.min(array)
        max_val = np.max(array)

        """
        DICOM Preprocessing
        """
        array = exposure.rescale_intensity(array)
        array = (array-np.min(array))/(np.max(array)-np.min(array)) #( 0, 1)
        array = 2*array-1                                           #(-1, 1)
        array = cv2.merge([array, array, array])

        # mask
        maskname = os.path.splitext(imgname)[0] + '.png'
        maskpath = os.path.join(self.opt.maskroot+ maskname)
        if os.path.exists(maskpath):
            mask = cv2.imread(maskpath, 0)
            mask = cv2.resize(mask, (self.opt.imgsize, self.opt.imgsize))
            ret, mask = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY) 

            # edge mask
            mask_edge = gaussian_filter(mask*255, sigma=5)
            mask_edge[ mask_edge > 200 ] = 0
            mask_edge[ mask_edge < 100 ] = 0
            ret, mask_edge = cv2.threshold(mask_edge, 0, 1, cv2.THRESH_BINARY) 

        else:
            mask = np.zeros([256, 256])
            mask_edge = np.zeros([256, 256])

        # the outputs are entire image and mask, respectively
        image = torch.from_numpy(array.astype(np.float32)).permute(2, 0, 1).contiguous() 
        mask = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).contiguous()
        mask_edge = torch.from_numpy(mask_edge.astype(np.float32)).unsqueeze(0).contiguous()
        return image, mask, mask_edge, imgname, min_val, max_val


class ValidationSet_with_Known_Mask_3D(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.namelist = utils.get_names(opt.baseroot)

    def __len__(self):
        return len(self.namelist)

    def __getitem__(self, index):
        # image
        imgname = self.namelist[index]
        path = os.path.join(self.opt.baseroot+ imgname)
        array = read_image(path)
        array = cv2.resize(array, (self.opt.imgsize, self.opt.imgsize))

        min_val = np.min(array)
        max_val = np.max(array)

        """
        DICOM Preprocessing
        """
        array = exposure.rescale_intensity(array)
        array = (array-np.min(array))/(np.max(array)-np.min(array)) #( 0, 1)
        array = 2*array-1                                           #(-1, 1)
        array = cv2.merge([array, array, array])

        # mask
        maskname = os.path.splitext(imgname)[0] + '.png'
        maskpath = os.path.join(self.opt.maskroot+ maskname)
        if os.path.exists(maskpath):
            mask = cv2.imread(maskpath, 0)
            mask = cv2.resize(mask, (self.opt.imgsize, self.opt.imgsize))
            ret, mask = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY) 

            # edge mask
            mask_edge = gaussian_filter(mask*255, sigma=5)
            mask_edge[ mask_edge > 200 ] = 0
            mask_edge[ mask_edge < 100 ] = 0
            ret, mask_edge = cv2.threshold(mask_edge, 0, 1, cv2.THRESH_BINARY) 

        else:
            mask = np.zeros([256, 256])
            mask_edge = np.zeros([256, 256])

        # the outputs are entire image and mask, respectively
        image = torch.from_numpy(array.astype(np.float32)).permute(2, 0, 1).contiguous() 
        mask = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).contiguous()
        mask_edge = torch.from_numpy(mask_edge.astype(np.float32)).unsqueeze(0).contiguous()
        return image, mask, mask_edge, imgname, min_val, max_val

