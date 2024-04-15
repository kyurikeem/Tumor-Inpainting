import os
import cv2
import argparse
import torch
import numpy as np
import SimpleITK as sitk
from torch.utils.data import DataLoader
import dataset
import utils
from utils import ssim, psnr, nrmse


if __name__ == "__main__":

    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # General parameters
    parser.add_argument('--pre_train', type = bool, default = True, help = 'the type of GAN for training')
    parser.add_argument('--load_name', type = str, default = "./ckpt/model.pth", help = 'the load name of models')
    parser.add_argument('--savepath', type = str, default = "./results")
    parser.add_argument('--test_batch_size', type = int, default = 1, help = 'test batch size')
    parser.add_argument('--num_workers', type = int, default = 1, help = 'num of workers')
    # Network parameters
    parser.add_argument('--in_channels', type = int, default = 4, help = 'input RGB image')
    parser.add_argument('--out_channels', type = int, default = 3, help = 'output RGB image')
    parser.add_argument('--mask_channels', type = int, default = 1, help = 'input mask')
    parser.add_argument('--latent_channels', type = int, default = 64, help = 'latent channels')
    parser.add_argument('--pad_type', type = str, default = 'reflect', help = 'the padding type')
    parser.add_argument('--activation', type = str, default = 'lrelu', help = 'the activation type')
    parser.add_argument('--norm', type = str, default = 'in', help = 'normalization type')
    parser.add_argument('--init_type', type = str, default = 'xavier', help = 'the initialization type')
    parser.add_argument('--init_gain', type = float, default = 0.02, help = 'the initialization gain')
    # Dataset parameters
    parser.add_argument('--baseroot', type = str, default = "./samples/t2/", help = 'Images folder')
    parser.add_argument('--maskroot', type = str, default = "./samples/mask/", help = 'Masks folder (if exist)')
    parser.add_argument('--imgsize', type = int, default = 256, help = 'size of image')
    opt = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"

    # ----------------------------------------
    #                Run Test
    # ----------------------------------------
    # Initialize
    generator = utils.create_generator(opt).cuda()
    discriminator = utils.create_discriminator(opt).cuda()
    print('Model Loaded ...', opt.load_name)
    
    test_dataset = dataset.ValidationSet_with_Known_Mask(opt)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = opt.test_batch_size, shuffle = False, num_workers = opt.num_workers, pin_memory = True)
    utils.check_path(opt.savepath)
    print('Process Running ...')

    # forward
    for i, (img, mask, mask_edge, imgname, min_val, max_val) in enumerate(test_loader):    
        
        img = img.cuda()                                        
        mask = mask.cuda()                                      
        mask_edge = mask_edge.cuda()                                     
        
        with torch.no_grad():
            coarse, fine = generator(img, mask, mask_edge)                  

        # ----------------------------------------
        #           Evaluate Metrix
        # ----------------------------------------
        fusion_coarse = img * (1 - mask) + coarse * mask         
        fusion_fine = img * (1 - mask) + fine * mask         

        t2 = img.clone().data[0, 1, :, :].cpu().numpy() 
        fusion_coarse = fusion_coarse.clone().data[0, 1, :, :].cpu().numpy()
        fusion_fine = fusion_fine.clone().data[0, 1, :, :].cpu().numpy()
        mask = mask.clone().data[0, 0, :, :].cpu().numpy()

        psnr_val = psnr(fusion_fine, t2)
        ssim_val = ssim(fusion_fine, t2)
        nrmse_val = nrmse(fusion_fine, t2)
        print('%i %s, PSNR: %f, SSIM: %f, NRMSE: %f, Mask_Size: %i' %(i, imgname[0], psnr_val, ssim_val, nrmse_val,  np.count_nonzero(mask)))

        # ----------------------------------------
        #              Image Save
        # ----------------------------------------
        t2 = (((t2+1)/2)* 255.0).astype(np.uint8)
        fusion_coarse_png = (((fusion_coarse+1)/2)* 255.0).astype(np.uint8)
        fusion_fine_png = (((fusion_fine+1)/2)* 255.0).astype(np.uint8)
        save_mask = (mask * 255.0).astype(np.uint8)

        max_val = max_val.clone().cpu().numpy()
        saver = np.concatenate((t2, save_mask, t2*(1-mask)+(mask), fusion_fine_png), axis = 1)        

        labels = ['Ground Truth', 'Mask', 'Masked Input', 'Output']
        result = utils.add_labels_to_image(saver, labels, font_size=20, label_color=[255], option='down')
        cv2.imwrite(opt.savepath + '/%s' % (imgname[0]), np.array(result))