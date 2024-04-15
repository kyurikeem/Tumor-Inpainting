import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision as tv
import network
import math
from PIL import Image, ImageDraw, ImageFont

# ----------------------------------------
#                 Network
# ----------------------------------------
def create_generator(opt):
    # Initialize the networks
    generator = network.GatedGenerator_HardAttention(opt)
    print('Generator is created!')
    if opt.load_name:
        generator = load_dict(generator, opt.load_name)
    else:
        # Init the networks
        network.weights_init(generator, init_type = opt.init_type, init_gain = opt.init_gain)
        print('Initialize generator with %s type' % opt.init_type)
    return generator

def create_discriminator(opt):
    # Initialize the networks
    discriminator = network.PatchDiscriminator(opt)
    print('Discriminator is created!')
    # Init the networks
    network.weights_init(discriminator, init_type = opt.init_type, init_gain = opt.init_gain)
    print('Initialize discriminator with %s type' % opt.init_type)
    return discriminator

def create_perceptualnet():
    # Get the first 15 layers of vgg16, which is conv3_3
    perceptualnet = network.PerceptualNet()
    # Pre-trained VGG-16
    vgg16 = torch.load('./vgg16_pretrained.pth')
    load_dict_perceptualnet(perceptualnet, vgg16)
    # It does not gradient
    for param in perceptualnet.parameters():
        param.requires_grad = False
    print('Perceptual network is created!')
    return perceptualnet

def load_dict_perceptualnet(process_net, pretrained_net):
    pretrained_dict = pretrained_net
    process_dict = process_net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in process_dict}
    process_dict.update(pretrained_dict)
    process_net.load_state_dict(process_dict)
    return process_net

def load_dict(process_net, pretrained_net):
    pretrained_dict = torch.load(pretrained_net)         
    process_dict = process_net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in process_dict}
    process_dict.update(pretrained_dict)
    process_net.load_state_dict(process_dict)
    return process_net
    
# ----------------------------------------
#             PATH processing
# ----------------------------------------
def text_readlines(filename):
    # Try to read a txt file and return a list.Return [] if there was a mistake.
    try:
        file = open(filename, 'r')
    except IOError:
        error = []
        return error
    content = file.readlines()
    # This for loop deletes the EOF (like \n)
    for i in range(len(content)):
        content[i] = content[i][:len(content[i])-1]
    file.close()
    return content

def savetxt(name, loss_log):
    np_loss_log = np.array(loss_log)
    np.savetxt(name, np_loss_log)

def get_files(path):
    # read a folder, return the complete path
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(os.path.join(root, filespath))
    return ret

def get_names(path):
    # read a folder, return the image name
    ret = []
    for root, dirs, files in os.walk(path):
       for filespath in files:
           ret.append(filespath)

    return ret

def text_save(content, filename, mode = 'a'):
    # save a list to a txt
    # Try to save a list variable in txt file.
    file = open(filename, mode)
    for i in range(len(content)):
        file.write(str(content[i]) + '\n')
    file.close()

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    
# ----------------------------------------
#    Validation and Sample at training
# ----------------------------------------
def save_sample_png(sample_folder, sample_name, img_list, name_list, pixel_max_cnt = 255):
    # Save image one-by-one
    for i in range(len(img_list)):
        img = img_list[i]
        # Recover normalization: * 255 because last layer is sigmoid activated
        # img = img * 255
        img = (((img+1)/2)* 255)
        # Process img_copy and do not destroy the data of img
        img_copy = img.clone().data.permute(0, 2, 3, 1)[0, :, :, 1].cpu().numpy()
        print('Saving img..', name_list[i],':', np.min(img_copy), np.max(img_copy))
        img_copy = np.clip(img_copy, 0, pixel_max_cnt)
        img_copy = img_copy.astype(np.uint8)
        # Save to certain path
        save_img_name = sample_name + '_' + name_list[i] + '.png'
        save_img_path = os.path.join(sample_folder, save_img_name)
        cv2.imwrite(save_img_path, img_copy)


def convert_save_format(img, pixel_max_cnt = 255):
    # Recover normalization: * 255 because last layer is sigmoid activated
    img = (((img+1)/2)* 255)
    # Process img_copy and do not destroy the data of img
    img_png = img.clone().data.permute(0, 2, 3, 1)[0, :, :, 1].cpu().numpy()
    img_png = np.clip(img_png, 0, pixel_max_cnt)
    img_png = img_png.astype(np.uint8)
    return img_png


def add_labels_to_image(image_array, labels, font_size=20, label_color=(255, 255, 255), font_path=None, option='up'):
    img = Image.fromarray(image_array)
    draw = ImageDraw.Draw(img)

    if font_path:
        font = ImageFont.truetype(font_path, font_size)
    else:
        font = ImageFont.load_default()
    section_width = image_array.shape[1] // len(labels)

    if option == 'up':
        y_position = 10  # Padding from the top
    elif option == 'down':
        text_height = draw.textbbox((0, 0), labels[0], font=font)[3]  # Height of text
        y_position = image_array.shape[0] - text_height - 10  # Padding from the bottom

    for i, label in enumerate(labels):
        text_width = draw.textbbox((0, 0), label, font=font)[2]  # Width of text
        x_position = i * section_width + (section_width - text_width) // 2
        draw.text((x_position, y_position), label, fill=tuple(label_color), font=font)

    return img


def psnr(img, gt):
    """
    Input: numpy array of image range [-1, 1]
    Return: psnr scalar
    """
    img = (((img+1)/2)*255).astype(np.float64) #convert to [0,1]
    gt = (((gt+1)/2)*255).astype(np.float64)   #convert to [0,1]
    mse = np.mean((img - gt)**2, dtype=np.float64)
    if mse == 0:
        return None
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


from skimage.metrics import structural_similarity
def ssim(img, gt):
    img = ((img+1)/2).astype(np.float64) #convert to [0,1]
    gt = ((gt+1)/2).astype(np.float64)   #convert to [0,1]
    return structural_similarity(img, gt, data_range=img.max() - img.min())
    

def nrmse(img, gt, normalization='euclidean'):
    #normalized_root_mse
    img = ((img+1)/2).astype(np.float64) #convert to [0,1]
    gt = ((gt+1)/2).astype(np.float64)   #convert to [0,1]
    normalization = normalization.lower()
    if normalization == 'euclidean':
        denom = np.sqrt(np.mean((gt * gt), dtype=np.float64))
    elif normalization == 'min-max':
        denom = gt.max() - gt.min()
    elif normalization == 'mean':
        denom = gt.mean()
    else:
        raise ValueError("Unsupported norm_type")
    return np.sqrt((np.mean((img - gt) ** 2, dtype=np.float64))) / denom


# ----------------------------------------
#           CVAE Mask Sampling
# ----------------------------------------
from skimage.measure import label   
import matplotlib.pyplot as plt

def idx2onehot(idx, n):

    assert torch.max(idx).item() < n

    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    onehot = torch.zeros(idx.size(0), n).to(idx.device)
    onehot.scatter_(1, idx, 1)
    
    return onehot

def getLargestCC(segmentation):
    labels = label(segmentation)
    if not labels.max() == 0:
        assert( labels.max() != 0 ) # assume at least 1 CC
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
        return largestCC*1

    else:
        return np.zeros(segmentation.shape)

def get_condition(filename):
    index = filename.split('_')[-1]
    index = os.path.splitext(index)[0]

    if 0 <= int(index) <= 15:
        label = 0
    if 15 < int(index) <= 30:
        label = 1
    if 30 < int(index) <= 45:
        label = 2
    if 45 < int(index) <= 60:
        label = 3
    if 60 < int(index) <= 75:
        label = 4
    if 75 < int(index) <= 90:
        label = 5
    if 90 < int(index) <= 105:
        label = 6
    if 105 < int(index) <= 120:
        label = 7
    if 120 < int(index) <= 135:
        label = 8
    if 135 < int(index) <= 155:
        label = 9
    return label


def get_sampled_mask(x, img):
    """
    Input:
    annotation binaty mask, input image
    
    Return:
    Temporally, return main channle mask only  - KR 21.20.21
    """

    img_show = x.view(3, 256, 256).cpu().data.numpy()  
    _, img_per_channel0 = cv2.threshold(img_show[0,:,:], 0.5, 1, cv2.THRESH_BINARY)
    _, img_per_channel1 = cv2.threshold(img_show[1,:,:], 0.5, 1, cv2.THRESH_BINARY)
    _, img_per_channel2 = cv2.threshold(img_show[2,:,:], 0.5, 1, cv2.THRESH_BINARY)
    
    img_per_channel0 = img_per_channel0[:, :, np.newaxis]
    img_per_channel1 = img_per_channel1[:, :, np.newaxis]
    img_per_channel2 = img_per_channel2[:, :, np.newaxis]
    
    Total = np.concatenate([img_per_channel0, img_per_channel1, img_per_channel2], axis = 2)    
    Total = getLargestCC(Total)

    sampled_mask = Total[:, :, 1]
    sampled_mask = getLargestCC(sampled_mask).astype(np.float32)

    # PostProcessing 
    if np.count_nonzero(sampled_mask) < 100:
        kernel = np.ones((5, 5), np.float32)
        sampled_mask = cv2.dilate(sampled_mask, kernel, iterations = 1)

    # Remove Over ROI
    img = img.clone().data[0, 0, :, :].cpu().numpy()
    _, img = cv2.threshold(img, np.min(img), 1, cv2.THRESH_BINARY)
    sampled_mask = cv2.bitwise_or((1-sampled_mask), (1-img))
    sampled_mask = 1-sampled_mask
    sampled_mask = torch.from_numpy(sampled_mask[np.newaxis, :, :].astype(np.float32)).unsqueeze(0).contiguous()

    return sampled_mask


def _get_sampled_mask(x, img):
    """
    Input:
    annotation binaty mask, input image
    
    Return:
    Temporally, return main channle mask only  - KR 21.20.21
    """

    img_show = x.view(1, 256, 256).cpu().data.numpy()  
    _, img_per_channel0 = cv2.threshold(img_show[0,:,:], 0.05, 1, cv2.THRESH_BINARY)
    img_per_channel0 = img_per_channel0[:, :, np.newaxis]
    
    # Total = np.concatenate([img_per_channel0, img_per_channel0, img_per_channel0], axis = 2)    
    Total = getLargestCC(img_per_channel0)

    sampled_mask = Total[:, :, 0]
    sampled_mask = getLargestCC(sampled_mask).astype(np.float32)
    sampled_mask = sampled_mask.astype(np.float32)

    # PostProcessing 
    # if np.count_nonzero(sampled_mask) < 100:
    kernel = np.ones((5, 5), np.float32)
    sampled_mask = cv2.dilate(sampled_mask, kernel, iterations = 1)

    # Remove Over ROI
    img = img.clone().data[0, 0, :, :].cpu().numpy()
    _, img = cv2.threshold(img, np.min(img), 1, cv2.THRESH_BINARY)
    sampled_mask = cv2.bitwise_or((1-sampled_mask), (1-img))
    sampled_mask = 1-sampled_mask
    sampled_mask = torch.from_numpy(sampled_mask[np.newaxis, :, :].astype(np.float32)).unsqueeze(0).contiguous()

    return sampled_mask

