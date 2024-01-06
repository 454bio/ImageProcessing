import os
import skimage as ski
from skimage import io
from skimage.transform import rescale
import numpy as np
from argparse import ArgumentParser
from joblib import Parallel, delayed

# Define Images tag labeler function
def Img_Labeler(total_images,image_ext):
    """
    input: total number of images 
    output: image name strings in "00001_645_C001" format
    """
    num_cycles = round(total_images/4)
    num_labels = ['{:05d}'.format(i) for i in range(1, total_images + 1)]
    cyc_labels = ['C{:03d}'.format(i) for i in range(1, num_cycles + 1)]
    channel_labels = ['645', '590', '525', '445']

    ImgIdx = 0
    ImgLabels=['']*total_images
    for cyc_label in cyc_labels:
        for channel in channel_labels:
            ImgLabels[ImgIdx]=f'{num_labels[ImgIdx]}_{channel}_{cyc_label}{image_ext}'
            ImgIdx = ImgIdx + 1
    return ImgLabels

# Define BG normalization correction
def bg_norm(filename):
    """
    input: list of all the images' path
    output: the correction factor list for each image
    """
    bk = np.zeros(len(filename)) 
    for i, file_path in enumerate(filename):
        img = io.imread(file_path)
        bk[i] = np.mean(img)
    corr_factor = np.max(bk) / bk
    return corr_factor

def img_preprocessing(image_path, ImgLabel, corr_factor, scale_factors, ind_img):
    # Read the image
    img = io.imread(image_path)

    # Apply median filter
    img = ski.filters.median(img, ski.morphology.disk(1.8))

        # 2-by-2 nearest neighbor binning
    img = rescale(img, 0.5, anti_aliasing=True, order=0)
    img_size = img.shape

    # Apply correction factor
    img = (img * corr_factor).astype(np.uint16)

    # Apply scale factor
    scale_idx = ind_img % 4
    if scale_idx == 0:
        img = rescale(img, scale_factors[3], anti_aliasing=True, order=0)
    else:
        img = rescale(img, scale_factors[scale_idx - 1], anti_aliasing=True, order=0)

    # Crop to its unscaled size, Save the processed image
    if scale_idx != 0:
        img_size_new = img.shape
        img_diff = (img_size_new[0] - img_size[0], img_size_new[1] - img_size[1])
        crop = (round(img_diff[0]/2), round(img_diff[1]/2), (img_size[0]+round(img_diff[0]/2)), (img_size[1]+round(img_diff[1]/2)))
        img_c = img[crop[0]:crop[2], crop[1]:crop[3]]
        io.imsave(ImgLabel, img_c)
    else:
        io.imsave(ImgLabel, img)


# Set parser for raw images path
parser = ArgumentParser()
parser.add_argument("-i", "--input",
            required=True,
            action='store',
            dest='img_folder_path',
            help="Directory with .tif files, e.g.: \20231217_S0592_AG-DISC\1_original")

if __name__ == "__main__":
    args = parser.parse_args()
    img_folder_path = args.img_folder_path  
    print(f"Input raw image path: {img_folder_path}")

# Specify some parameters
cutoff = [150, 200, 200, 200]
image_ext = '.tif'
scale_factors = [1, 1.0018, 1.0023, 1.0023]  # Modify as needed, based on MK3

# Read original images
#img_folder_path="D:/454 Work/Raw Data/Key_Seq_Results/20231217_S0592_AG-DISC/1_original"
os.chdir(img_folder_path)
file_list = sorted(os.listdir(img_folder_path), key=lambda x: int(x.split('_')[0]))
filename = [os.path.join(img_folder_path, file) for file in file_list]
total_images = len(filename)
ImgLabels=Img_Labeler(total_images,image_ext)
corr_factors=bg_norm(filename)

# Create output folder
output_folder = '2_temp'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
os.chdir(output_folder)

Parallel(n_jobs=-1)(delayed(img_preprocessing)(filename[i], ImgLabels[i], corr_factors[i], scale_factors, i) for i in range(total_images))   
os.chdir('..')

'''
# Image pre-processing
# test section
img = ski.io.imread(filename[0])
#ski.io.imshow(img,cmap='gray',vmin=0,vmax=3000)
#ski.io.show()
#1.8 is the optimzied magic number for MK3, 2 is too big, 1.5 is not enough
img_med = ski.filters.median(img, ski.morphology.disk(1.8))  
img = rescale(img_med, 0.5, anti_aliasing=True, order=0)
io.imsave(ImgLabels[0], img)

img = (img * corr_factor[0]).astype(np.uint16)
io.imsave(ImgLabels[1], img)
img = rescale(img, scale_factors[3], anti_aliasing=True, order=0)
img_size_new=img.shape
img_diff = (img_size_new[0] - img_size[0], img_size_new[1] - img_size[1])
img_diff


for ind_img in range(total_images):
    img = io.imread(filename[ind_img])
    
    # Apply median filter
    img = ski.filters.median(img, ski.morphology.disk(1.8))

    # 2-by-2 nearest neighbor binning
    img = rescale(img, 0.5, anti_aliasing=True, order=0)
    img_size = img.shape

    # Apply correction factor
    img = (img * corr_factor[ind_img]).astype(np.uint16)

    # Apply scale factor
    scale_idx = ind_img % 4
    if scale_idx == 0:
        img = rescale(img, scale_factors[3], anti_aliasing=True, order=0)
    else:
        img = rescale(img, scale_factors[scale_idx - 1], anti_aliasing=True, order=0)

    # Crop to its unscaled size, Save the processed image
    if scale_idx != 0:
        img_size_new = img.shape
        img_diff = (img_size_new[0] - img_size[0], img_size_new[1] - img_size[1])
        crop = (round(img_diff[0]/2), round(img_diff[1]/2), (img_size[0]+round(img_diff[0]/2)), (img_size[1]+round(img_diff[1]/2)))
        img_c = img[crop[0]:crop[2], crop[1]:crop[3]]
        io.imsave(ImgLabels[ind_img], img_c)
    else:
        io.imsave(ImgLabels[ind_img], img)

# Change directory to one level up 
'''

