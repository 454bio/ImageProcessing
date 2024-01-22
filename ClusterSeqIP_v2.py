import os
import imagej
import skimage as ski
from skimage import io
from skimage.transform import rescale
import numpy as np
from argparse import ArgumentParser
from joblib import Parallel, delayed
os.environ['JAVA_HOME']='C:\Program Files\Microsoft\jdk-11.0.21.9-hotspot'
fijipath='C:/Users/yujin/OneDrive/Desktop/Fiji.app'

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
def bg_norm(filenames):
    """
    input: list of all the images' path
    output: the correction factor list for each image
    """
    bk = np.zeros(len(filenames)) 
    for i, file_path in enumerate(filenames):
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

# dark edge cropping algorithm
def dark_edge_crop(filenames):
    max_first_row = max_first_col = 0
    min_last_row = min_last_col = float('inf')

    for filename in filenames:
        img = io.imread(filename)

        # Sum pixel values along rows and columns
        row_sum = np.sum(img, axis=1)
        col_sum = np.sum(img, axis=0)

        # Find the first and last indices where the sum is greater than zero
        first_row = np.argmax(row_sum > (200 * img.shape[1]))
        last_row = len(row_sum) - np.argmax(row_sum[::-1] > (200 * img.shape[1])) - 1   #reverse the array index [::-1] start,end,step=-1
        first_col = np.argmax(col_sum > (200 * img.shape[0]))
        last_col = len(col_sum) - np.argmax(col_sum[::-1] > (200 * img.shape[0])) - 1

        # Ensure safe cropping
        max_first_row = max(max_first_row, first_row)
        max_first_col = max(max_first_col, first_col)
        min_last_row = min(min_last_row, last_row)
        min_last_col = min(min_last_col, last_col)

    crop = (max_first_row, min_last_row, max_first_col, min_last_col)
    return crop

#Define light source correction
def img_gaussianCorr(image_path, ImgLabel, crop):
    # Read the image
    img = io.imread(image_path)
    img_crop = img[crop[0]:crop[1], crop[2]:crop[3]]

    # Apply filter
    img_blur = ski.filters.gaussian(img_crop, sigma=50, preserve_range=True)
    img_corr = (img_crop / img_blur * np.mean(img_blur)).astype(np.uint16)

    # Crop to its unscaled size, Save the processed image
    io.imsave(ImgLabel, img_corr)

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
image_ext = '.tif'
scale_factors = [1, 1.0018, 1.0023, 1.0023]  # Modify as needed, based on MK3.2

# Read original images
#img_folder_path="D:/454 Work/Raw Data/Key_Seq_Results/20231217_S0592_AG-DISC/1_original/2_temp"
os.chdir(img_folder_path)
file_list = sorted(os.listdir(img_folder_path), key=lambda x: int(x.split('_')[0]))
filename = [os.path.join(img_folder_path, file) for file in file_list]
total_images = len(filename)
ImgLabels=Img_Labeler(total_images,image_ext)
corr_factors=bg_norm(filename)

# Create output folder
output_folder = '2_preprocess'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
os.chdir(output_folder)
Parallel(n_jobs=-1)(delayed(img_preprocessing)(filename[i], ImgLabels[i], corr_factors[i], scale_factors, i) for i in range(total_images))
temp_path=os.getcwd()

os.chdir('..')
output_folder = '2_Regis'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
os.chdir(output_folder)
output_path=os.getcwd()
ij = imagej.init(fijipath,mode='interactive')

#ImageJ registration section
try:
    ij.IJ.run("Image Sequence...", "open=["+temp_path+"] sort use")
    ij.IJ.run("Descriptor-based series registration (2d/3d + t)", "series_of_images=2_preprocess brightness_of=[Interactive ...] approximate_size=[Interactive ...] type_of_detections=[Maxima only] subpixel_localization=None transformation_model=[Rigid (2d)] number_of_neighbors=3 redundancy=1 significance=3 allowed_error_for_ransac=5 global_optimization=[All-to-all matching with range ('reasonable' global optimization)] range=5 choose_registration_channel=1 image=[Fuse and display] interpolation=[Linear Interpolation]")
    #ij.IJ.run("Descriptor-based series registration (2d/3d + t)", "series_of_images=2_temp brightness_of=Medium Strong approximate_size=[5 px] [Interactive ...] type_of_detections=[Maxima only] subpixel_localization=[3-dimensional quadratic fit] transformation_model=[Rigid (2d)] number_of_neighbors=3 redundancy=1 significance=3 allowed_error_for_ransac=5 global_optimization=[All-to-all matching with range ('reasonable' global optimization)] range=5 choose_registration_channel=1 image=[Fuse and display] interpolation=[Linear Interpolation]")
    ij.IJ.selectWindow("registered [XYCTZ] 2_preprocess")
    ij.IJ.run("Image Sequence... ", "format=TIFF name=[] start=1 digits=5 save=["+output_path+"]")
    ij.IJ.run("Close All")
finally:
    ij.getContext().dispose()

#dark edge cropping
output_path="D:/454 Work/Raw Data/Key_Seq_Results/20231217_S0592_AG-DISC/1_original/2_Regis"
os.chdir(output_path)
file_list_reg = sorted(os.listdir(output_path))
filename_reg = [os.path.join(output_path, file) for file in file_list_reg]
crop = dark_edge_crop(filename_reg)

# Create final processed image output folder
os.chdir('..')
output_folder = '2_processed_final'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
os.chdir(output_folder)
Parallel(n_jobs=8)(delayed(img_gaussianCorr)(filename_reg[i], ImgLabels[i], crop) for i in range(total_images))  #-1 is not working? make sure your cpu has 8 cores
os.chdir('..')
