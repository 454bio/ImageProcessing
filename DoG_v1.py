import os
import csv
from skimage import io, filters, morphology, measure
import numpy as np
from scipy import ndimage, stats
from argparse import ArgumentParser
from scipy.spatial import KDTree
from joblib import Parallel, delayed

def DoG(image_path):
    """
    input: list of all the images' path
    output: BW_mask binary mask of detected spots
    """
    img = io.imread(image_path)
    #ski.io.imshow(img,cmap='gray',vmin=0,vmax=2000)
    #ski.io.show()
    img_g1 = filters.gaussian(img, sigma=s1, preserve_range=True)
    img_g2 = filters.gaussian(img, sigma=s2, preserve_range=True)
    img_dog = img_g1 - img_g2
    img_dog = np.clip(img_dog, 0, None)  #auto set negative number to zero
    img_dog = img_dog.astype(np.uint16)  # Scale and convert to uint16
    #io.imshow(img_dog,cmap='gray',vmin=0,vmax=1000)
    #io.show()

    # Lower end filter to remove debris
    img_dog[img_dog < Th_low] = 0

    # Higher end filter to remove scatters
    BW_s = img_dog.copy()
    BW_s[BW_s < Th_high] = 0
    BW_s = BW_s > 0  # Binarize
    BW_s = morphology.dilation(BW_s, se5)
    BW_s = ndimage.binary_fill_holes(BW_s)
    img_dog[BW_s] = 0

    # Single pixel noise removal 
    img_dog = morphology.dilation(morphology.erosion(img_dog, se2), se2)

    # Mask generation and cluster Area-based filtration
    BW_mask = img_dog > 0
    #BW_mask = morphology.dilation(BW_mask, se2)
    regions = morphology.label(BW_mask)
    props = measure.regionprops(regions)
    areas = [prop.area for prop in props]
    Area_Th = np.mean(areas) + 4*np.std(areas)   #careful, don't miss large clusters

    # Keeping regions with area <= Area_Th
    BW_mask = np.zeros_like(BW_mask, dtype=bool)
    for prop in props:
        if prop.area <= Area_Th:
            BW_mask[regions == prop.label] = True
    
    # do a offset correction:
    # Shift dimensions
    shift_right = 1
    shift_down = 1

    # Create a new mask with the same shape, filled with False (0)
    new_mask = np.zeros_like(BW_mask)

    # Calculate the dimensions of the shifted area
    shifted_rows = BW_mask.shape[0] - shift_down
    shifted_cols = BW_mask.shape[1] - shift_right

    # Shift the mask content
    new_mask[shift_down:, shift_right:] = BW_mask[:shifted_rows, :shifted_cols]

    # Replace the original mask with the new, shifted mask
    BW_mask = new_mask
    
    return BW_mask


def filter_centroids_kdtree(centroids, min_distance=4):
    # Create a KDTree for efficient nearest neighbor search
    tree = KDTree(centroids)

    # Initialize list to keep track of indices to remove
    to_remove = set()

    # Iterate over each centroid and query the KDTree for nearest neighbors
    for idx, centroid in enumerate(centroids):
        # Query the KDTree for nearest neighbors within min_distance
        # query_ball_point returns indices of points within min_distance
        if idx not in to_remove:  # Skip if already marked for removal
            neighbors = tree.query_ball_point(centroid, min_distance)
            for neighbor in neighbors:
                if neighbor != idx:  # Exclude the point itself
                    to_remove.add(neighbor)

    # Filter out centroids that are too close to each other
    filtered_centroids = [centroid for idx, centroid in enumerate(centroids) if idx not in to_remove]

    return filtered_centroids

def filter_centroids_bd(image_path, centroids, bd):
    """Filter out centroids that are within 'bd' pixels of the image boundary."""
    img = io.imread(image_path)
    height, width = img.shape[:2]
    # Filter centroids
    filtered_centroids = [centroid for centroid in centroids if bd <= centroid[0] < height - bd and bd <= centroid[1] < width - bd]
    return filtered_centroids

def compute_intensities(image_path, centroids, half_size):
    """Function to compute sum of intensities around each centroid for one image"""
    img = io.imread(image_path)
    intensities = []
    BG = round((np.mean(img)-np.std(img)/3)*(half_size*2+1)**2)
    for centroid in centroids:
        row, col = int(centroid[0]), int(centroid[1])
        start_row = max(row - half_size, 0)
        end_row = min(row + half_size + 1, img.shape[0])
        start_col = max(col - half_size, 0)
        end_col = min(col + half_size + 1, img.shape[1])
        region = img[start_row:end_row, start_col:end_col]
        intensities.append(np.sum(region))
    return intensities, BG 

def cal_thres(column_values):
    """Function to calculate threshold for a column intensity"""
    return np.mean(column_values) + 3 * np.std(column_values)

def calculate_chastity(row):
    """Function to calculate chastity"""
    top_two = np.partition(row[:4], -2)[-2:]  # Get the two highest values of the row[:4] first four columns
    return top_two[1] / np.sum(top_two)  # Ratio of max to sum of max and second max

def mask_draw(image_path, centroids, half_size):
    """Function to draw a 2D binary mask based on centroids location and dilated to square shape based on half_size"""
    img = io.imread(image_path)
    mask = np.zeros_like(img, dtype=bool)
    for centroid in centroids:
        row, col = int(centroid[0]), int(centroid[1])
        start_row = max(row - half_size, 0)
        end_row = min(row + half_size + 1, img.shape[0])
        start_col = max(col - half_size, 0)
        end_col = min(col + half_size + 1, img.shape[1])
        mask[start_row:end_row, start_col:end_col] = True
    return mask

def find_median_rows_bin(data, col_index, num_bins):
    """Function to find rows closest to the median in each bin for the specified column."""
    # Bin the data
    min_val = np.min(data[:, col_index])
    max_val = np.max(data[:, col_index])
    bins = np.linspace(min_val, max_val, num_bins+1)   
    median_rows = []

    for i in range(num_bins):
        # Find indices in the current bin
        bin_mask = (data[:, col_index] >= bins[i]) & (data[:, col_index] < bins[i+1])
        bin_data = data[bin_mask]
        
        if len(bin_data) > 0:
            # Compute median for the current bin
            median_val = np.median(bin_data[:, col_index])
            # Find the index in the bin_data where the column value is closest to the median
            closest_idx = np.argmin(np.abs(bin_data[:, col_index] - median_val))
            median_rows.append(bin_data[closest_idx])
    
    return np.array(median_rows)

# Main function section
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
    print(f"Input final processed image path: {img_folder_path}")

# Read images
#img_folder_path = "D:/454 Work/Raw Data/Key_Seq_Results/20231217_S0592_AG-DISC/1_original/2_processed_final"
os.chdir(img_folder_path)
file_list = sorted(os.listdir(img_folder_path), key=lambda x: int(x.split('_')[0]))
filename = [os.path.join(img_folder_path, file) for file in file_list]
total_images = len(filename)

# Initialize mask array
img = io.imread(filename[0])
BW_composite = np.zeros_like(img, dtype=bool)

# Magic setting parameters
s1 = 0.5      # sigma1
s2 = 1        # sigma2
Th_low = 10    # Magic number, very sensitive
Th_high = 400 # Magic number, optimize this one
se2 = morphology.square(2)
se3 = morphology.square(3)
se5 = morphology.disk(5)

BW_all = Parallel(n_jobs=-1)(delayed(DoG)(filename[i]) for i in range(4))

# Extract centroids and add them to the centroid_all list
centroid_all = []
for mask in BW_all:
    BW_composite = BW_composite | mask
    # Label the connected regions in the mask, Compute properties of each labeled region
    labeled_mask = measure.label(mask)
    props = measure.regionprops(labeled_mask)

    # Centroids are returned as (row, col) which corresponds to (y, x)
    for prop in props:
        centroid_all.append(prop.centroid)

print("Total DoG detected centroids:",len(centroid_all))
filtered_centroids = filter_centroids_kdtree(centroid_all)   #remove double counting in multi-channel detection
bd_pixel = 5   #remove any centroids that is within 5 pixels of the image boundary 
filtered_centroids = filter_centroids_bd(filename[0],filtered_centroids, bd_pixel) 
print("Total number of filtered centroids:", len(filtered_centroids))

#check the mask, optional  
#io.imshow(BW_composite,cmap='gray',vmin=0,vmax=2)
#io.show()

# Assuming BW_all is a list of four 2D boolean masks
os.chdir('..')
output_folder = 'Binary_Mask'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
os.chdir(output_folder)
for idx, bw_mask in enumerate(BW_all, start=1):
    Maskname = f"{idx:05}.tif"  # This will create filenames like '00001.tif', '00002.tif', etc.
    io.imsave(Maskname, bw_mask.astype('uint16') * 65535)
os.chdir('..')

#Intensity extraction
half_size = 1   # 1 -> 3x3 square
# Run in parallel
results = Parallel(n_jobs=-1)(delayed(compute_intensities)(file, filtered_centroids, half_size) for file in filename)
# Separate results into intensities and background values
intensities = [result[0] for result in results]
BG_values = [result[1] for result in results]
# Convert results to a NumPy array
intensities = np.array(intensities).T  # Transpose to match the desired shape
BG_values = np.array(BG_values)

# Auto Dictionary Section
# Scatter removal fitler, iterate through the last cycle 4 images
# Calculate thresholds for the last four columns and stored in a list
Th_int = [cal_thres(intensities[:, -4 + i]) for i in range(4)]
# Create a 1D boolean numpy array, has the size of entire number of centroids, and make sure int<Th_int for all last 4 cols or 4 images
mask = np.all([intensities[:, -4 + i] < Th_int[i] for i in range(4)], axis=0)
# Apply the mask to filter centroids and intensities
filtered_centroids = np.array(filtered_centroids)[mask]   #note filtered_centroids become a 2D numpy array not a list of tuples, be careful!
intensities = intensities[mask]
# Now, filtered_centroids and intensities only contain the data below the thresholds for the last four columns
print("Total number of filtered centroids after scatter removal:", filtered_centroids.shape[0])

#Define Chastity filter
Th_chastity = 0.55
# Calculate chastity for each row, apply threshold
chastity_values = np.array([calculate_chastity(row) for row in intensities])
chastity_mask = chastity_values >= Th_chastity
# Filter intensities and centroids
intensities = intensities[chastity_mask]
filtered_centroids = filtered_centroids[chastity_mask]
filtered_centroids = filtered_centroids.astype(int)  #all convert to integer now
print("Total number of filtered centroids after chastity:", filtered_centroids.shape[0])

#save the final mask after chastity
os.chdir(output_folder)
Maskname = "Mask_afterChastity.tif"  # This will create filenames like '001.tif', '002.tif', etc.
BW_chastity = mask_draw(filename[0], filtered_centroids, half_size)
io.imsave(Maskname, BW_chastity.astype('uint16') * 65535)
os.chdir('..')

#Color Classification
Data_cluster = np.hstack((filtered_centroids, intensities))  #horizontal stack
# Find the index of the maximum value in the first four columns of intensities
max_indices = np.argmax(Data_cluster[:, 2:6], axis=1)

# Classify the data based on the index of the maximum intensity
data_T = Data_cluster[max_indices == 0]
data_A = Data_cluster[max_indices == 1]
data_C = Data_cluster[max_indices == 2]
data_G = Data_cluster[max_indices == 3]
# Sorting data_T, data_A, data_C, and data_G based on their respective columns
data_T = data_T[data_T[:, 2].argsort()[::-1]]  # Sort by column 2 (T intensity)
data_A = data_A[data_A[:, 3].argsort()[::-1]]  # Sort by column 3 (A intensity)
data_C = data_C[data_C[:, 4].argsort()[::-1]]  # Sort by column 4 (C intensity)
data_G = data_G[data_G[:, 5].argsort()[::-1]]  # Sort by column 5 (G intensity)

print(f"Number of T: {data_T.shape[0]}, A: {data_A.shape[0]}, C: {data_C.shape[0]}, G: {data_G.shape[0]}")

# Concatenate all arrays vertically after chastity
combined_data = np.vstack((data_T, data_A, data_C, data_G)) 

#key index sorting for dictionary candidates pick-up
Th_cleave = 0.2    #magic number to ensure cleavage is more than 20%, without cutting bg   
Th_incorp = 0.1    #magic number to ensure incorporation is more than 10%

CleaveRate_T = (data_T[:,2] - data_T[:,6]) / data_T[:,2]
Incorp_T = (data_T[:,8] - data_T[:,4]) / data_T[:,4]
idx_maskT = (CleaveRate_T > Th_cleave)&(Incorp_T > Th_incorp)
data_T = data_T[idx_maskT]

CleaveRate_A = (data_A[:,3] - data_A[:,7]) / data_A[:,3]
Incorp_A = (data_A[:,6] - data_A[:,2]) / data_A[:,2]
idx_maskA = (CleaveRate_A > Th_cleave)&(Incorp_A > Th_incorp)
data_A = data_A[idx_maskA]

CleaveRate_C = (data_C[:,4] - data_C[:,8]) / data_C[:,4]
Incorp_C = (data_C[:,9] - data_C[:,5]) / data_C[:,5]
idx_maskC = (CleaveRate_C > Th_cleave)&(Incorp_C > Th_incorp)
data_C = data_C[idx_maskC]

CleaveRate_G = (data_G[:,5] - data_G[:,9]) / data_G[:,5]
Incorp_G = (data_G[:,7] - data_G[:,3]) / data_G[:,3]
idx_maskG = (CleaveRate_G > Th_cleave)&(Incorp_G > Th_incorp)
data_G = data_G[idx_maskG]

print(f"Number of T: {data_T.shape[0]}, A: {data_A.shape[0]}, C: {data_C.shape[0]}, G: {data_G.shape[0]} after key screening")

# Dictionary pick up, find the row of the data that is closest to the median
num_bins=1   #set it to non-one only for research
dict_T = find_median_rows_bin(data_T, 2, num_bins)
dict_A = find_median_rows_bin(data_A, 3, num_bins)
dict_C = find_median_rows_bin(data_C, 4, num_bins)
dict_G = find_median_rows_bin(data_G, 5, num_bins)

# Concatenate dictionaries
BG_values = np.concatenate(([0, 0], BG_values))  #add fake centroid in front of BG_values
combined_data = np.vstack((combined_data, dict_T, dict_A, dict_C, dict_G, BG_values.reshape(1,-1)))

# Generate Spot_IDs and create headers
num_rows = combined_data.shape[0]
spot_ids = [str(i) for i in range(num_rows - 5)] + ["T", "A", "C", "G", "BG"]
image_headers = [os.path.splitext(os.path.basename(file))[0] for file in filename]
headers = ['Spot_ID', 'Row(Y)', 'Col(X)'] + image_headers

# Combine Spot_IDs with combined_data
table = [headers] + [ [spot_ids[i]] + combined_data[i].tolist() for i in range(num_rows)]

# Write the table to a CSV file
with open('Cluster_intensities.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerows(table)

"""
# Initialize the table with headers
image_headers = [os.path.splitext(os.path.basename(file))[0] for file in filename]
headers = ['Spot_ID', 'X(col)', 'Y(row)'] + image_headers
table = [headers]
# Add the centroid information and the intensities to the table
for idx, (centroid, intensity_values) in enumerate(zip(filtered_centroids, intensities)):
    row_data = [str(idx)] + centroid.tolist()[::-1] + intensity_values.tolist()  #switch centroid order to match X, Y format, [::-1]
    table.append(row_data)
"""