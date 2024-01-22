# ImageProcessing

## Using Python

### Installation Guide: 

Python 3.9 and above

Libarary: numpy, roifile, matplotlib, opencv, pandas, scipy, scikit-learn, scikit-image, joblib (parallel computing), PyImageJ (which also requires OpenJDK 11 and Maven)
https://py.imagej.net/en/latest/Install.html

Use either "conda install numpy" or "pip install pyimagej" in the Anaconda terminal 

Download and install [fiji.app](http://fiji.app) [https://imagej.net/software/fiji/](https://imagej.net/software/fiji/)
Make sure the newest fiji contains: “…\plugins\Descriptor_based_registration-2.1.8.jar” in the "plugins" folder

### ClusterSeqIP_v2.py 

Pre-process the raw images from the Seq run and output processed images in a folder that is ready for color transformation 
#### What it does: 
Image rename, filtering, binning, background normalization, magnification correction, image registration and cropping, illumination correction  
#### Executing Program:
Before running it, first make sure 
Fiji.app is downloaded and installed
PyImageJ, OpenJDK 11 and Maven are all properly installed
Then open the script and make sure JAVA_HOME is set to the correct directory in your PC
os.environ['JAVA_HOME']='C:\Program Files\Microsoft\jdk-11.0.21.9-hotspot'
and same for the fiji parth
fijipath='...your directory/Fiji.app'

Last, run the file in the terminal with "-i +your_raw_image_folder"
```
& C:/Users/...your directory/python.exe "...your directory/ClusterSeqIP_v2.py" -i "/path/to/unprocessed/raw_image_folder"
```
Python will call fiji algorithm to perform the image registration, requires users to drag the cropping rectangle to center and manually adjust the threshold to ~0.08

#### Input:
Raw .tif image folder path (don't put things that are not .tif files in that folder)
#### Output: 
2_preprocess: this folder stores preprocessed images (for debugging, and manual registration)
2_Regis: this folder stores the registered images for enduse to visualize and check the registration quality 
2_processed_final: this folder stores the final processed images for next step color transformation, pass this directory to the color_transform.py once you believe all the images are properly processed and would like to move on.  

## Using MATLAB

ClusterSeqIP_v2_IJauto.m is the matlab version of the ClusterSeqIP_v2.py that will complete the entire image processing pipeline:

### Installation Guide: 

MATLAB R2020 and above

Download and install [fiji.app](http://fiji.app) [https://imagej.net/software/fiji/](https://imagej.net/software/fiji/)
MATLAB imageJ: mij.jar, ij.jar
Make sure the fiji contains: “…\plugins\Descriptor_based_registration-2.1.8.jar”

### Executing program:
First, Add mij.jar, ij.jar and Descriptor_based_registration-2.1.8.jar to the script path

ImgFolderPath='Your Raw Image Path Folder'; %Replace this with the folder path of your raw images

Click “Run” button, the processed images will be output in the same directory of your raw image path.
