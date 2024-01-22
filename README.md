# ImageProcessing

## Using Python

### Installation Guide: 

Python 3.9 and above

Libarary: numpy, roifile, matplotlib, opencv, pandas, scipy, scikit-learn, scikit-image, joblib (parallel computing), PyImageJ (which also requires OpenJDK 11 and Maven)
https://py.imagej.net/en/latest/Install.html

Use either "conda install numpy" or "pip install pyimagej" in the Anaconda terminal 

Download and install [fiji.app](http://fiji.app) [https://imagej.net/software/fiji/](https://imagej.net/software/fiji/)

### ClusterSeqIP_v2.py 

Pre-process the raw images from the Seq run and output processed images in a folder that is ready for color transformation: 
What it does: Image rename, filtering, binning, background normalization, magnification correction, image registration and cropping, illumination correction  
Executing Program:
Before running it, first make sure 
Fiji.app is downloaded and installed
PyImageJ, OpenJDK 11 and Maven are all properly installed
Then, make sure JAVA_HOME is set to the correct directory in your PC
os.environ['JAVA_HOME']='C:\Program Files\Microsoft\jdk-11.0.21.9-hotspot'

modi
fijipath='C:/Users/yujin/OneDrive/Desktop/Fiji.app'
```
& C:/Users/...your directory/python.exe "...your directory/ClusterSeqIP_v2.py" -i "/path/to/unprocessed/raw_image_folder"
```



### Color Transformation:



### Dephasing Correction and Basecall:

Run default_analysis.sh file in terminal

with “color_transformed_spots.csv” add to the directory path

## MATLAB

ClusterSeqIP_v2_IJauto.m is the matlab version of the ClusterSeqIP_v2.py that will complete the entire image processing pipeline:

### Installing:

MATLAB R2020 and above

Download and install [fiji.app](http://fiji.app) [https://imagej.net/software/fiji/](https://imagej.net/software/fiji/)

MATLAB imageJ: mij.jar, ij.jar

Make sure the fiji contains: “…\plugins\Descriptor_based_registration-2.1.8.jar”

### Executing program:

Add mij.jar, ij.jar and Descriptor_based_registration-2.1.8.jar to the script path

ImgFolderPath='Your Raw Image Path Folder'; %Replace this with the folder path of your raw images

Click “Run” button, the processed images will be output in the same directory of your raw image path.
