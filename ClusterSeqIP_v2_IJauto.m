% Image processing pipeline, Last Update 12/13/2023, fully automated for
% ImageJ registration
% version 2
%% End User Input Section
clear
close all
clc

ImgFolderPath='';  %add the folder path for your raw images
CutOff = [150, 200, 200, 200];   %Define cutoff value of find maxima for imageJ

%% Read Original Images
[folderPath, ~, ~] = fileparts(ImgFolderPath);
filetype='*.tif';
f=fullfile(ImgFolderPath,filetype);
d=dir(f);
fileNumbers = cellfun(@(x) sscanf(x, '%d'), {d.name});
[~, sortIndex] = sort(fileNumbers);
d_sorted = d(sortIndex);
filename=transpose(fullfile(ImgFolderPath,{d_sorted(:).name}));
totImg=length(filename);
cd(folderPath);

%% Rename Images
% Generate row and column labels
numCycles = round(totImg/4);   %Specify your Seq cycles number, default is 16
numLabels = compose('%05d', 1:totImg);
cycLabels = compose('C%03d', 1:numCycles);
channelLabels = {'645', '590', '525', '445'};
numChannels = length(channelLabels);

ImgLabels = cell(1, numChannels * numCycles);
ImgIdx = 1;
for cycIdx = 1:numCycles
    for channelIdx = 1:numChannels
        ImgLabels{ImgIdx} = strcat(numLabels(ImgIdx), '_', channelLabels{channelIdx}, '_', cycLabels(cycIdx));  
        ImgIdx = ImgIdx + 1;
    end
end
imageExt='.tif';

%% bk normalization, med fitler, 2-by-2 binning, scale, 
bk(1:totImg)=0;
parfor i=1:totImg
    Img = imread(filename{i});
    bk(i)=mean(mean(Img));
end
corr_factor=max(bk)./bk;

%Scale factors
Img = imread(filename{1});
scale_factor=[1, 1.0018, 1.0023, 1.0023];    %645, 590, 525, 445  modify this!
img_size=size(imresize(Img,0.5));

%Create output folder
outputFolder='2_temp';
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end
cd(outputFolder);

%% Image pre-processing
filename_new=cell(totImg,1);
parfor ind_Img=1:totImg
    Img = imread(filename{ind_Img});
    Img = medfilt2(Img, [3,3]);  %either [3,3] will be slightly better than [2,2]
    Img = imresize(Img,0.5,"bilinear");  %2-by-2 nearest neighbor binning
    Img = uint16(double(Img).*corr_factor(ind_Img));
    %Apply scale factor
    if mod(ind_Img,4)==0
        Img = imresize(Img,scale_factor(4),"bilinear");
    else
        Img = imresize(Img,scale_factor(mod(ind_Img,4)),"bilinear");
    end
    % crop to its unscaled size
    if mod(ind_Img,4)~=1
        img_size_new=size(Img);
        img_diff = img_size_new-img_size;
        crop=[round(img_diff(2)/2) round(img_diff(1)/2) (img_size(2)-1) (img_size(1)-1)];
        Img = imcrop(Img,crop);
    end
    %Img_Seq(:,:,ind_Img)=Img;  
    imwrite(Img, [char(ImgLabels{ind_Img}), imageExt]);  %release this for imageJ
    filename_new{ind_Img} = fullfile(pwd, [char(ImgLabels{ind_Img}), imageExt]);
end

%% ImageJ registration section
%Create output folder
outputFolder='Regis';
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end
cd(outputFolder);
javaaddpath 'C:\Users\yujin\OneDrive\Desktop\Fiji.app\ij.jar';
javaaddpath 'C:\Users\yujin\OneDrive\Desktop\Fiji.app\mij.jar';
MIJ.start('C:\Users\yujin\OneDrive\Desktop\Fiji.app\plugins');

for i=1:totImg
    ij.IJ.open(filename_new{i});
end
%ij.IJ.run("Image Sequence...", "open=[D:/454 Work/Raw Data/Key_Seq_Results/20231215_S0586/2_temp/00001_645_C001.tif] sort use");
ij.IJ.run("Images to Stack", "use");
ij.IJ.run("Descriptor-based series registration (2d/3d + t)", "series_of_images=Stack brightness_of=[Interactive ...] approximate_size=[Interactive ...] type_of_detections=[Interactive ...] subpixel_localization=None transformation_model=[Rigid (2d)] number_of_neighbors=3 redundancy=1 significance=3 allowed_error_for_ransac=5 global_optimization=[All-to-all matching with range ('reasonable' global optimization)] range=5 choose_registration_channel=1 image=[Fuse and display] interpolation=[Linear Interpolation]");
%ij.IJ.run("Image Sequence... ", "select=["+pwd+"\Regis\] dir=["+pwd+"\Regis\] format=TIFF name=[] start=1 digits=5");
%ij.IJ.run("Image Sequence... ", "dir=["+pwd+"] format=TIFF name=[] start=1 digits=5");
ij.IJ.run("Image Sequence... ", "format=TIFF name=[] start=1 digits=5 save=["+pwd+"\00001.tif]");

MIJ.run("Close All"," ");
MIJ.closeAllWindows;  
MIJ.exit;

%% Scatter detect and Gaussian blur correction
% Read Registered Images
filetype='*.tif';
f=fullfile(pwd,filetype);
d=dir(f);
fileNumbers = cellfun(@(x) sscanf(x, '%d'), {d.name});
[~, sortIndex] = sort(fileNumbers);
d_sorted = d(sortIndex);
filename=transpose(fullfile({d_sorted(:).folder},{d_sorted(:).name}));
cd(folderPath);

%Thresholding and generating scatter removal mask
ImgSetNum=4;
Img=imread(filename{1});
figure, imshow(Img,[0,1000]);
BW = imbinarize(Img);
BW(:,:)=0;
parfor i=1:(numCycles*ImgSetNum)
    Img = imread(filename{i});
    %T1 = graythresh(I/3);
    T1 = graythresh(Img/2);
    if T1<0.035
        T1=0.035;   %incase graythresh goes wrong, default 0.02
    end
    BW1 = imbinarize(Img,T1);
    BWdfill = imfill(BW1, 'holes');
    %figure, imshow(BW1);
    se2 = strel('square', 2);
    se5= strel('disk',5);
    BW2=imdilate(imerode(BWdfill,se2),se5);
    BW2 = imfill(BW2, 'holes');
    BW = BW | BW2;
end

figure, imshow(BW);

%% dark edge cropping algorithm
for i = 1:totImg
    Img = imread(filename{i});
    % Sum pixel values along rows and columns
    rowSum = sum(Img, 2);
    colSum = sum(Img, 1);

    % Find the first and last indices where the sum is greater than zero
    firstRow = find(rowSum > (200*size(Img,2)), 1, 'first');   %200 is the threshold for min BG
    lastRow = find(rowSum > (200*size(Img,2)), 1, 'last');
    firstCol = find(colSum > (200*size(Img,1)), 1, 'first');
    lastCol = find(colSum > (200*size(Img,1)), 1, 'last');

    % Ensure safe cropping by taking the maximum firstRow and firstCol
    % and minimum lastRow and lastCol across all images
    if i == 1
        maxFirstRow = firstRow;
        maxFirstCol = firstCol;
        minLastRow = lastRow;
        minLastCol = lastCol;
    else
        maxFirstRow = max(maxFirstRow, firstRow);
        maxFirstCol = max(maxFirstCol, firstCol);
        minLastRow = min(minLastRow, lastRow);
        minLastCol = min(minLastCol, lastCol);
    end
end
crop2=[maxFirstCol maxFirstRow (minLastCol-maxFirstCol) (minLastRow-maxFirstRow)];
Img=imcrop(Img,crop2);
figure,imshow(Img,[0,1000])
Img_Seq = uint16(zeros([size(Img) totImg]));   %Store full Image Sequence
BW=imcrop(BW,crop2);

%% apply scatter removal mask, and Gaussian normalization
bk2(1:totImg)=0;
parfor i=1:totImg
    Img = imcrop(imread(filename{i}),crop2);
    %Img(BW)=mean(mean(Img));
    Img(BW)=0;
    Iblur = imgaussfilt(Img,50);     %default is 50, could to up to 100
    mean_Iblur = mean(mean(Iblur));
    Img_corr = uint16(double(Img) ./ double(Iblur) .* mean_Iblur);
    %Img_corr = imgaussfilt(Img_corr,0.7);   %Magic number, for find maxima later
    bk2(i)=mean(mean(Img_corr));
    Img_Seq(:,:,i)=Img_corr;
end
corr_factor2=max(bk2)./bk2;

%% Output final images
cd(folderPath);
outputFolder='2_processed_final';
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end
cd(outputFolder);
parfor ind_Img=1:totImg
    Img = Img_Seq(:,:,ind_Img);
    Img = uint16(double(Img).*corr_factor2(ind_Img));
    Img_Seq(:,:,ind_Img) = Img;
    filename_new{ind_Img} = fullfile(pwd, [char(ImgLabels{ind_Img}), imageExt]);
    imwrite(Img, filename_new{ind_Img});
end
cd(folderPath);

%% start the imageJ section
javaaddpath 'C:\Users\yujin\OneDrive\Documents\MATLAB\454 ROI\ij.jar';
javaaddpath 'C:\Users\yujin\OneDrive\Documents\MATLAB\454 ROI\mij.jar';
MIJ.start;

%find maxima and output results as CSV
csvLabels = {'645roi.csv', '590roi.csv', '525roi.csv', '445roi.csv'};
for i=1:ImgSetNum
    ij.IJ.open(filename_new{i});
    ij.IJ.run("Find Maxima...","prominence="+num2str(CutOff(i))+" output=[Point Selection]");
    ij.IJ.run("Set Measurements...","area mean min redirect=None decimal=3");
    ij.IJ.run("Measure");
    ij.IJ.saveAs("Results", fullfile(pwd, char(csvLabels{i})));
    ij.IJ.run("Clear Results");
    MIJ.run("Close All"," ");
    MIJ.closeAllWindows;  
end
MIJ.exit;
