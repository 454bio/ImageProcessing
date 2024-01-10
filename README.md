# ImageProcessing

ClusterSeqIP_v1.py will pre-process the 1_original folder images from the Seq run and output processed images in a folder that are ready for ImageJ registration: 

What it does: Image rename, filtering, binning, background normalization, magnification correction 

Installing: 

Python 3.9 and above

Libarary: numpy, roifile, matplotlib, opencv, pandas, scipy, scikit-learn, scikit-image

Executing Program:
```bash
& C:/Users/.../python.exe ".../ClusterSeqIP_v1.py" -i "/path/to/unprocessed/data"
```

## Color Transformation:

Extract basic metrics from ROIs
```bash
extract_roiset_metrics_to_csv.py \
-i /mnt/nas_share/GoogleData/InstrumentData/MK27_02/20230504_1510_S0096_0001/raws/ \
-r /mnt/nas_share/GoogleData/InstrumentData/MK27_02/20230504_1510_S0096_0001/Analysis/S0096_RoiSet.zip \
-o S0096.csv
extract_roiset_metrics_to_csv.py \
-i /mnt/nas_share/GoogleData/InstrumentData/MK27_02/20230504_1540_S0097_0001/raws/ \
-r /mnt/nas_share/GoogleData/InstrumentData/MK27_02/20230504_1540_S0097_0001/Analysis/S0097_RoiSet.zip \
-o S0097.csv
extract_roiset_metrics_to_csv.py \
-i /mnt/nas_share/GoogleData/InstrumentData/MK27_02/20230504_1607_S0098_0001/raws/ \
-r /mnt/nas_share/GoogleData/InstrumentData/MK27_02/20230504_1607_S0098_0001/Analysis/S0098_RoiSet.zip \
-o S0098.csv
extract_roiset_metrics_to_csv.py \
-i /mnt/nas_share/GoogleData/InstrumentData/MK27_02/20230504_1631_S0099_0001/raws/ \
-r /mnt/nas_share/GoogleData/InstrumentData/MK27_02/20230504_1631_S0099_0001/Analysis/S0099_RoiSet5.zip \
-o S0099.csv
plot_roiset_run_comparison.py -h

plot_roiset_run_comparison.py -i S0096.csv S0097.csv S0098.csv S0099.csv -o orig.jpg

plot_roiset_run_comparison.py -i S0096.csv S0097.csv S0098.csv S0099.csv -o normalized.jpg -n
```

Create browser based graph with spot trajectories
```bash
plot_spot_trajectories.py -i analysis/metrics.csv
plot_spot_trajectories.py -i analysis/metrics.csv -c G445 G525 R590 B445
plot_spot_trajectories.py -i analysis/metrics.csv -s S1 S2 -c G445 G525 R590 B445
plot_spot_trajectories.py -i analysis/metrics.csv -o analysis/trajectories.png
```

Create triangle graphs
```bash
extract_roiset_pixel_data.py -i . -r RoiSet.zip -o RoiSet.csv -e 7 -n 500

triangle_graph.py -i RoiSet.csv -o triangle.png -g -m 20 33.3 12

triangle_graph.py -i RoiSet.csv -o triangle.png -g -c G445 G525 R590 B445

triangle_graph.py -i RoiSet.csv -o triangle.png -g -s A C G T BG S1 S2

triangle_graph.py -i RoiSet.csv -o triangle.png -g -c G445 G525 R590 B445 -s A C G T BG S1 S2
```

Create basic basecaller graph
```bash
extract_roiset_pixel_data.py -e 7 -n 500 \
-i /mnt/nas_share/GoogleData/InstrumentData/MK27_02/20230517_1458_S0115_0001/raws \
-r /mnt/nas_share/GoogleData/InstrumentData/MK27_02/20230517_1458_S0115_0001/analysis/RoiSetJRC4_28spotsACGT.zip \
-o /tmp/roi_pixel_data.csv

color_transformation.py \
-i /mnt/nas_share/GoogleData/InstrumentData/MK27_02/20230517_1458_S0115_0001/raws \
-r /mnt/nas_share/GoogleData/InstrumentData/MK27_02/20230517_1458_S0115_0001/analysis/RoiSetJRC4_28spotsACGT.zip \
-p /mnt/nas_share/GoogleData/InstrumentData/MK27_02/20230517_1458_S0115_0001/analysis/roi_pixel_data.csv \
-o /tmp

cd /mnt/nas_share/GoogleData/InstrumentData/MK27_02/20230517_1458_S0115_0001

extract_roiset_pixel_data.py \
-e 7 \
-n 500 \
-i raws \
-r analysis/RoiSetJRC4_28spotsACGT.zip \
-o /tmp/spot_pixel_data.csv

color_transformation.py \
-i raws \
-r analysis/RoiSetJRC4_28spotsACGT.zip \
-p analysis/spot_pixel_data.csv \
-c G445 G525 R590 B445 \
-s A C G T BG S1 S2 \
-o /tmp
```

## Dephasing Correction and Basecall:

Run default_analysis.sh file in terminal

with “color_transformed_spots.csv” add to the directory path
