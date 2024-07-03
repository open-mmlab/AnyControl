#!/bin/bash

echo "Start to download images from s3://open-images-dataset ..."
mkdir -p datasets/OpenImages/train
echo "aws s3 --no-sign-request sync s3://open-images-dataset/train datasets/OpenImages/train"
aws s3 --no-sign-request sync s3://open-images-dataset/train datasets/OpenImages/train 
echo "Images downloading finished."


subfolders=("0" "1" "2" "3" "4" "5" "6" "7" "8" "9" "a" "b" "c" "d" "e" "f")
echo "Start to download training masks from https://storage.googleapis.com ..."
mkdir -p datasets/OpenImages/train-masks && cd datasets/OpenImages/train-masks
for split in ${subfolders[@]} 
do 
    echo "wget https://storage.googleapis.com/openimages/v5/train-masks/train-masks-${split}.zip"
    wget https://storage.googleapis.com/openimages/v5/train-masks/train-masks-${split}.zip 
done
echo "Training masks downloading finished."

echo "Start to unzip training masks ..."
for split in $(subfolders[@]) 
do 
    echo "unzip -qq train-masks-${split}.zip"
    unzip -qq train-masks-${split}.zip 
done

for split in ${subfolders[@]} 
do 
    rm train-masks-${split}.zip 
done
echo "Unzip finished."



echo "Start to download annotations ..."
cd ..
echo "wget https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv"
wget https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv
echo "wget https://storage.googleapis.com/openimages/v5/train-annotations-object-segmentation.csv"
wget https://storage.googleapis.com/openimages/v5/train-annotations-object-segmentation.csv
echo "Annotations downloading finished."

cd ../..
