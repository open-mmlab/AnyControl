#!/bin/bash

mkdir -p datasets/MSCOCO && cd datasets/MSCOCO
echo "Start to download images and annotations http://images.cocodataset.org ..."
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
echo "Finish downloading."

echo "Start to unzip files ..."
unzip train2017.zip
unzip annotations_trainval2017.zip 
delete train2017.zip
delele annotations_trainval2017.zip
echo "Finish unzip."

cd ../..
