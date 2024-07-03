#!/usr/bin/bash

export PATH=$PATH:`pwd`/tools/google-cloud-sdk/bin/

mkdir -p datasets
cd datasets
gsutil ls gs://sfr-unicontrol-data-research

mkdir -p MultiGen-20M/images && cd MultiGen-20M/images
echo "Start to download images from MultiGen-20M ..."
gsutil cp gs://sfr-unicontrol-data-research/dataset/images/aesthetics_6_25_plus_3m.zip .
gsutil cp gs://sfr-unicontrol-data-research/dataset/images/aesthetics_6_plus_0.zip .
gsutil cp gs://sfr-unicontrol-data-research/dataset/images/aesthetics_6_plus_1.zip .
gsutil cp gs://sfr-unicontrol-data-research/dataset/images/aesthetics_6_plus_2.zip .
gsutil cp gs://sfr-unicontrol-data-research/dataset/images/aesthetics_6_plus_3.zip .
cd ../..

conditions=("canny" "depth" "hed" "openpose" "segbase")
mkdir -p MultiGen-20M/conditions && cd MultiGen-20M/conditions
echo "Start to download conditions from MultiGen-20M ..."
for split in $(seq 0 1 14)
do
  for cond in ${conditions[@]}
  do 
      gsutil cp gs://sfr-unicontrol-data-research/dataset/group_${split}_${cond}.zip .
  done
done
cd ../..

mkdir -p MultiGen-20M/json_files && cd MultiGen-20M/json_files
echo "Start to download json_files from MultiGen-20M ..."
gsutil cp gs://sfr-unicontrol-data-research/dataset/aesthetics_plus_all_group_canny_all.json .
gsutil cp gs://sfr-unicontrol-data-research/dataset/aesthetics_plus_all_group_depth_all.json .
gsutil cp gs://sfr-unicontrol-data-research/dataset/aesthetics_plus_all_group_hed_all.json .
gsutil cp gs://sfr-unicontrol-data-research/dataset/aesthetics_plus_all_group_openpose_all.json .
gsutil cp gs://sfr-unicontrol-data-research/dataset/aesthetics_plus_all_group_segbase_all.json .
cd ../..


cd MultiGen-20M/images
echo "Start to unzip images ..."
unzip -p aesthetics_6_25_plus_3m.zip
unzip -p aesthetics_6_plus_0.zip
unzip -p aesthetics_6_plus_1.zip
unzip -p aesthetics_6_plus_2.zip
unzip -p aesthetics_6_plus_3.zip

rm aesthetics_6_25_plus_3m.zip
rm aesthetics_6_plus_0.zip
rm aesthetics_6_plus_1.zip
rm aesthetics_6_plus_2.zip
rm aesthetics_6_plus_3.zip
cd ../..

cd MultiGen-20M/conditions
echo "Start to unzip conditions ..."
for split in $(seq 0 1 14)
do
  for cond in ${conditions[@]}
  do
      unzip -p group_${split}_${cond}.zip
  done
done

for split in $(seq 0 1 14)
do
  for cond in ${conditions[@]}
  do
      rm group_${split}_${cond}.zip
  done
done

cd ../..

cd ..
