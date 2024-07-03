# Prepare datasets

We train AnyControl on MultiGen-20M and synthetic unaligned data. Please prepare training data following the instructions.


## Prerequisites

- (Optional) Install [gsutil](https://cloud.google.com/storage/docs/gsutil_install?hl=en#linux). Skip this step if you prefer to download [MultiGen-20M](https://console.cloud.google.com/storage/browser/sfr-unicontrol-data-research/dataset) on a browser.

```shell
mkdir tools && cd tools
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz
tar -xf google-cloud-cli-linux-x86_64.tar.gz
./google-cloud-sdk/install.sh
./google-cloud-sdk/bin/gcloud init
rm google-cloud-cli-linux-x86_64.tar.gz
cd ..
```

- (Optional) Install [awscli](https://pypi.org/project/awscli/). Skip this step if you have already downloaded [Open Images](https://github.com/cvdfoundation/open-images-dataset) or prefer to download in other ways.

```shell
pip install awscli
```

- Install [PowerPaint](https://github.com/open-mmlab/PowerPaint) and download `PowerPaint-v1` model. Actually we have include PowerPaint code in `data/PowerPaint` with a commit id of `e037c3f2ff62e3fc55072ef91a891c85f419a0cb`. Please go to [PowerPaint](https://github.com/open-mmlab/PowerPaint.git) for the latest version. To download `PowerPaint-v1` model, do

```
cd data

conda install git-lfs
git lfs install

git clone https://huggingface.co/JunhaoZhuang/PowerPaint-v1

cd ..
```



## MultiGen-20M

**Step 0. Download [MultiGen-20M](https://console.cloud.google.com/storage/browser/sfr-unicontrol-data-research/dataset).**

```shell
cd AnyControl

sh scripts/download_multigen.sh
```

The folder structure of `AnyControl/datasets/MultiGen-20M` should be

```none
MultiGen-20M
├── conditions
│   ├── group_0_canny
│   ├── group_0_depth
│   ├── ...
├── images
│   ├── aesthetics_6_plus_0
│   ├── aesthetics_6_plus_1
│   ├── ...
├── json_files
│   ├── aesthetics_plus_all_group_canny_all.json
│   ├── aesthetics_plus_all_group_depth_all.json
│   ├── ...

```

**Step 1. Generate `.jsonl` file.**

```shell
python scripts/genereate_jsonl.py --dataset MultiGen-20M
```

## COCO

**Step 0. Download [COCO](https://cocodataset.org/#download) dataset.**

```shell
cd AnyControl

sh scripts/download_coco.sh
```

The folder structure of `AnyControl/datasets/MSCOCO` should be

```none
MSCOCO
├── train2017
├── annotations
│   ├── instances_train2017.json
│   ├── captions_train2017.json
│   ├── ...
```

**Step 1. Inpaiting with PowerPaint.**

```shell
python scripts/prepare_unaligned_coco.py
```

**Step 2. Extract multiple conditions.**

```shell
python scripts/prepare_conditions --dataset COCO
```

**Step 3. Generate `.jsonl` file.**

```shell
python scripts/genereate_jsonl.py --dataset COCO
```

## Open Images

**Step 0. Download [Open Images](https://storage.googleapis.com/openimages/web/index.html) dataset.** 

```shell
cd AnyControl

sh scripts/download_openimages.sh
```

The folder structure of `AnyControl/datasets/OpenImages` should be

```none
OpenImages
├── train
├── train-annotations-object-segmentation.csv
├── oidv7-class-descriptions-boxable.csv
├── ...
```

**Step 1. Generate captions for Open Images data with [blip2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2).**

```shell
pip install salesforce-lavis

python scripts/prepare_openimages_captions.py
```

**Step 2. Inpainting with PowerPaint.**

```shell
python scripts/prepare_unaligned_openimages.py
```

**Step 3. Extract multiple conditions.** 

```shell
python scripts/prepare_conditions --dataset OpenImages
```

**Step 4. Generate `.jsonl` file.**

```shell
python scripts/genereate_jsonl.py --dataset OpenImages
```
