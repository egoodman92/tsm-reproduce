# Surgery Action Recognition

## Overview
This repository consists of the PyTorch library used for action recognition on the Youtube Surgery dataset.


### Setup

1) Launch an AMI with CUDA  

```
Deep Learning AMI (Ubuntu 18.04) Version 30.0 - ami-029510cec6d69f121
```


2) Clone the repo and install dependencies

```
git clone git@github.com:yeung-lab/surgery-action-recognition.git
cd surgery-action-recognition
conda activate python3
pip install -r requirements.txt
```

3) Download TSM code and pretrained models

```
git clone git@github.com:mit-han-lab/temporal-shift-module.git
wget -P pretrained https://file.lzhu.me/projects/tsm/models/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e100_dense_nl.pth
```

4) Download videos

```
mkdir data; mkdir data/videos/;
wget -P data https://marvl-surgery.s3.amazonaws.com/videos.zip
unzip data/videos.zip -d data/videos/
```

5) Download annotations
```
wget -P data https://marvl-surgery.s3.amazonaws.com/v0.5.0-anns-5sec.csv
wget -P data https://marvl-surgery.s3.amazonaws.com/train.csv
wget -P data https://marvl-surgery.s3.amazonaws.com/test.csv
```

6) Run training
```
mkdir runs
python train.py --model-name TSM --anns-path data/v0.5.0-anns-5sec.csv --exp-dir runs
```


### Inference on pretrained model 
1) Download TSM code and pretrained models

```
wget -P pretrained https://marvl-surgery.s3.amazonaws.com/tsm-kinetics-pretrain.pt
```

2) Run inference on video

```
python inference.py --model-path=pretrained/tsm-kinetics-pretrain.pt --video-ids=0EeZIRDKYO4 --results-path=results.csv
```

## Data
The Youtube Surgery dataset consists of ~390 open-surgery videos curated from YouTube. A subset of these videos were annotated for three surgical actions: cutting, tying, and suturing.



