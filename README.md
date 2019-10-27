# bLVNet

This repository holds the code and models for our paper,

Quanfu Fan*, Chun-Fu (Richard) Chen*, Hilde Kuehne, Marco Pistoia, David Cox, "More Is Less: Learning Efficient Video Representations by Temporal Aggregation Modules"

If you use the code and models from this repo, please cite our work. Thanks!
```
@incollection{
    fan2019blvnet,
    title={{More Is Less: Learning Efficient Video Representations by Temporal Aggregation Modules}},
    author={Quanfu Fan and Chun-Fu (Ricarhd) Chen and Hilde Kuehne and Marco Pistoia and David Cox},
    booktitle={Advances in Neural Information Processing Systems 33},
    year={2019}
}
```

## Requirements

```
pip install -r requirement.txt
```


## Pretrained Models on Something-Something
The results below (top-1 accuracy) are reported under the single-crop and single-clip setting.

### V1

| Name | Top-1 Val Acc. |
|------|----------------|
|[bLVNet-TAM-50-a2-b4-f8x2](https://ibm.box.com/v/st2stv1-bLVNet-TAM-50-f8x2) | 46.4 |  
|[bLVNet-TAM-50-a2-b4-f16x2](https://ibm.box.com/v/st2stv1-bLVNet-TAM-50-f16x2) | 48.4 | 
|[bLVNet-TAM-101-a2-b4-f8x2](https://ibm.box.com/v/st2stv1-bLVNet-TAM-101-f8x2) | 47.8 | 
|[bLVNet-TAM-101-a2-b4-f16x2](https://ibm.box.com/v/st2stv1-bLVNet-TAM-101-f16x2) | 49.6 |
|[bLVNet-TAM-101-a2-b4-f24x2](https://ibm.box.com/v/st2stv1-bLVNet-TAM-101-f24x2) | 52.2 |
|[bLVNet-TAM-101-a2-b4-f32x2](https://ibm.box.com/v/st2stv1-bLVNet-TAM-101-f32x2) | 53.1|

### V2

| Name | Top-1 Val Acc. |
|------|------------|
|[bLVNet-TAM-50-a2-b4-f8x2](https://ibm.box.com/v/st2stv2-bLVNet-TAM-50-f8x2) | 59.1 |
|[bLVNet-TAM-50-a2-b4-f16x2](https://ibm.box.com/v/st2stv2-bLVNet-TAM-50-f16x2) | 61.7 | 
|[bLVNet-TAM-101-a2-b4-f8x2](https://ibm.box.com/v/st2stv2-bLVNet-TAM-101-f8x2) | 60.2 |
|[bLVNet-TAM-101-a2-b4-f16x2](https://ibm.box.com/v/st2stv2-bLVNet-TAM-101-f16x2) | 61.9 |
|[bLVNet-TAM-101-a2-b4-f24x2](https://ibm.box.com/v/st2stv2-bLVNet-TAM-101-f24x2) | 64.0 |
|[bLVNet-TAM-101-a2-b4-f32x2](https://ibm.box.com/v/st2stv2-bLVNet-TAM-101-f32x2) | 65.2 |
 
## Data Preparation
We provide two scripts in the folder `tools` for prepare input data for model training. The scripts sample an image sequence from a video and then resize each image to have its shorter side to be `256` while keeping the aspect ratio of the image.
You may need to set up `folder_root` accordingly to assure the extraction works correctly.
 
## Training
To reproduce the results in our paper, the pretrained models of bLNet are required and they are available at [here](https://github.com/IBM/BigLittleNet).

With the pretrained models placed in the folder `pretrained`, the following script can be used to train
a bLVNet-101-TAM-a2-b4-f8x2 model on Something-Something V2

```
python3 train.py --datadir /path/to/folder \
--dataset st2stv2 -d 101 --groups 16  \ 
--logdir /path/to/logdir --lr 0.01 -b 64 --dropout 0.5 -j 36 \
--blending_frames 3 --epochs 50 --disable_scaleup --imagenet_blnet_pretrained
```

## Test 

First download the models and put them in the `pretrained` folder. Then follow the example below to evaluate a model. 
Example: evaluating the bLVNet-101-TAM-a2-b4-f8x2 model on Something-Something V2
```
python3 test.py --datadir /path/to/folder --dataset st2stv2 -d 101 --groups 16 \ 
--alpha 2 --beta 4 --evaluate --pretrained --dataset --disable_scaleup \
--logdir /path/to/logdir
```

You can add `num_crops` and `num_clips` arguments to perform multi-crops and multi-clips evaluation to video-level accuracy.

Please feel free to let us know if you encounter any issue when using our code and models.

