# ASDNet: Convolutional Neural Networks for Autism Spectrum Disorder Binary Calssification Using Resting State fMRI
## Usage

To run the models, you'll need to download this repository:
```
$ git clone 
$ cd ASDNet-3D
```
## Overview
# Download data
If you would like to download the ABIDE-Preprocessed dataset, you will need to clone the following repository and follow the steps they recommend there
```
$ git clone https://github.com/preprocessed-connectomes-project/abide.git
```
Otherwise, you will be able to find the database in json format in the following path
```
ASDNet-3D/data/data.json
ASDNet-3D/data/annotations.json
```
# Download pre-trained models

----------
| Method | Pretrained-models |
|---|---|
| 3D-ResNets-PyTorch |[https://drive.google.com/drive/folders/1xbYbZ7rpyjftI_KCk6YuL-XrfQDz7Yd4](https://drive.google.com/drive/folders/1xbYbZ7rpyjftI_KCk6YuL-XrfQDz7Yd4)|
| MedicalNet |[https://drive.google.com/file/d/13tnSvXY7oDIEloNFiGTsjUIYfS3g3BfG/view](https://drive.google.com/file/d/13tnSvXY7oDIEloNFiGTsjUIYfS3g3BfG/view)]|
| AD-Prediction |[https://pytorch.org/hub/pytorch_vision_resnet/](https://pytorch.org/hub/pytorch_vision_resnet/)|

Unzip and move files of MedicalNet
```
mv MedicalNet_pytorch_files2.zip MedicalNet/.
cd pretrain
unzip MedicalNet_pytorch_files2.zip
```

## Requirements

## Training and validation
In order to train and validate the model, you'll need to run the following lines:
# Training and validate with ASDNet-2D
```
$ cd ASDNet-2D
$ python train.py 
```
# Training and validate with ASDNet-3D
```
$ cd ASDNet-3D
$ python train.py 
```
## Method

----------
| Method | Original Link |
|---|---|
| 3D-ResNets-PyTorch |[https://github.com/cszn/DnCNN](https://github.com/kenshohara/3D-ResNets-PyTorch.git)|
| MedicalNet |[https://github.com/Tencent/MedicalNet](https://github.com/Tencent/MedicalNet.git)|
| AD-Prediction |[https://github.com/wangyirui/AD_Prediction](https://github.com/wangyirui/AD_Prediction.git)|

## Network architectures
# ASDNet-2D
# ASDNet-3D

## Testing
In order to test the model, you'll need to run the following lines:
# Testing with ASDNet-2D
```
$ cd ASDNet-2D
$ python main.py 
```
# Testing with ASDNet-3D
```
$ cd ASDNet-3D
$ python main.py -mode test
```
## Demo
To test the model only on a random image of the dataset, you'll need to run the following lines:
```
$ cd ASDNet-3D
$ python main.py -mode demo
```

## References
```
    @article{chen2019med3d,
        title={Med3D: Transfer Learning for 3D Medical Image Analysis},
        author={Chen, Sihong and Ma, Kai and Zheng, Yefeng},
        journal={arXiv preprint arXiv:1904.00625},
        year={2019}
    }
    
    @inproceedings{hara3dcnns,
        author={Kensho Hara and Hirokatsu Kataoka and Yutaka Satoh},
        title={Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet?},
        booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
        pages={6546--6555},
        year={2018},
    }

```
