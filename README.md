# ASDNet: Convolutional Neural Networks for Autism Spectrum Disorder Binary Classification Using Resting State fMRI
## Usage

To run the models, you'll need to download this repository:
```
$ git clone 
$ cd ASDNet
```
## Overview
# Download data
If you would like to download the ABIDE-Preprocessed dataset, you will need to clone the following repository and follow the steps they recommend there
```
$ git clone https://github.com/preprocessed-connectomes-project/abide.git
```
Otherwise, you will be able to find the database in json format in the following path
```
/home/lvbellon/ASDNet/data/data.json
/home/lvbellon/ASDNet/data/annotations.json
```
# Download pre-trained models

----------
| Method | Pretrained-models |
|---|---|
| 3D-ResNets-PyTorch |[https://drive.google.com/drive/folders/1xbYbZ7rpyjftI_KCk6YuL-XrfQDz7Yd4](https://drive.google.com/drive/folders/1xbYbZ7rpyjftI_KCk6YuL-XrfQDz7Yd4)|
| MedicalNet |[https://drive.google.com/file/d/13tnSvXY7oDIEloNFiGTsjUIYfS3g3BfG/view](https://drive.google.com/file/d/13tnSvXY7oDIEloNFiGTsjUIYfS3g3BfG/view)|
| AD-Prediction |[https://pytorch.org/hub/pytorch_vision_resnet/](https://pytorch.org/hub/pytorch_vision_resnet/)|
| ASDNet |[https://drive.google.com/drive/folders/1d74NKw6Md4difhD4p8HPeGXvXr2-b_QK?usp=sharing](https://drive.google.com/drive/folders/1d74NKw6Md4difhD4p8HPeGXvXr2-b_QK?usp=sharing)|

Unzip and move files of MedicalNet
```
mv MedicalNet_pytorch_files2.zip MedicalNet/.
cd pretrain
unzip MedicalNet_pytorch_files2.zip
```

## Requirements
```
pip install monai
```
## Usage (Train and validation)
In order to train and validate the model, you'll need to run the following lines:
# ASDNet2D
```
$ python train_ASDNet2D.py --model resnet50 --pretrained-model --batch-size 64 --test-batch-size 128 --n-class 2 --epochs 15 --lr 0.0001 --optimizer Adam --momentum 0.99 --weight_decay 0 --alpha -1 --gamma 0.5 --loss BCE --act-loss --reduction mean --no-cuda False --seed 1 --log-interval 10 --save model_ASDNet2D.pt 
```
# ASDNet3D
```
$ python train_ASDNet3D.py 
```
## Method

----------
| Method | Original Link |
|---|---|
| 3D-ResNets-PyTorch |[https://github.com/cszn/DnCNN](https://github.com/kenshohara/3D-ResNets-PyTorch.git)|
| MedicalNet |[https://github.com/Tencent/MedicalNet](https://github.com/Tencent/MedicalNet.git)|
| AD-Prediction |[https://github.com/wangyirui/AD_Prediction](https://github.com/wangyirui/AD_Prediction.git)|
| MONAI |[https://github.com/Project-MONAI/MONAI](https://github.com/Project-MONAI/MONAI.git)|

## Network architectures
# ASDNet-2D
![My Image](/assets/ASDNet2D.png)
# ASDNet-3D

## Usage (Test)
----------
| Method | model_zoo |
|---|---|
| ASDNet-2D |model_resnet50_with_pre-training.pt model_resnet50_without_pre-training.pt|
| ASDNet-3D |model_monai_without_pre-training_BCE.pt model_monai_with_pre-training.pt model_monai_with_pre-training_CE.pt mod el_monai_with_pre-training_sfl.pt model_monai_without_pre-training_BCE_4layers.pt model_monai_without_pre-training_BCE_6layers.pt model_monai_without_pre-training_CB_6layers.pt|


In order to test the model, you'll need to run the following lines:
# ASDNet2D
```
$ python main_ASDNet2D.py --model resnet50 --pretrained-model --batch-size 128 --n-class 2 --epochs 15 --lr 0.0001 --optimizer Adam --momentum 0.99 --weight_decay 0 --alpha -1 --gamma 0.5 --loss BCE --act-loss --reduction mean --no-cuda False --seed 1 --log-interval 10 --save model_ASDNet2D.pt
```
# ASDNet3D
```
$ python main_ASDNet3D.py -mode test
```
## Demo
To test the model only on a random image of the dataset, you'll need to run the following lines:
```
$ python main_ASDNet3D.py -mode demo
```

## References
```BibTex
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
    
@article{klein2012101,
      title={101 labeled brain images and a consistent human cortical labeling protocol},
      author={Klein, Arno and Tourville, Jason},
      journal={Frontiers in neuroscience},
      volume={6},
      pages={171},
      year={2012},
      publisher={Frontiers}
}

```
