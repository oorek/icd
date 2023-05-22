

# Unsupervised Knowledge Distillation for Image Copy Detection on Lightweight models.
This repository contains strong SSCD baseline[A Self-Supervised Descriptor for Image Copy Detection](https://arxiv.org/abs/2202.10261) implementation.

This is the open-source codebase for
"[A Self-Supervised Descriptor for Image Copy Detection](https://arxiv.org/abs/2202.10261)",
recently accepted to [CVPR 2022](https://cvpr2022.thecvf.com/).

## About this codebase

This implementation is built on [Pytorch Lightning](https://pytorchlightning.ai/),
with some components from [Classy Vision](https://classyvision.ai/).

## Unsupervised Knowledge Distillation
- similarity distillation 
- KoLeo regularization
- DirectCLR contrastive learning

## Teacher Models(Pretrained)
- SSCD: ResNet-50
- DINO: ViT-B/16

## Student Models
- ResNet-18
- EfficientNet-B0
- MobileNet-V3-Large
- ViT-S
- MobileViT


## SSCD Pretrained models

| name                   | dataset  | trunk           | augmentations    | dimensions | classy vision                                                                               | torchvision                                                                                      | torchscript                                                                                      |
|------------------------|----------|-----------------|------------------|------------|---------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| sscd_disc_blur         | DISC     | ResNet50        | strong blur      | 512        | [link](https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_blur.classy.pt)         | [link](https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_blur.torchvision.pt)         | [link](https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_blur.torchscript.pt)         |
| sscd_disc_advanced     | DISC     | ResNet50        | advanced         | 512        | [link](https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_advanced.classy.pt)     | [link](https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_advanced.torchvision.pt)     | [link](https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_advanced.torchscript.pt)     |
| sscd_disc_mixup        | DISC     | ResNet50        | advanced + mixup | 512        | [link](https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_mixup.classy.pt)        | [link](https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_mixup.torchvision.pt)        | [link](https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_mixup.torchscript.pt)        |
| sscd_disc_large        | DISC     | ResNeXt101 32x4 | advanced + mixup | 1024       | [link](https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_large.classy.pt)        |                                                                                                  | [link](https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_large.torchscript.pt)        |
| sscd_imagenet_blur     | ImageNet | ResNet50        | strong blur      | 512        | [link](https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_imagenet_blur.classy.pt)     | [link](https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_imagenet_blur.torchvision.pt)     | [link](https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_imagenet_blur.torchscript.pt)     |
| sscd_imagenet_advanced | ImageNet | ResNet50        | advanced         | 512        | [link](https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_imagenet_advanced.classy.pt) | [link](https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_imagenet_advanced.torchvision.pt) | [link](https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_imagenet_advanced.torchscript.pt) |
| sscd_imagenet_mixup    | ImageNet | ResNet50        | advanced + mixup | 512        | [link](https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_imagenet_mixup.classy.pt)    | [link](https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_imagenet_mixup.torchvision.pt)    | [link](https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_imagenet_mixup.torchscript.pt)    |

## Citation
```
@article{pizzi2022self,
  title={A Self-Supervised Descriptor for Image Copy Detection},
  author={Pizzi, Ed and Roy, Sreya Dutta and Ravindra, Sugosh Nagavara and Goyal, Priya and Douze, Matthijs},
  journal={Proc. CVPR},
  year={2022}
}
```
