

# Weakly-supervised Knowledge Distillation for Image Copy Detection
This repository contains strong SSCD baseline implementation

This is the open-source codebase for
"[A Self-Supervised Descriptor for Image Copy Detection](https://arxiv.org/abs/2202.10261)",
recently accepted to [CVPR 2022](https://cvpr2022.thecvf.com/).

## About this codebase

This implementation is built on [Pytorch Lightning](https://pytorchlightning.ai/),
with some components from [Classy Vision](https://classyvision.ai/).


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

We recommend `sscd_disc_mixup` (ResNet50) as a default SSCD model,
especially when comparing to other standard ResNet50 models,
and `sscd_disc_large` (ResNeXt101) as a higher accuracy alternative
using a bit more compute.

Classy Vision and Torchvision use different default cardinality settings
for ResNeXt101. We do not provide a Torchvision version of the
`sscd_disc_large` model for this reason.

## Installation

## Inference using SSCD models

This section describes how to use pretrained SSCD models for inference.
To perform inference for DISC and Copydays evaluations, see
[Evaluation](docs/Evaluation.md).

#### Descriptor post-processing
L2 Norm -> Centering/Whitening -> Score Normalization (but has no effect on ranking metrics)
Score Normalization improve global accuracy.


## Reproducing evaluation results

To reproduce evaluation results, see [Evaluation](docs/Evaluation.md).

## Training SSCD models

For information on how to train SSCD models, see 
[Training](docs/Training.md).


## Citation
```
@article{pizzi2022self,
  title={A Self-Supervised Descriptor for Image Copy Detection},
  author={Pizzi, Ed and Roy, Sreya Dutta and Ravindra, Sugosh Nagavara and Goyal, Priya and Douze, Matthijs},
  journal={Proc. CVPR},
  year={2022}
}
```
