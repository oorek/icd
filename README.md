

# Unsupervised Knowledge Distillation for Image Copy Detection on Lightweight Models.
This repository contains strong SSCD baseline implementation.
"[A Self-Supervised Descriptor for Image Copy Detection](https://cvpr2022.thecvf.com/)[CVPR 2022]".

## About this codebase

This implementation is built on [Pytorch Lightning](https://pytorchlightning.ai/),
with some components from [Classy Vision](https://classyvision.ai/).

## Datasets used
- DISC (Facebook Image Similarity Challenge 2021)
- Copydays

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

## Installation
- git repository download
```shell
$ cd ${WORKSPACE}
$ git clone https://github.com/oorek/icd.git
$ cd icd
```
- docker-compose.yml 수정
  - volume 경로, ports(8000, 22) 수정 필요
```shell
$ vim docker-compose.yml
version: '2.3'

services:
  main:
    container_name: grad_new
    ipc: host
    build:
      context: .
      dockerfile: Dockerfile
    runtime: nvidia
    env_file:
      - docker-compose-env/main.env
    restart: always
    tty: true
    stdin_open: true
    volumes:
      - /home/mmlab/hdd/grad/grad_new/grad_new:/user
      - /home/mmlab/hdd/dataset:/dataset
    ports:
      - "12100:8000"
      - "12122:22"
    expose:
      - "8080"

```
- 도커 컨테이너 업로드 후 패키지 설치
```shell
$ docker-compose up -d --build
$ docker attach ${CONTAINER NAME}
$ pip install -r requirements.txt
```
## Train
```shell
$ sh train.sh
MASTER_ADDR="localhost" MASTER_PORT="20281" NODE_RANK="0" WORLD_SIZE=2 \
  ./sscd/train.py --nodes=1 --gpus=2 \                                     # DDP 셋팅
  --train_dataset_path=/nfs_shared/DISC/images/train \                     # 학습 데이터셋 경로
  --val_dataset_path=/nfs_shared/DISC/val_images \                         # 검증 데이터셋 경로
  --query_dataset_path /nfs_shared/DISC/images/final_queries \             # Query 데이터셋 경로
  --ref_dataset_path /nfs_shared/DISC/images/references \                  # Reference 데이터셋 경로
  --augmentations=ADVANCED --mixup=false \                                 # 이미지 증강
  --batch_size 256 \                                                       # 배치사이즈 
  --base_learning_rate 1 \                                                 # 러닝레이트
  --output_path=./ \                                                       # 모델 저장 경로
  --backbone=TV_EFFICIENTNET_B0 \                                          # stuent 백본 아키텍쳐 설정
  --ckpt=./weights/sscd_disc_mixup.torchvision.pt \                        # 티쳐 모델 체크포인트
  --workers 16                                                             # CPU worker 수
```
## Evaluation
```shell
$ sh disc_eval.sh
sscd/disc_eval.py
--disc_path /dataset/DISC \                                                    # 데이터셋 경로
--gpus=2 \                                                                     # gpu 수
--output_path=./disc_eval/simclr \                                             # 아웃풋 경로
--size=288                                                                     # 이미지 사이즈
--preserve_aspect_ratio=false \                                                # 이미지 비율 설정
--backbone=CV_RESNET50                                                         # 백본 아키텍쳐
--dims=512                                                                     # 피쳐 차원
--checkpoint=lightning_logs/version_26/checkpoints/epoch\=81-step\=472625.ckpt # 체크포인트 
```
## SSCD Pretrained Models

| name                   | dataset  | trunk           | augmentations    | dimensions | classy vision                                                                               | torchvision                                                                                      | torchscript                                                                                      |
|------------------------|----------|-----------------|------------------|------------|---------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| sscd_disc_blur         | DISC     | ResNet50        | strong blur      | 512        | [link](https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_blur.classy.pt)         | [link](https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_blur.torchvision.pt)         | [link](https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_blur.torchscript.pt)         |
| sscd_disc_advanced     | DISC     | ResNet50        | advanced         | 512        | [link](https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_advanced.classy.pt)     | [link](https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_advanced.torchvision.pt)     | [link](https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_advanced.torchscript.pt)     |
| sscd_disc_mixup        | DISC     | ResNet50        | advanced + mixup | 512        | [link](https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_mixup.classy.pt)        | [link](https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_mixup.torchvision.pt)        | [link](https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_mixup.torchscript.pt)        |
| sscd_disc_large        | DISC     | ResNeXt101 32x4 | advanced + mixup | 1024       | [link](https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_large.classy.pt)        |                                                                                                  | [link](https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_large.torchscript.pt)        |
| sscd_imagenet_blur     | ImageNet | ResNet50        | strong blur      | 512        | [link](https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_imagenet_blur.classy.pt)     | [link](https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_imagenet_blur.torchvision.pt)     | [link](https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_imagenet_blur.torchscript.pt)     |
| sscd_imagenet_advanced | ImageNet | ResNet50        | advanced         | 512        | [link](https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_imagenet_advanced.classy.pt) | [link](https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_imagenet_advanced.torchvision.pt) | [link](https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_imagenet_advanced.torchscript.pt) |
| sscd_imagenet_mixup    | ImageNet | ResNet50        | advanced + mixup | 512        | [link](https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_imagenet_mixup.classy.pt)    | [link](https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_imagenet_mixup.torchvision.pt)    | [link](https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_imagenet_mixup.torchscript.pt)    |

## DINO Pretrained Models

<table>
  <tr>
    <th>arch</th>
    <th>params</th>
    <th>k-nn</th>
    <th>linear</th>
    <th colspan="6">download</th>
  </tr>
  <tr>
    <td>ViT-S/16</td>
    <td>21M</td>
    <td>74.5%</td>
    <td>77.0%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth">backbone only</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain_full_checkpoint.pth">full ckpt</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deits16.onnx">onnx</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/args.txt">args</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain_log.txt">logs</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain_eval_linear_log.txt">eval logs</a></td>
  </tr>
  <tr>
    <td>ViT-S/8</td>
    <td>21M</td>
    <td>78.3%</td>
    <td>79.7%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth">backbone only</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain_full_checkpoint.pth">full ckpt</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deits8.onnx">onnx</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/args.txt">args</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain_log.txt">logs</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain_eval_linear_log.txt">eval logs</a></td>
  </tr>
  <tr>
    <td>ViT-B/16</td>
    <td>85M</td>
    <td>76.1%</td>
    <td>78.2%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth">backbone only</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain_full_checkpoint.pth">full ckpt</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitb16.onnx">onnx</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/args.txt">args</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain_log.txt">logs</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain_eval_linear_log.txt">eval logs</a></td>
  </tr>
  <tr>
    <td>ViT-B/8</td>
    <td>85M</td>
    <td>77.4%</td>
    <td>80.1%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth">backbone only</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain_full_checkpoint.pth">full ckpt</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitb8.onnx">onnx</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/args.txt">args</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain_log.txt">logs</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain_eval_linear_log.txt">eval logs</a></td>
  </tr>
  <tr>
    <td>ResNet-50</td>
    <td>23M</td>
    <td>67.5%</td>
    <td>75.3%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth">backbone only</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain_full_checkpoint.pth">full ckpt</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50.onnx">onnx</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/args.txt">args</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain_log.txt">logs</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain_eval_linear_log.txt">eval logs</a></td>
  </tr>
</table>

## DINO-V2 Pretrained Models

<table style="margin: auto">
  <tr>
    <th>model</th>
    <th># of<br />params</th>
    <th>ImageNet<br />k-NN</th>
    <th>ImageNet<br />linear</th>
    <th>download</th>
  </tr>
  <tr>
    <td>ViT-S/14 distilled</td>
    <td align="right">21 M</td>
    <td align="right">79.0%</td>
    <td align="right">81.1%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth">backbone only</a></td>
  </tr>
  <tr>
    <td>ViT-B/14 distilled</td>
    <td align="right">86 M</td>
    <td align="right">82.1%</td>
    <td align="right">84.5%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth">backbone only</a></td>
  </tr>
  <tr>
    <td>ViT-L/14 distilled</td>
    <td align="right">300 M</td>
    <td align="right">83.5%</td>
    <td align="right">86.3%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth">backbone only</a></td>
  </tr>
  <tr>
    <td>ViT-g/14</td>
    <td align="right">1,100 M</td>
    <td align="right">83.5%</td>
    <td align="right">86.5%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth">backbone only</a></td>
  </tr>
</table>

### Pretrained models via PyTorch Hub

Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install PyTorch (the only required dependency for loading the model). Installing PyTorch with CUDA support is strongly recommended.


## Citation
```
@article{pizzi2022self,
  title={A Self-Supervised Descriptor for Image Copy Detection},
  author={Pizzi, Ed and Roy, Sreya Dutta and Ravindra, Sugosh Nagavara and Goyal, Priya and Douze, Matthijs},
  journal={Proc. CVPR},
  year={2022}
}
```
