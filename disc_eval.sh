sscd/disc_eval.py --disc_path /nfs_shared_/MLVD/DISC/images --gpus=2 \
  --output_path=./disc_eval \
  --size=288 --preserve_aspect_ratio=true \
  --backbone=CV_RESNET50 --dims=512 --model_state=weights/sscd_disc_mixup.classy.pt
