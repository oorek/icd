MASTER_ADDR="localhost" MASTER_PORT="20281" NODE_RANK="0" WORLD_SIZE=2 \
  ./sscd/train.py --nodes=1 --gpus=2 \
  --train_dataset_path=/nfs_shared_/MLVD/DISC/images/train \
  --val_dataset_path=/nfs_shared_/MLVD/DISC/val_images \
  --query_dataset_path /nfs_shared_/DISC/images/final_queries \
  --ref_dataset_path /nfs_shared_/DISC/images/references \
  --augmentations=ADVANCED --mixup=true \
  --batch_size 256 \
  --absolute_learning_rate 0.05 \
  --output_path=./ \
  --backbone=TV_RESNET18 \
  --ckpt=./weights/sscd_disc_mixup.torchvision.pt \
  --kd True \
  --workers 16
