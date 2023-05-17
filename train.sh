MASTER_ADDR="localhost" MASTER_PORT="20281" NODE_RANK="0" WORLD_SIZE=2 \
  ./sscd/train.py --nodes=1 --gpus=2 \
  --train_dataset_path=/nfs_shared_/MLVD/DISC/images/final_queries \
  --val_dataset_path=/nfs_shared_/MLVD/DISC/val_images \
  --augmentations=ADVANCED --mixup=true \
  --batch_size 512 \
  --base_learning_rate 3 \
  --output_path=./ \
  --ckpt=./weights/sscd_disc_blur.torchvision.pt
