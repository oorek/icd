sscd/copydays_eval.py --gpus=2 --copydays_path /nfs_shared_/MLVD/cd10k/copydays \
  --distractor_path /nfs_shared_/MLVD/cd10k/distractors \
  --codec_train_path /nfs_shared_/MLVD/cd10k/whitening \
  --output_path=./ \
  --backbone=CV_RESNET50 --dims=512 \
  --model_state=sscd_disc_mixup.classy.pt \
  --size=288 --preserve_aspect_ratio=true \
  --codecs="PCAW512,L2norm,Flat;PCA512,L2norm,Flat"