# TaWQ
The official implementation of TaWQ

## Requirements

pytorch
spikingjelly==0.0.0.12
timm==0.6.12

## ImageNet

train with 8 GPUs or NPUs

python -m torch.distributed.launch --nproc_per_node=8 main.py --accum_iter 1 --batch_size 64 --blr 6e-4 --model qkformer_tawq_10_768 --output_dir ./qkformer_tawq_10_768 --num_workers 79 --data_path ../Data_ImageNet2012/ --epochs 200

## CIFAR100

python train.py