#!/bin/bash
#SBATCH -N 1
#SBATCH -p cox
#SBATCH -n 4
#SBATCH --mem 32Gb
#SBATCH --gres=gpu:1
#SBATCH -t 1-00:00
#SBATCH -o outfiles/%j.out
#SBATCH -e outfiles/%j.err
module load python/3.6.3-fasrc01
module load cuda
module load cudnn
source activate vit
python ViT-pytorch/train.py --name wordassoc_prototype_ViT-B_16 --dataset word_assoc --model_type ViT-B_16 
python3 ViT-pytorch/train.py --name wordassoc_prototype_ViT-B_16 --dataset image_captioning --model_type ViT-B_16 
# python word2tensor.py