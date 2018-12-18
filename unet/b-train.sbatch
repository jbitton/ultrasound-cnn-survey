#!/bin/bash

#SBATCH --job-name=unet-nerve
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=100GB
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END
#SBATCH --mail-user=jtb470@nyu.edu
module purge
unset XDG_RUNTIME_DIR
if [ "$SLURM_JOBTMP" != "" ]; then
    export XDG_RUNTIME_DIR=$SLURM_JOBTMP
fi

module load python3/intel/3.6.3
module load cuda/9.0.176
module load cudnn/9.0v7.3.0.29

source env/bin/activate

python batch_norm_train.py
