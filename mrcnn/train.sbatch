#!/bin/bash

#SBATCH --job-name=nerve-mrcnn
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=100GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END
#SBATCH --mail-user=jtb470@nyu.edu
module purge
unset XDG_RUNTIME_DIR
if [ "$SLURM_JOBTMP" != "" ]; then
    export XDG_RUNTIME_DIR=$SLURM_JOBTMP
fi

singularity exec --nv /beegfs/work/public/singularity/cuda-9.0-cudnn7-devel-ubuntu16.04.simg bash -c "source /home/jtb470/.bashrc && conda activate nerve-mrcnn && python tools/train_net.py --config-file 'configs/nerve-101.yaml' OUTPUT_DIR final"
