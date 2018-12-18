#!/bin/bash

#SBATCH --job-name=jupyter
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=100GB
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:1

module purge

port=11000

/usr/bin/ssh -N -f -R $port:localhost:$port log-0
/usr/bin/ssh -N -f -R $port:localhost:$port log-1

unset XDG_RUNTIME_DIR
if [ "$SLURM_JOBTMP" != "" ]; then
    export XDG_RUNTIME_DIR=$SLURM_JOBTMP
fi

module load python3/intel/3.6.3
module load cuda/9.0.176
module load cudnn/9.0v7.3.0.29

source env/bin/activate
jupyter notebook --no-browser --port $port --notebook-dir=$(pwd)
