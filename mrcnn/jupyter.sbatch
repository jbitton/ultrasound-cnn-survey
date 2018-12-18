#!/bin/bash

#SBATCH --job-name=jupyter
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=100GB
#SBATCH --time=4:00:00

module purge

port=11000

/usr/bin/ssh -N -f -R $port:localhost:$port log-0
/usr/bin/ssh -N -f -R $port:localhost:$port log-1

unset XDG_RUNTIME_DIR
if [ "$SLURM_JOBTMP" != "" ]; then
    export XDG_RUNTIME_DIR=$SLURM_JOBTMP
fi

singularity exec --nv /beegfs/work/public/singularity/cuda-9.0-cudnn7-devel-ubuntu16.04.simg bash -c "source /home/jtb470/.bashrc && conda activate nerve-mrcnn && jupyter notebook --no-browser --port $port --notebook-dir=$(pwd)"
