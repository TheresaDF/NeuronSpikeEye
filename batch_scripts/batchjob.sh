#!/bin/sh

#  assert correct run dir
run_dir="src"
if ! [ "$(basename $PWD)" = $run_dir ];
then
    echo -e "\033[0;31mScript must be submitted from the directory: $run_dir\033[0m"
    exit 1
fi

# create dir for logs
mkdir -p "logs/"

### General options
### –- specify queue --
#BSUB -q hpc
### -- set the job Name --
#BSUB -J preprocessing
### -- ask for number of cores (default: 1) -- 
#BSUB -n 1 
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now 
#BSUB -W 7:00
### -- request 5GB of system-memory --
#BSUB -R "rusage[mem=8GB]"
### -- set the email address --
##BSUB -u s194329@student.dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion-- 
#BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o logs/preprocess%J.out 
#BSUB -e logs/preprocess%J.err 
### -- end of LSF options --

# activate env

# load additional modules
module load cuda/11.4

# run scripts
python src/data/preprocess_data.py