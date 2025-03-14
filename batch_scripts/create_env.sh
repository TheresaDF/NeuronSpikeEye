#!/bin/bash

echo "\e[33mINFO: create_env.sh will create a virtual environment to use on the DTUs HPC cluster"

# set env name
env_name="neuron_spike"

if [ $( basename $PWD ) != "src" ]
then 
    echo "\e[33mWARN: Virtual environment about to be created without src as basename, instead it will be created at $PWD\e[0m" 
fi


### Make python environment
module load python3/3.12 # only use if on HPC
python3 -m venv $env_name

# source AutoVC-env/bin/activate
source $env_name/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt


if [ $( basename $PWD ) = "src" ]
then 
    echo "Virtual environment created at $PWD"
else
    echo "\e[33mWARN: Virtual environment was not created with src as basename, instead it was created at $PWD\e[0m"
fi

deactivate

# tell use to manually install some packages as they have problems being installed through requirements.txt
echo "\e[33mPlease activate the environment and use 'python -m pip install six setuptools appdirs'\e[0m"