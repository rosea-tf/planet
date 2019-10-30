#!/bin/bash
#SBATCH -D /home/rosea/planet/results

source $HOME/.bashrc
source $HOME/env/planet/bin/activate

export LD_LIBRARY_PATH=$HOME/local/cuda/lib64:$LD_LIBRARY_PATH
export CPATH=$HOME/local/cuda/include:$CPATH
export LIBRARY_PATH=$HOME/local/cuda/lib64:$LIBRARY_PATH

printenv SLURMD_NODENAME
cd $HOME/planet/results
python compile_scores.py /home/rosea/planet/runs/ch /home/rosea/planet/runs/ch.json