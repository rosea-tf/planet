#!/bin/bash
#SBATCH -D /home/rosea/planet/results/scores

source $HOME/.bashrc
source $HOME/env/planet/bin/activate

export LD_LIBRARY_PATH=$HOME/local/cuda/lib64:$LD_LIBRARY_PATH
export CPATH=$HOME/local/cuda/include:$CPATH
export LIBRARY_PATH=$HOME/local/cuda/lib64:$LIBRARY_PATH

printenv SLURMD_NODENAME
cd $HOME/planet/results/scores
python compile_histograms.py /home/rosea/planet/runs/bk /home/rosea/planet/results/scores/histograms.pkl

