#!/bin/bash
#SBATCH -D /home/rosea/planet/results/scores
printenv SLURMD_NODENAME

source $HOME/.bashrc
# source $HOME/env/planet/bin/activate
conda activate tf-gpu

cd $HOME/planet/results/scores
python compile_scores.py /home/rosea/planet/runs /home/rosea/planet/results/scores/scalars.json