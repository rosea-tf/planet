#!/bin/bash
#SBATCH -D /home/rosea/planet/runs/curr/$LABEL

source ~/.bashrc

printenv SLURMD_NODENAME

rm /home/rosea/planet/runs/curr/$LABEL/00001/DONE
rm /home/rosea/planet/runs/curr/$LABEL/00001/FAIL

conda activate tf-gpu

cd $HOME/planet
python -m planet.scripts.train --logdir /home/rosea/planet/runs/curr/$LABEL --config default --params '{$PARAMS}'
