#!/usr/bin/sh

#SBATCH -p cpu20
#SBATCH -t 1-23:59:00
#SBATCH -J FFBS1.0
#SBATCH -D /HPS/deep_topopt/work/topopt-fourfeat-3d                                # FIX THIS
#SBATCH -o /HPS/deep_topopt/work/topopt-fourfeat-3d/logs/slurm/ff/slurm-%x-%j.log  # FIX THIS
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 128
#SBATCH --mem-per-cpu 2000

# Make conda available:
eval "$(conda shell.bash hook)"
# Activate a conda environment:
conda activate voxelfem_3d_py37  # FIX THIS


# call your program here  # FIX THIS
python3 /HPS/deep_topopt/work/topopt-fourfeat-3d/training/train_xdg.py --jid ${SLURM_JOBID} --grid "[320, 160, 80]" --prob "problems/3d/bridge.json" --mgl 3 --iter 5000 --v0 "0.4" --es 1024 --nn 512 --nl 4 --sigma 1.0 --cs 100
