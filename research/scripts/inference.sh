#!/bin/sh
#SBATCH --cpus-per-task=2
#SBATCH --mem=15gb
#SBATCH --time=24:00:00
#SBATCH --account=yanjun.li
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --job-name=inferenceBH

## slurm script to run inference on BindHard model

cd /blue/yanjun.li/ryan.hulke/BindHard/research

pwd; hostname; date
export XDG_RUNTIME_DIR=${SLURM_TMPDIR}

module load conda
conda activate bindhard

python test_set_inference.py