#!/bin/sh
#SBATCH --cpus-per-task=4
#SBATCH --mem=20gb
#SBATCH --account=yanjun.li
#SBATCH --time=3-00:00:00
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --job-name=trainBH

cd /blue/yanjun.li/ryan.hulke/BindHard/research

pwd; hostname; date
export XDG_RUNTIME_DIR=${SLURM_TMPDIR}

module load conda
conda activate bindhard

python -m scripts.train_generator
