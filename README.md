# Usage
- `conda create -n bindhard python=3.11`
- `conda activate bindhard`
- this repo may or may not be compatible with older versions of torch/cuda. If you want, you can try with your already-installed cuda version if you are on a personal device, not an HPC. Otherwise if you are on Hipergator, run:
- `pip install -r requirements.txt`
- download and split dataset in `research/create_dataset.ipynb`
- `curl -L https://github.com/rdkit/rdkit/raw/master/Contrib/SA_Score/fpscores.pkl.gz -o data/fpscores.pkl.gz`
- run locally with `python research/train.py` or via SLURM with `sbatch research/scripts/train.sh`
- track experiments at https://wandb.ai/rshulke-university-of-florida/bindhard/

- Note: current dataset is filtered to very strong binders. This is good for the main task of generating strong binders, but not necessarily for the potential downstream task of predicting binding affinity.



## TO-DO:
#### Return n valid molecules
- right now, the user requests n samples, but some % of generated samples are invalid. Instead of returning invalid samples, we can keep generating until we have n valid samples.
