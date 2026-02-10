# Usage
- `conda create -n bindhard python=3.10`
- `conda activate bindhard`
- this repo may or may not be compatible with older versions of torch/cuda. If you want, you can try with your already-installed cuda version if you are on a personal device, not an HPC. Otherwise if you are on Hipergator, run:
- `pip install -r requirements.txt`
- download and split dataset in `create_dataset.ipynb`
- run locally with `python train.py` or via SLURM with `sbatch train.sh`
- track experiments at https://wandb.ai/rshulke-university-of-florida/bindhard/

- Note: current dataset is filtered to very strong binders. This is good for the main task of generating strong binders, but not necessarily for the potential downstream task of predicting binding affinity.

### Model
- differences: 
  - TargetDiff's best model uses **Graph Attention** message passing compared to our MLP message passing
  - they have 9 layers, we have 6
  - their embed dim 128 ours 256


### TO-DO:
- evaluation metrics
  - RMSE (in Angstroms) against test set, sampled, but with known atom count & types?
  - look to the TargetDiff paper & similar papers for metrics
- build & test graph attention architecture
- 3D visualization tool of molecule being generated in protein pocket and diffusion of atom locations