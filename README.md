# Usage
- `conda create -n bindhard python=3.10`
- `conda activate bindhard`
- this repo may or may not be compatible with older versions of torch/cuda. If you want, you can try with your already-installed cuda version if you are on a personal device, not an HPC. Otherwise if you are on Hipergator, run:
- `pip install -r requirements.txt`
- download and split dataset in `create_dataset.ipynb`


- Note: current dataset is filtered to very strong binders. This is good for the main task of generating strong binders, but not necessarily for the potential downstream task of predicting binding affinity.