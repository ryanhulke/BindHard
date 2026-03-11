# Usage
- `conda create -n bindhard python=3.10`
- `conda activate bindhard`
- this repo may or may not be compatible with older versions of torch/cuda. If you want, you can try with your already-installed cuda version if you are on a personal device, not an HPC. Otherwise if you are on Hipergator, run:
- `pip install -r requirements.txt`
- download and split dataset in `create_dataset.ipynb`
- `curl -L https://github.com/rdkit/rdkit/raw/master/Contrib/SA_Score/fpscores.pkl.gz -o data/fpscores.pkl.gz`
- run locally with `python train.py` or via SLURM with `sbatch train.sh`
- track experiments at https://wandb.ai/rshulke-university-of-florida/bindhard/

- Note: current dataset is filtered to very strong binders. This is good for the main task of generating strong binders, but not necessarily for the potential downstream task of predicting binding affinity.



## TO-DO:
#### Wire Cloudflare bindings
- Add your Pages/Functions config with a D1 binding and MODAL_GENERATE_URL.
- Pages Functions can bind D1 directly through Wrangler or the dashboard.
- Create the D1 database and apply the schema
- Create the DB.
- Put your SQL into app/db/migrations/0001_init.sql.
- Run the migration.
- D1 migrations are the intended workflow for versioning schema changes.
#### Deploy Modal
- Create the model-weights volume.
- Upload your checkpoint and YAML config into that volume.
- Create the bearer-token secret.
- Deploy serve_inference.py.
- Modal’s documented pattern is Volume for model weights, Secret for tokens, @modal.enter for one-time container init, and web endpoints for HTTPS access.
#### Smoke test Modal directly
  - Hit the deployed Modal endpoint with a tiny known PDB and box.
  - Confirm you get back per-sample:
    - ligand_pos
    - ligand_type
    - smiles
    - vina_score
    - qed_score
    - sa_score
#### Smoke test the Cloudflare route
- POST a .pdb to your Pages Function.
- Confirm:
  - one row in jobs
  - N rows in samples
  - summary_json filled in
  - failed jobs write error_message