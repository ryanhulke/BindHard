from pathlib import Path
import json
import os
import requests


endpoint = os.environ["BINDHARD_ENDPOINT"].rstrip("/")
token = os.environ.get("BINDHARD_TOKEN", "").strip()

folder = Path("/blue/yanjun.li/ryan.hulke/BindHard/research/data/pocket_examples")
pocket_pdb_path = folder / "protein_pocket10.pdb"

payload = {
    "pdb_text": pocket_pdb_path.read_text(),
    "samples_per_target": 10,
    "return_trajectory": True,
    "trajectory_stride": 1,
}

headers = {"Content-Type": "application/json"}
if token:
    headers["Authorization"] = f"Bearer {token}"

response = requests.post(endpoint, headers=headers, json=payload, timeout=600)

response.raise_for_status()

data = response.json()
print(json.dumps(data.get("summary", {}), indent=2))

assert "samples" in data
assert "summary" in data
assert len(data["samples"]) == 10