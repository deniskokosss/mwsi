set -euo pipefail

rm -rf .venv


if ! command -v python3.10 >/dev/null 2>&1; then
  echo "python3.10 not found; attempting: uv python install 3.10" >&2
  uv python install 3.10
fi
uv venv --python python3.10
source .venv/bin/activate

# Ensure pip tooling is sane
uv pip install -U pip setuptools wheel 
uv pip install "numpy==1.25.2" "scipy==1.11.1" bitarray
uv pip install -r requirements-mwsi.txt
uv pip install git+https://github.com/Samsung/LexSubGen.git --no-deps
uv pip install fairseq==0.12.2 --no-deps
uv pip install torch==2.9.1 --force-reinstall

# Monkey-patches to work on modern torch
sed -i 's/torch\.load(self\.filename, lambda storage, loc: storage)/torch.load(self.filename, lambda storage, loc: storage, weights_only=False)/' .venv/lib/python3.10/site-packages/stanza/models/common/pretrain.py
sed -i 's/torch\.load(f, map_location=torch\.device("cpu"))/torch.load(f, map_location=torch.device("cpu"), weights_only=False)/' .venv/lib/python3.10/site-packages/fairseq/checkpoint_utils.py

wget https://dl.fbaipublicfiles.com/fairseq/models/xlmr.large.tar.gz
tar -xzf xlmr.large.tar.gz
mv xlmr.large xlmr_large

python multilang_wsi_evaluation/evaluate_vectorizer.py --help

echo "\n\n\nEnvironment installed successfully\n"