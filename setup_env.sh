curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
rm -rf uv.lock && rm -rf .venv && uv venv --python python3.10 && source .venv/bin/activate && uv sync
cp setup/contact.py .venv/lib64/python3.10/site-packages/brax/contact.py
cp setup/json.py .venv/lib64/python3.10/site-packages/brax/io/json.py