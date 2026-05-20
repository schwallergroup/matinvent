#!/bin/bash

pip install uv
uv venv .venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements.txt --index-strategy unsafe-best-match

# MatterGen model code is bundled in the mattergen/ directory — no separate install needed.
