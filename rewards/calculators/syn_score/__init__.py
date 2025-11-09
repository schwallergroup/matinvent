from __future__ import annotations

import os

EMB_PATH = os.path.join(__path__[0], "element_emb.json")
MODEL_PATH = os.path.join(__path__[0], "model_pt")
EMB_PATH = os.path.abspath(EMB_PATH)
MODEL_PATH = os.path.abspath(MODEL_PATH)
