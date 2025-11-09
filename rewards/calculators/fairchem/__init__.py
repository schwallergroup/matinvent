from __future__ import annotations

import os

ELASTIC_PATH = os.path.join(__path__[0], "elastic.py")
PHONON_PATH = os.path.join(__path__[0], "phonon.py")
ELASTIC_PATH = os.path.abspath(ELASTIC_PATH)
PHONON_PATH = os.path.abspath(PHONON_PATH)
