"""sun_filter — standalone SUN/MSUN pipeline for pymatgen Structures.

Standalone module: depends only on pymatgen, ase, numpy, scipy, pandas,
datasets/huggingface_hub, mace-torch, orb-models, fairchem-core,
and optionally torch-sim for accelerated batch relaxation.

Usage
-----
    from pymatgen.core import Structure
    from sun_filter import sun_filter, SUNFilterResult

    result = sun_filter(
        structures,          # list[pymatgen.Structure]
        mlip_names=["mace_mp"],
        device="cuda",
        n_jobs=8,
    )
    print(f"SUN rate:  {result.sun_rate:.3f}")
    print(f"MSUN rate: {result.msun_rate:.3f}")
"""

from utils.sun_filter.pipeline import vsun_filter

__all__ = ["vsun_filter"]
