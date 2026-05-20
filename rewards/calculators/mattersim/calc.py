import io
import contextlib
import logging
import os
from typing import Tuple, List

import numpy as np
from pymatgen.core.structure import Structure
from huggingface_hub import hf_hub_download

from utils.evaluation.reference_dataset_serializer import LMDBGZSerializer
from utils.evaluation.metrics_structure_summary import get_metrics_structure_summaries
from utils.evaluation.relaxation import relax_structures
from utils.evaluation.energy import EnergyMetricsCapability

from rewards.calculators.base import Calculator


def _get_device(device=None):
    import torch
    if device is None:
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    return device


class MatterSimEhull(Calculator):
    """
    Computes energy above the convex hull (Ehull, eV/atom) for a list of structures.

    Uses MatterSim (via mattergen's relax_structures) to relax each structure and obtain
    total energies, then evaluates Ehull against the MP2020 reference phase diagram using
    mattergen's EnergyMetricsCapability.

    Structures whose terminal elements are absent from the reference dataset return np.nan.
    """

    def __init__(
        self,
        root_dir: str,
        task: str = 'ehull',
        device: str = None,
        reference_path: str = None,
        potential_load_path: str = 'MatterSim-v1.0.0-5M.pth',
        silent: bool = True,
    ) -> None:
        super().__init__(root_dir, task)
        self.device = _get_device(device)
        self.potential_load_path = potential_load_path
        self.silent = silent

        if reference_path is None:
            reference_path = hf_hub_download(
                repo_id="jwchen25/MatInvent",
                filename="reference_MP2020correction.gz",
            )
        self.reference = LMDBGZSerializer().deserialize(reference_path)

    def _valid_mask(self, structures: List[Structure]) -> np.ndarray:
        """
        Return a boolean array of length len(structures).
        True when all terminal elements of a structure exist in the reference dataset
        with at least one non-NaN energy entry.
        """
        ref_keys = set(self.reference.entries_by_chemsys.keys())
        all_elements = {str(e) for s in structures for e in s.composition.elements}

        # Elements missing from the reference entirely
        bad_elements = all_elements - ref_keys

        # Elements present in the reference but with only NaN energies
        for el in all_elements & ref_keys:
            entries = self.reference.entries_by_chemsys.get(el, [])
            if entries and all(np.isnan(e.energy) for e in entries):
                bad_elements.add(el)

        return np.array([
            len({str(e) for e in s.composition.elements} & bad_elements) == 0
            for s in structures
        ])

    def calc(
        self,
        samples: Tuple[List[Structure], str],
        label: str = 'tmp',
    ) -> np.ndarray:
        structures = samples[0]
        out_path = os.path.abspath(os.path.join(self.root_dir, f'{label}.txt'))
        results = np.full(len(structures), np.nan)

        if not structures:
            np.savetxt(out_path, results, fmt='%.8f')
            return results

        # Pre-filter: skip structures whose elements are not in the reference dataset
        valid_mask = self._valid_mask(structures)
        valid_idx = np.where(valid_mask)[0]
        valid_strucs = [structures[i] for i in valid_idx]

        if not valid_strucs:
            logging.warning("MatterSimEhull: no structures have valid terminal elements in reference.")
            np.savetxt(out_path, results, fmt='%.8f')
            return results

        # Relax structures with MatterSim to obtain total energies
        try:
            if self.silent:
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                    relaxed, energies = relax_structures(
                        valid_strucs,
                        device=self.device,
                        potential_load_path=self.potential_load_path,
                    )
            else:
                relaxed, energies = relax_structures(
                    valid_strucs,
                    device=self.device,
                    potential_load_path=self.potential_load_path,
                )
        except Exception as exc:
            logging.warning(f"MatterSimEhull: MatterSim relaxation failed – {exc}")
            np.savetxt(out_path, results, fmt='%.8f')
            return results

        # Build MetricsStructureSummary objects (applies MP2020 energy corrections)
        # and compute Ehull via pymatgen PhaseDiagram
        try:
            summaries = get_metrics_structure_summaries(
                structures=relaxed,
                energies=energies.tolist(),
                original_structures=valid_strucs,
            )
            energy_cap = EnergyMetricsCapability(
                structure_summaries=summaries,
                reference_dataset=self.reference,
            )
            # energy_above_hull[i] corresponds to summaries[i] (indexed by entry_id = i)
            ehull = energy_cap.energy_above_hull
            for local_i, orig_i in enumerate(valid_idx):
                results[orig_i] = ehull[local_i]
        except Exception as exc:
            logging.warning(f"MatterSimEhull: Ehull computation failed {exc}")

        np.savetxt(out_path, results, fmt='%.8f')
        return results
