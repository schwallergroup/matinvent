import io
import os
import contextlib
import numpy as np
from typing import Tuple, List
from pymatgen.core.structure import Structure
from jarvis.core.atoms import pmg_to_atoms

from rewards.calculators.base import Calculator
from rewards.calculators.alignn.prediction import get_multiple_predictions


TASK_MODEL_DICT = {
    'band_gap': 'mp_bandgap_hf',
    'formation_energy': 'mp_e_form_alignn',
    'bulk_modulus': 'mp_bulk_modulus_hf',
    'shear_modulus': 'mp_shear_modulus_hf',
    'magnetic_density': 'mp_total_mag_per_atom_hf',
    'total_dielectric_constant': 'mp_dielectric_hf',
    'vickers_hardness': 'mp_vickers_hardness_hf',
    'figure_of_merit': '',
    'pugh_ratio': '',
    'young_modulus': '',
}


def get_prediction(
    atoms_list: List,
    model_name: str,
    device: str | None = None,
    silent: bool = True
) -> np.ndarray[float]:
    if silent:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            results = get_multiple_predictions(
                atoms_array=atoms_list,
                model_name=model_name,
                device=device,
            )
    else:
        results = get_multiple_predictions(
            atoms_array=atoms_list,
            model_name=model_name,
            device=device,
        )
    results = np.array(results)
    return results


class ALIGNN(Calculator):
    def __init__(
        self,
        root_dir: str,
        task: str = 'band_gap',
        device: str | None = None,
        silent: bool = True
    ) -> None:
        super().__init__(root_dir, task)
        self.device = device
        self.silent = silent

    def calc(
        self,
        samples: Tuple[List[Structure], str],
        label: str = 'tmp'
    ) -> np.ndarray[float]:

        struc_list = samples[0]
        atoms_list = [pmg_to_atoms(struc) for struc in struc_list]
        out_path = os.path.join(self.root_dir, f'{label}.txt')
        out_path = os.path.abspath(out_path)

        if self.task not in TASK_MODEL_DICT:
            raise ValueError(
                f"{self.task} is unknown task for ALIGNN calculator!"
            )

        if self.task == 'vickers_hardness':
            bulk = get_prediction(
                atoms_list=atoms_list,
                model_name=TASK_MODEL_DICT['bulk_modulus'],
                device=self.device,
                silent=self.silent,
            )
            bulk[bulk < 0.0] = 0.0

            shear = get_prediction(
                atoms_list=atoms_list,
                model_name=TASK_MODEL_DICT['shear_modulus'],
                device=self.device,
                silent=self.silent,
            )
            shear[shear < 0.0] = 0.0

            # Tian's model for Vickers hardness
            # Ref: Microscopic theory of hardness and design of novel superhard crystals. Int. J. Refract. Met. H. 33, 93–106 (2012).
            k = shear / bulk
            results = 0.92 * (k ** 1.137) * (shear ** 0.708)

            # Teter's model for Vickers hardness
            # Ref: Computational alchemy: The search for new superhard materials. MRS Bull. 23, 22–27 (1998).
            results[bulk < 25.0] = 0.151 * shear[bulk < 25.0]

            # save results of bulk and shear modulus
            results[results < 0.0] = 0.0
            bulk_path = os.path.join(self.root_dir, f'{label}_bulk.txt')
            shear_path = os.path.join(self.root_dir, f'{label}_shear.txt')
            np.savetxt(bulk_path, bulk, fmt="%.6f")
            np.savetxt(shear_path, shear, fmt="%.6f")
        elif self.task == 'pugh_ratio':
            bulk = get_prediction(
                atoms_list=atoms_list,
                model_name=TASK_MODEL_DICT['bulk_modulus'],
                device=self.device,
                silent=self.silent,
            )
            bulk[bulk < 0.0] = 0.0

            shear = get_prediction(
                atoms_list=atoms_list,
                model_name=TASK_MODEL_DICT['shear_modulus'],
                device=self.device,
                silent=self.silent,
            )
            shear[shear <= 0.0] = 0.01

            results = bulk / shear

            # save results of bulk and shear modulus
            bulk_path = os.path.join(self.root_dir, f'{label}_bulk.txt')
            shear_path = os.path.join(self.root_dir, f'{label}_shear.txt')
            np.savetxt(bulk_path, bulk, fmt="%.6f")
            np.savetxt(shear_path, shear, fmt="%.6f")
        elif self.task == 'young_modulus':
            bulk = get_prediction(
                atoms_list=atoms_list,
                model_name=TASK_MODEL_DICT['bulk_modulus'],
                device=self.device,
                silent=self.silent,
            )
            bulk[bulk <= 0.0] = 0.01

            shear = get_prediction(
                atoms_list=atoms_list,
                model_name=TASK_MODEL_DICT['shear_modulus'],
                device=self.device,
                silent=self.silent,
            )
            shear[shear <= 0.0] = 0.01

            results = 9 * bulk * shear / (3 * bulk + shear)
            # save results of bulk and shear modulus
            bulk_path = os.path.join(self.root_dir, f'{label}_bulk.txt')
            shear_path = os.path.join(self.root_dir, f'{label}_shear.txt')
            np.savetxt(bulk_path, bulk, fmt="%.6f")
            np.savetxt(shear_path, shear, fmt="%.6f")
        elif self.task == 'figure_of_merit':
            gap = get_prediction(
                atoms_list=atoms_list,
                model_name=TASK_MODEL_DICT['band_gap'],
                device=self.device,
                silent=self.silent,
            )
            gap[gap < 0.0] = 0.0

            die = get_prediction(
                atoms_list=atoms_list,
                model_name=TASK_MODEL_DICT['total_dielectric_constant'],
                device=self.device,
                silent=self.silent,
            )
            die[die < 0.0] = 0.0
            results = gap * die

            # save results of bulk and shear modulus
            gap_path = os.path.join(self.root_dir, f'{label}_gap.txt')
            die_path = os.path.join(self.root_dir, f'{label}_die.txt')
            np.savetxt(gap_path, gap, fmt="%.6f")
            np.savetxt(die_path, die, fmt="%.6f")
        else:
            results = get_prediction(
                atoms_list=atoms_list,
                model_name=TASK_MODEL_DICT[self.task],
                device=self.device,
                silent=self.silent,
            )

        if self.task == 'band_gap':
            results[results < 0.0] = 0.0

        if self.task == 'magnetic_density':
            # Correction
            results = results / 0.84
            natom = np.array([len(s) for s in struc_list])
            volumes = np.array([s.volume for s in struc_list])
            results = results * natom / volumes
            # volumes = np.array([s.volume for s in struc_list])
            # fu = np.array(
            #     [s.composition.get_reduced_composition_and_factor()[1] for s in struc_list]
            # )
            # results = results * fu / volumes
            results[results < 0.0] = 0.0

        np.savetxt(out_path, results, fmt="%.6f")

        return results
