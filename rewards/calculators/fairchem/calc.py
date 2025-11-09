import os
import subprocess
import numpy as np
from typing import Tuple, List
from pymatgen.core.structure import Structure

from rewards.calculators.base import Calculator
from rewards.calculators.fairchem import ELASTIC_PATH, PHONON_PATH


class FairChem(Calculator):
    def __init__(
        self,
        root_dir: str,
        task: str = 'bulk_modulus',
        env_name: str = 'fair-chem-v1',
        worker: int = 1,
    ) -> None:
        super().__init__(root_dir, task)
        self.env_name = env_name
        self.worker = worker

    def calc(
        self,
        samples: Tuple[List[Structure], str],
        label: str = 'tmp'
    ) -> np.ndarray[float]:

        xyz_path = samples[1]
        out_path = os.path.join(self.root_dir, f'{label}.txt')

        # Absolute path
        xyz_path = os.path.abspath(xyz_path)
        out_path = os.path.abspath(out_path)

        if self.task == 'bulk_modulus':
            script_name = ELASTIC_PATH
        elif self.task == 'heat_capacity':
            script_name = PHONON_PATH
        else:
            raise ValueError(
                f"{self.task} is unknown task for FairChem calculator!"
            )

        process = subprocess.run(
            [
                'conda', 'run', '-n', self.env_name,
                'python', script_name, xyz_path, out_path, str(self.worker),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # proc_error = process.stderr.decode()
        # if proc_error != "":
        #     logging.error(proc_error)
        # print(proc_error)

        assert os.path.isfile(out_path)
        results = np.genfromtxt(out_path)

        return results
