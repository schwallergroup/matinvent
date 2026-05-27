import os
import numpy as np
from typing import Tuple, List
from pymatgen.core.structure import Structure

from rewards.calculators.base import Calculator


class FakeEhull(Calculator):
    def __init__(
        self,
        root_dir: str,
        task: str = 'ehull',
    ) -> None:
        super().__init__(root_dir, task)

    def calc(
        self,
        samples: Tuple[List[Structure], str],
        label: str = 'tmp'
    ) -> np.ndarray[float]:

        struc_list = samples[0]
        out_path = os.path.join(self.root_dir, f'{label}.txt')
        out_path = os.path.abspath(out_path)

        results = [
            s.properties.get('ehull', np.nan) 
            for s in struc_list
        ]
        np.savetxt(out_path, results, fmt="%.6f")

        return results