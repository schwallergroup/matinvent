import os
import logging
import contextlib
import multiprocessing as mp
import yaml
import numpy as np
from typing import Tuple, List
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifWriter

from rewards.calculators.base import Calculator
from rewards.calculators.dft import DFT_CONFIG_PATH
from rewards.calculators.dft.job import RemoteQueueJob


@contextlib.contextmanager
def suppress_logging(logger_name=None):

    if logger_name:
        logger = logging.getLogger(logger_name)
    else:
        logger = logging.getLogger()
    
    original_level = logger.level
    logger.setLevel(logging.CRITICAL + 1)
    try:
        yield
    finally:
        logger.setLevel(original_level)


def dft_run(task, dir, cif_path, config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config['task'] = task
    config['dir'] = dir
    config['cif'] = cif_path
    config['config'] = config_path

    if config['machine'] == 'remote':
        job = RemoteQueueJob.from_config(config)

    try:
        with suppress_logging('paramiko'):
            results = job.submit_wait_read()
            results = float(results)
    except:
        results = np.nan

    return results


class DFTCalc(Calculator):
    def __init__(
        self,
        root_dir: str,
        task: str = 'band_gap',
        max_node: int = 8,
        config_path: str = DFT_CONFIG_PATH,
    ) -> None:
        super().__init__(root_dir, task)
        self.max_node = max_node
        self.config_path = os.path.abspath(config_path)

    def calc(
        self,
        samples: Tuple[List[Structure], str],
        label: str = 'tmp',
    ) -> np.ndarray[float]:

        struc_list = samples[0]
        cif_dir = os.path.join(self.root_dir, label)
        os.makedirs(cif_dir, exist_ok=True)

        param_list = []
        for i, struc in enumerate(struc_list):
            cif_writer = CifWriter(struc)
            cif_path = os.path.join(cif_dir, f'{i}.cif')
            cif_path = os.path.abspath(cif_path)
            cif_writer.write_file(cif_path)
            dir = os.path.join(label, f'{i:0>2d}')
            param_list.append(
                (self.task, dir, cif_path, self.config_path)
            )

        with mp.Pool(processes=self.max_node) as pool:
            results = pool.starmap(dft_run, param_list)

        results = np.array(results, dtype=float)
        out_path = os.path.join(self.root_dir, f'{label}.txt')
        np.savetxt(out_path, results, fmt='%.6f')

        return results
