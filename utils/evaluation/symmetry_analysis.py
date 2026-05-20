from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from utils.evaluation.structure_matcher import try_make_structure_disordered


class DefaultSpaceGroupAnalyzer(SpacegroupAnalyzer):
    def __init__(self, structure: Structure):
        super().__init__(structure, symprec=0.1, angle_tolerance=5.0)


class DisorderedSpaceGroupAnalyzer(SpacegroupAnalyzer):
    def __init__(self, structure: Structure):
        structure, _ = try_make_structure_disordered(structure, 0.3, 1.0)
        super().__init__(structure, symprec=0.1, angle_tolerance=5.0)
