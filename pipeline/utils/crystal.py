from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


def get_symmetry_primitive(struc: Structure):

    analyzer = SpacegroupAnalyzer(struc, symprec=0.1)
    symmetrized_structure = analyzer.get_refined_structure()
    primitive_structure = symmetrized_structure.get_primitive_structure()

    return primitive_structure
