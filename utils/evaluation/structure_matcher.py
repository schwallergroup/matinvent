from itertools import combinations

import numpy as np
from pymatgen.analysis.structure_matcher import AbstractComparator, OrderDisorderElementComparator, StructureMatcher
from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import Structure

from utils.evaluation.globals import MAX_RMSD


class RMSDStructureMatcher(StructureMatcher):
    def __init__(self):
        super().__init__(
            ltol=0.5, stol=MAX_RMSD, angle_tol=10, primitive_cell=True,
            scale=False, attempt_supercell=True, allow_subset=False,
        )


class OrderedStructureMatcher(StructureMatcher):
    def __init__(self, ltol=0.2, stol=0.3, angle_tol=5, primitive_cell=True,
                 scale=True, attempt_supercell=False, allow_subset=False, *args, **kwargs):
        super().__init__(
            ltol=ltol, stol=stol, angle_tol=angle_tol, primitive_cell=primitive_cell,
            scale=scale, attempt_supercell=attempt_supercell, allow_subset=allow_subset,
            *args, **kwargs,
        )

    @property
    def name(self) -> str:
        return "OrderedStructureMatcher"


class DefaultOrderedStructureMatcher(OrderedStructureMatcher):
    def __init__(self):
        super().__init__()


class DisorderedStructureMatcher(StructureMatcher):
    def __init__(
        self, ltol=0.2, stol=0.3, angle_tol=5.0, primitive_cell=True, scale=True,
        comparator: AbstractComparator = OrderDisorderElementComparator(),
        attempt_supercell=True, allow_subset=True,
        relative_radius_difference_threshold=0.3, electronegativity_difference_threshold=1.0,
        reduced_formula_atol=1e-2, reduced_formula_rtol=1e-1, *args, **kwargs,
    ):
        super().__init__(
            ltol=ltol, stol=stol, angle_tol=angle_tol, primitive_cell=primitive_cell,
            allow_subset=allow_subset, attempt_supercell=attempt_supercell, scale=scale,
            comparator=comparator, *args, **kwargs,
        )
        self.relative_radius_difference_threshold = relative_radius_difference_threshold
        self.electronegativity_difference_threshold = electronegativity_difference_threshold
        self.ordered_structurematcher = OrderedStructureMatcher(
            ltol=ltol, stol=stol, angle_tol=angle_tol,
            primitive_cell=primitive_cell, scale=scale,
        )
        self.reduced_formula_atol = reduced_formula_atol
        self.reduced_formula_rtol = reduced_formula_rtol

    @property
    def name(self) -> str:
        return "DisorderedStructureMatcher"

    def fit(self, structure_1: Structure, structure_2: Structure) -> bool:
        s1 = structure_1.copy().remove_oxidation_states()
        s2 = structure_2.copy().remove_oxidation_states()
        if s1 == s2:
            return True
        if s1.is_ordered and s2.is_ordered:
            if s2.composition.reduced_formula != s1.composition.reduced_formula:
                return False
            if self.ordered_structurematcher.fit(s1, s2):
                return True
            s1, can_disorder = try_make_structure_disordered(
                s1,
                relative_radius_difference_threshold=self.relative_radius_difference_threshold,
                electronegativity_difference_threshold=self.electronegativity_difference_threshold,
            )
            if can_disorder:
                return super().fit(s1, s2)
            return False
        if not s1.composition.fractional_composition.almost_equals(
            s2.composition.fractional_composition,
            atol=self.reduced_formula_atol, rtol=self.reduced_formula_rtol,
        ):
            return False
        return super().fit(s1, s2)


class DefaultDisorderedStructureMatcher(DisorderedStructureMatcher):
    def __init__(self):
        super().__init__()


def do_elements_substitute(e1: Element, e2: Element, rr_thresh=0.3, en_thresh=1.0) -> bool:
    rel_r = abs(e1.atomic_radius - e2.atomic_radius) / np.mean([e1.atomic_radius, e2.atomic_radius])
    return rel_r <= rr_thresh and abs(e1.X - e2.X) <= en_thresh


def _get_cliques(pairs: list) -> list:
    cliques: list = [[]]
    for pair in pairs:
        prev = None
        for i, group in enumerate(cliques):
            if pair[0] in group or pair[1] in group:
                if prev is not None:
                    cliques[prev].extend(group)
                    cliques[i] = []
                else:
                    cliques[i].extend(pair)
                    prev = i
        if prev is None:
            cliques.append(pair)
    return [list(set(g)) for g in cliques if g]


def check_is_disordered(structure: Structure, rr_thresh=0.3, en_thresh=1.0):
    s = structure.copy().remove_oxidation_states()
    pairs = [
        [e1, e2] for e1, e2 in combinations(list(s.composition), 2)
        if do_elements_substitute(e1, e2, rr_thresh, en_thresh)
    ]
    if not pairs:
        return False, [[]]
    return True, _get_cliques(pairs)


def make_structure_disordered(structure: Structure, substitution: list) -> Structure:
    s = structure.copy().remove_oxidation_states()
    fracs = {str(sp): s.composition.get_atomic_fraction(str(sp)) for sp in s.composition}
    for clique in substitution:
        total = sum(fracs[str(sp)] for sp in clique)
        s.replace_species({
            str(sp): "".join(str(sp2) + str(fracs[str(sp2)] / total) for sp2 in clique)
            for sp in clique
        })
    return s


def try_make_structure_disordered(structure: Structure, relative_radius_difference_threshold=0.3,
                                   electronegativity_difference_threshold=1.0):
    can, subs = check_is_disordered(
        structure, relative_radius_difference_threshold, electronegativity_difference_threshold,
    )
    return (make_structure_disordered(structure, subs) if can else structure, can)
