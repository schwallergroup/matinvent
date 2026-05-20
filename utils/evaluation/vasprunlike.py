import re
from functools import cached_property

from pymatgen.analysis.structure_analyzer import oxide_type
from pymatgen.core import Structure
from pymatgen.entries.compatibility import Compatibility
from pymatgen.entries.computed_entries import ComputedEntry, ComputedStructureEntry, EnergyAdjustment
from pymatgen.io.vasp.sets import MPRelaxSet


class IdentityCorrectionScheme(Compatibility):
    def get_adjustments(self, entry: ComputedEntry | ComputedStructureEntry) -> list[EnergyAdjustment]:
        return []


class VasprunLike:
    """Mocks a VASP run using only the structure, to obtain MP2020 energy corrections."""

    def __init__(self, structure: Structure, energy: float, user_potcar_functional: str = "PBE") -> None:
        self.structure = structure
        self.energy = energy
        self.user_potcar_functional = user_potcar_functional

    @cached_property
    def mp_set(self) -> MPRelaxSet:
        return MPRelaxSet(
            self.structure,
            user_incar_settings={"KSPACING": 0.5},
            user_kpoints_settings=None,
        )

    @property
    def potcar_symbols(self) -> list[str]:
        return [f"{self.user_potcar_functional.upper()} {sym}" for sym in self.mp_set.potcar_symbols]

    @property
    def aspherical(self) -> bool:
        return self.mp_set.incar.get("LASPH", False)

    @property
    def hubbards(self) -> dict:
        symbols = [s.split()[1] for s in self.potcar_symbols]
        symbols = [re.split(r"_", s)[0] for s in symbols]
        if not self.mp_set.incar.get("LDAU", False):
            return {}
        us = self.mp_set.incar.get("LDAUU", [])
        js = self.mp_set.incar.get("LDAUJ", [])
        if len(js) != len(us):
            js = [0] * len(us)
        if len(us) == len(symbols):
            return {symbols[i]: us[i] - js[i] for i in range(len(symbols))}
        if sum(us) == 0 and sum(js) == 0:
            return {}
        raise ValueError("Length of U value parameters and atomic symbols are mismatched")

    @property
    def is_hubbard(self) -> bool:
        return bool(self.hubbards) and sum(self.hubbards.values()) > 1e-8

    @property
    def run_type(self) -> str:
        return "GGA+U" if self.is_hubbard else "GGA"

    def get_computed_entry(
        self,
        inc_structure: bool = True,
        energy_correction_scheme: Compatibility = IdentityCorrectionScheme(),
    ) -> ComputedEntry:
        entry_dict = {
            "correction": 0.0,
            "composition": self.structure.composition,
            "energy": self.energy,
            "parameters": {
                "is_hubbard": self.is_hubbard,
                "hubbards": self.hubbards,
                "run_type": self.run_type,
                "potcar_symbols": self.potcar_symbols,
            },
            "data": {"oxide_type": oxide_type(self.structure), "aspherical": self.aspherical},
            "structure": self.structure,
        }
        if inc_structure:
            entry = ComputedStructureEntry.from_dict(entry_dict)
        else:
            entry = ComputedEntry.from_dict(entry_dict)
        energy_correction_scheme.process_entry(entry)
        return entry
