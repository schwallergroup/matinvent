import gzip
import os
import shutil
import weakref
from collections import defaultdict
from functools import cached_property
from pathlib import Path
from tempfile import mkdtemp
from typing import Iterator, Mapping

from monty.json import MontyDecoder
from pymatgen.core import Composition
from pymatgen.entries.computed_entries import ComputedStructureEntry
from tqdm.autonotebook import tqdm

from utils.evaluation.lmdb_utils import lmdb_get, lmdb_open, lmdb_put, lmdb_read_metadata
from utils.evaluation.reference_dataset import ReferenceDataset, ReferenceDatasetImpl


def gzip_decompress(gzip_file_path: str | os.PathLike, output_dir: str | os.PathLike) -> Path:
    output_path = Path(output_dir) / Path(gzip_file_path).name[:-3]
    with gzip.open(gzip_file_path, "rb") as fin:
        with open(output_path, "wb") as fout:
            fout.write(fin.read())
    return output_path


class LMDBGZSerializer:
    def deserialize(self, dataset_path: str | os.PathLike) -> ReferenceDataset:
        tempdir = mkdtemp()
        lmdb_path = gzip_decompress(dataset_path, tempdir)
        name = lmdb_read_metadata(lmdb_path, "name")
        return ReferenceDataset(
            name=name,
            impl=LMDBBackedReferenceDatasetImpl(lmdb_path, cleanup_dir=True),
        )


class LMDBBackedReferenceDatasetImpl(ReferenceDatasetImpl):
    def __init__(self, lmdb_path: Path, cleanup_dir: bool = False):
        self.env = lmdb_open(lmdb_path, readonly=True)
        self._num_entries = self._build_num_entries(lmdb_path)
        self.total_num_entries = sum(
            sum(d.values()) for d in self._num_entries.values()
        )
        weakref.finalize(self, self._cleanup, self.env, cleanup_dir)

    def _build_num_entries(self, lmdb_path):
        chemical_systems = lmdb_read_metadata(lmdb_path, "chemical_systems")
        result = defaultdict(dict)
        with self.env.begin() as txn:
            for chemsys in chemical_systems:
                reduced_formulas = lmdb_read_metadata(lmdb_path, f"{chemsys}.reduced_formulas")
                for rf in reduced_formulas:
                    result[chemsys][rf] = lmdb_get(txn, f"{chemsys}.{rf}.length")
        return {k: v for k, v in result.items()}

    def __iter__(self) -> Iterator[ComputedStructureEntry]:
        for chemsys, by_rf in self._num_entries.items():
            for rf in by_rf:
                yield from self._get_entries(chemsys, rf)

    def __len__(self) -> int:
        return self.total_num_entries

    @property
    def chemical_systems(self) -> tuple[str, ...]:
        return tuple(self._num_entries.keys())

    @cached_property
    def reduced_formulas(self) -> tuple[str, ...]:
        return tuple(rf for by_rf in self._num_entries.values() for rf in by_rf)

    def _get_entries(self, chemsys: str, rf: str) -> Iterator[ComputedStructureEntry]:
        n = self._num_entries[chemsys][rf]
        for i in range(n):
            with self.env.begin() as txn:
                d = lmdb_get(txn, f"{chemsys}.{rf}.{i}")
            yield MontyDecoder().process_decoded(d)

    def get_entries_by_chemsys(self, chemsys: str) -> Iterator[ComputedStructureEntry]:
        for rf in self._num_entries.get(chemsys, {}):
            yield from self._get_entries(chemsys, rf)

    def get_entries_by_reduced_formula(self, rf: str) -> Iterator[ComputedStructureEntry]:
        chemsys = Composition(rf).chemical_system
        yield from self._get_entries(chemsys, rf)

    @cached_property
    def entries_by_reduced_formula(self) -> "LMDBBackedReducedFormulaLookup":
        return LMDBBackedReducedFormulaLookup(self)

    @cached_property
    def entries_by_chemsys(self) -> "LMDBBackedChemicalSystemLookup":
        return LMDBBackedChemicalSystemLookup(self)

    @classmethod
    def _cleanup(cls, env, cleanup_dir: bool) -> None:
        try:
            database_dir = Path(env.path()).parent
        except Exception:
            return
        env.close()
        if cleanup_dir:
            shutil.rmtree(database_dir)


class _WeakRefMixin:
    def __init__(self, impl: LMDBBackedReferenceDatasetImpl):
        self._impl = weakref.ref(impl)

    @property
    def impl(self) -> LMDBBackedReferenceDatasetImpl:
        impl = self._impl()
        assert impl is not None
        return impl


class LMDBBackedChemicalSystemLookup(_WeakRefMixin, Mapping[str, list[ComputedStructureEntry]]):
    def __init__(self, impl: LMDBBackedReferenceDatasetImpl):
        super().__init__(impl)
        self._chemsys = frozenset(impl.chemical_systems)

    def __len__(self) -> int:
        return len(self._chemsys)

    def __iter__(self) -> Iterator[str]:
        return iter(self.impl.chemical_systems)

    def __contains__(self, chemsys: object) -> bool:
        return chemsys in self._chemsys

    def __getitem__(self, chemsys: str) -> list[ComputedStructureEntry]:
        return list(self.impl.get_entries_by_chemsys(chemsys))


class LMDBBackedReducedFormulaLookup(_WeakRefMixin, Mapping[str, list[ComputedStructureEntry]]):
    def __init__(self, impl: LMDBBackedReferenceDatasetImpl):
        super().__init__(impl)
        self._rfs = frozenset(impl.reduced_formulas)

    def __len__(self) -> int:
        return len(self._rfs)

    def __iter__(self) -> Iterator[str]:
        return iter(self.impl.reduced_formulas)

    def __contains__(self, rf: object) -> bool:
        return rf in self._rfs

    def __getitem__(self, rf: str) -> list[ComputedStructureEntry]:
        return list(self.impl.get_entries_by_reduced_formula(rf))
