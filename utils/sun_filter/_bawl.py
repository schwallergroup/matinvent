"""Inline BAWL fingerprint for novelty checking.

Reproduces material_hasher.hasher.bawl.BAWLHasher (SPGLib symmetry variant)
without importing material-hasher, structuregraph_helpers, or moyopy.

Dependencies: pymatgen, networkx (already a pymatgen transitive dep), spglib.
"""
from __future__ import annotations

from collections import Counter
from hashlib import blake2b
from typing import Optional

import networkx as nx
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import EconNN
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# Default EconNN kwargs — matches material-hasher defaults.
_ECONNN_KWARGS: dict = {"tol": 0.2, "cutoff": 10, "use_fictive_radius": True}

# --------------------------------------------------------------------------- #
# Weisfeiler-Lehman graph hash                                                 #
# (BSD-licensed, adapted from structuregraph_helpers/_hasher.py which is      #
#  itself adapted from NetworkX; original copyright NetworkX Developers 2004)  #
# --------------------------------------------------------------------------- #

def _wl_hash(
    G: nx.Graph,
    *,
    node_attr: Optional[str] = "specie",
    edge_attr: Optional[str] = None,
    iterations: int = 100,
    digest_size: int = 16,
) -> str:
    def _h(label: str) -> str:
        return blake2b(label.encode("ascii"), digest_size=digest_size).hexdigest()

    def _aggregate(node: int, labels: dict) -> str:
        parts = []
        for nbr in G.neighbors(node):
            if isinstance(G, (nx.MultiDiGraph, nx.MultiGraph)):
                prefix = "" if edge_attr is None else str(G[node][nbr][0][edge_attr])
            else:
                prefix = "" if edge_attr is None else str(G[node][nbr][edge_attr])
            parts.append(prefix + labels[nbr])
        return labels[node] + "".join(sorted(parts))

    if node_attr:
        node_labels = {u: str(dd.get(node_attr, "")) for u, dd in G.nodes(data=True)}
    else:
        node_labels = {u: str(deg) for u, deg in G.degree()}

    counts: list = []
    for _ in range(iterations):
        node_labels = {n: _h(_aggregate(n, node_labels)) for n in G.nodes()}
        counts.extend(sorted(Counter(node_labels.values()).items()))

    return _h(str(tuple(counts)))


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #

def get_bawl_hash(
    structure: Structure,
    symprec: float = 0.01,
    bonding_kwargs: Optional[dict] = None,
) -> str:
    """Compute a BAWL fingerprint hash for *structure*.

    The hash encodes: bonding graph topology (WL), spacegroup number (SPGLib),
    and reduced composition — identical to BAWLHasher(symmetry_labeling='SPGLib').

    Parameters
    ----------
    structure:
        Input crystal structure.
    symprec:
        Symmetry tolerance passed to SpacegroupAnalyzer.
    bonding_kwargs:
        kwargs forwarded to EconNN (uses package defaults when None).

    Returns
    -------
    str
        Hash string of the form "<wl_hash>_<spg_number>_<formula>".
    """
    if bonding_kwargs is None:
        bonding_kwargs = _ECONNN_KWARGS

    # Build bonding graph
    sg = StructureGraph.with_local_env_strategy(structure, EconNN(**bonding_kwargs))
    graph = sg.graph
    for n, site in enumerate(structure):
        graph.nodes[n]["specie"] = site.specie.name
    for edge in graph.edges:
        graph.edges[edge]["voltage"] = graph.edges[edge]["to_jimage"]

    bonding_hash = _wl_hash(graph, node_attr="specie", edge_attr=None, iterations=100)

    try:
        spg_number: Optional[int] = SpacegroupAnalyzer(
            structure, symprec=symprec
        ).get_symmetry_dataset().number
    except Exception:
        spg_number = None

    composition = structure.composition.formula.replace(" ", "")
    return "_".join([bonding_hash, str(spg_number) if spg_number is not None else "", composition])
