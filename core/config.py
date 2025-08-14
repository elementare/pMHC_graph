# core/config.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, List, Optional, Union
from pathlib import Path

# Apenas para tipagem opcional
try:
    import networkx as nx  # type: ignore
except Exception:
    nx = Any  # fallback de tipo

Granularity = Union[str]  # "centroids", "CA", "atom"
AltLocsOpts = Union[str, bool]  # "max_occupancy" | "min_occupancy" | "first" | "last" | "exclude" | "include" | True | False

@dataclass
class DSSPConfig:
    """
    Espelho do Graphein: apenas o executável.
    """
    executable: str = "mkdssp"   # troque para "dssp" se for o seu caso

@dataclass
class ProteinGraphConfig:
    """
    """
    granularity: Granularity = "centroids"
    keep_hets: List[str] = field(default_factory=list)
    insertions: bool = True
    alt_locs: AltLocsOpts = "max_occupancy"
    pdb_dir: Optional[Path] = None
    verbose: bool = False
    exclude_waters: bool = True   

    # Parâmetros que usamos no builder
    chains: Optional[Iterable[str]] = None
    residue_distance_cutoff: float = 10.0
    water_distance_cutoff: float = 6.0
    compute_rsa: bool = True

    # Funções estilo Graphein
    protein_df_processing_functions: Optional[List[Callable]] = None
    edge_construction_functions: List[Callable] = field(default_factory=list)
    node_metadata_functions: Optional[List[Callable]] = None
    edge_metadata_functions: Optional[List[Callable]] = None
    graph_metadata_functions: Optional[List[Callable]] = None

    # DSSP
    dssp_config: Optional[DSSPConfig] = field(default_factory=DSSPConfig)

def make_default_config(
    centroid_threshold: float,
    granularity: str = "centroids",
    exclude_waters: bool = False
) -> ProteinGraphConfig:
    """
    Config pronto para uso, no padrão Graphein (usa exclude_waters).
    """
    from functools import partial
    from core.edges import add_distance_threshold
    # você pode usar as funções do Graphein também (rsa/secondary_structure) no graph.py
    from core.metadata import rsa, secondary_structure

    return ProteinGraphConfig(
        granularity=granularity,
        exclude_waters=exclude_waters,  # False = inclui águas
        compute_rsa=True,
        edge_construction_functions=[
            partial(
                add_distance_threshold,
                threshold=centroid_threshold,
                long_interaction_threshold=0
            )
        ],
        graph_metadata_functions=[rsa, secondary_structure],
        dssp_config=DSSPConfig(executable="mkdssp"),
    )
