# core/pipeline.py
from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
import networkx as nx

from core.pdb_graph_builder import PDBGraphBuilder, GraphBuildConfig
from core.config import ProteinGraphConfig, DSSPConfig

def build_graph_with_config(pdb_path: str, config: ProteinGraphConfig) -> nx.Graph:
    """
    Usa a sua dataclass 'graphein-like' e mapeia diretamente para o builder.
    """
    include_waters = not bool(config.exclude_waters)  # padrão Graphein -> builder

    builder_cfg = GraphBuildConfig(
        chains=config.chains,
        include_waters=include_waters,
        residue_distance_cutoff=float(config.residue_distance_cutoff),
        water_distance_cutoff=float(config.water_distance_cutoff),
        compute_rsa=bool(config.compute_rsa),
        store_distance_matrix=True,
        rsa_method="dssp",
        dssp_exec=config.dssp_config.executable if config.dssp_config else "mkdssp",
        dssp_acc_array="Sander",  # Sander/Wilke/Miller — escolha aqui se quiser
    )

    builder = PDBGraphBuilder(pdb_path, builder_cfg)
    built = builder.build_graph()
    G = built.graph

    # Compatibilidade com funções do Graphein (ex.: rsa/secondary_structure):
    # dssp.py do Graphein olha para G.graph["config"].dssp_config.executable, "path", "name", "raw_pdb_df".
    G.graph["config"] = SimpleNamespace(
        dssp_config=SimpleNamespace(executable=builder_cfg.dssp_exec),
        verbose=config.verbose,
        insertions=config.insertions,
        pdb_dir=config.pdb_dir,
        granularity=config.granularity,
        exclude_waters=config.exclude_waters,
        alt_locs=config.alt_locs,
    )
    G.graph["path"] = pdb_path
    G.graph["name"] = Path(pdb_path).stem
    if getattr(built, "raw_pdb_df", None) is not None:
        G.graph["raw_pdb_df"] = built.raw_pdb_df

    # Aplicar funções definidas no config (se houver)
    ctx = {
        "pdb_path": pdb_path,
        "structure": builder.structure,
        "residue_map": {nid: res for nid, res in built.residue_index},
        "dssp_config": config.dssp_config,
        "config": config,
    }

    for fn in (config.edge_construction_functions or []):
        G = fn(G, **ctx)
    for fn in (config.node_metadata_functions or []):
        G = fn(G, **ctx)
    for fn in (config.edge_metadata_functions or []):
        G = fn(G, **ctx)
    for fn in (config.graph_metadata_functions or []):
        G = fn(G, **ctx)

    # Artefatos padrão (naming estilo Graphein + seu pedido)
    G.graph["residue_labels"]  = [nid for nid, _ in built.residue_index]
    G.graph["water_labels"]    = [nid for nid, _ in built.water_index]
    G.graph["distance_matrix"] = built.distance_matrix          # (R, R)
    G.graph["coords"]          = built.residue_centroids        # (R, 3)
    G.graph["water_positions"] = built.water_centroids          # (W, 3) ou None

    # DataFrames do builder (se habilitados)
    for key in ("raw_pdb_df", "pdb_df", "rgroup_df"):
        val = getattr(built, key, None)
        if val is not None:
            G.graph[key] = val

    return G

if __name__ == "__main__":
    from core.config import make_default_config
    cfg = make_default_config(centroid_threshold=8.5, exclude_waters=False)
    G = build_graph_with_config(
        "/home/elementare/GithubProjects/pMHC_graph/pdb_input/pmhc_titin_5bs0_renumber.pdb", cfg
    )
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    # print(G.nodes(data=True))
    # print(G.graph)
    # print("Graph-level keys:", sorted(G.graph.keys()))
