from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Union

import logging
import networkx as nx
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# -------------------------- utilidades & erros --------------------------

class ProteinGraphConfigurationError(RuntimeError):
    """Erro de configuração/anotação do grafo."""


def _node_attr(d: dict, key: str, *fallbacks: str):
    for k in (key, *fallbacks):
        if k in d and d[k] is not None:
            return d[k]
    return None


def _node_coords(d: dict) -> Optional[np.ndarray]:
    """Retorna coordenadas do nó como np.ndarray, usando 'coords' ou 'centroid'."""
    c = _node_attr(d, "coords", "centroid")
    if c is None:
        return None
    arr = np.asarray(c, dtype=float)
    return arr


def _ensure_set(x) -> set:
    """Converte um rótulo de ligação em set de strings."""
    if x is None:
        return set()
    if isinstance(x, str):
        return {x}
    try:
        return set(str(v) for v in x)
    except Exception:
        return {str(x)}


def _update_coords_graph(g: nx.Graph) -> None:
    """Atualiza g.graph['coords'] = Nx3, g.graph['residue_labels'] alinhado."""
    labels = list(g.nodes())
    coords = []
    missing = 0
    for n in labels:
        arr = _node_coords(g.nodes[n])
        if arr is None:
            missing += 1
            arr = np.array([np.nan, np.nan, np.nan], dtype=float)
        coords.append(arr)
    g.graph["residue_labels"] = labels
    g.graph["coords"] = np.vstack(coords) if coords else np.zeros((0, 3), dtype=float)
    if missing:
        log.debug(f"[subgraph] {missing} nós sem coords/centroid; preenchidos com NaN.")


def compute_distmat(pdb_df: pd.DataFrame) -> np.ndarray:
    """
    Reimplementação leve do compute_distmat do Graphein:
      - Agrupa pdb_df por 'node_id'
      - Faz média (x,y,z) por nó
      - Retorna matriz de distâncias Euclidianas NxN (ordem = primeira ocorrência por node_id)
    """
    if pdb_df is None or len(pdb_df) == 0:
        return np.zeros((0, 0), dtype=float)

    seen, order = set(), []
    for nid in pdb_df["node_id"].tolist():
        if nid not in seen:
            seen.add(nid)
            order.append(nid)

    grouped = (
        pdb_df.groupby("node_id")[["x_coord", "y_coord", "z_coord"]]
        .mean()
        .reindex(order)
    )
    P = grouped.to_numpy(dtype=float)  # (N,3)
    if P.size == 0:
        return np.zeros((0, 0), dtype=float)

    diff = P[:, None, :] - P[None, :, :]
    D = np.sqrt((diff * diff).sum(axis=2))
    return D


# -------------------------- núcleo de subgráfos --------------------------

def _filter_df_by_nodes(df, node_list):
    """Filtra um DataFrame do graphein pela coluna/index 'node_id'."""
    if df is None:
        return None
    try:
        if "node_id" in df.columns:
            return df[df["node_id"].isin(node_list)].copy()
    except Exception:
        pass
    # tentar pelo índice
    try:
        idx_name = getattr(df.index, "name", None)
        if idx_name == "node_id" or idx_name is not None:
            return df[df.index.isin(node_list)].copy()
    except Exception:
        pass
    return df.copy()

def _carry_graph_level(parent: nx.Graph,
                       child: nx.Graph,
                       node_list: List[str],
                       filter_dataframe: bool,
                       update_coords: bool,
                       recompute_distmat: bool):
    """
    Copia e ajusta artefatos de nível-grafo do grafo pai para o subgrafo.
    """
    # sempre carregar alguns metadados básicos
    for k in ("config", "path", "name"):
        if k in parent.graph:
            child.graph[k] = parent.graph[k]

    # dssp_df (crítico p/ RSA)
    if "dssp_df" in parent.graph:
        child.graph["dssp_df"] = (
            _filter_df_by_nodes(parent.graph["dssp_df"], node_list)
            if filter_dataframe else parent.graph["dssp_df"]
        )

    # DataFrames PDB
    for key in ("pdb_df", "raw_pdb_df", "rgroup_df"):
        if key in parent.graph:
            child.graph[key] = (
                _filter_df_by_nodes(parent.graph[key], node_list)
                if filter_dataframe else parent.graph[key]
            )
            if filter_dataframe and hasattr(child.graph[key], "reset_index"):
                child.graph[key] = child.graph[key].reset_index(drop=True)

    # coords (graphein-style): se não vamos recomputar, propagar
    if "coords" in parent.graph and not update_coords:
        # manter mesma ordem dos nós do subgrafo
        coords_map = {n: c for n, c in zip(parent.nodes(), parent.graph["coords"])}
        child.graph["coords"] = np.array([coords_map[n] for n in child.nodes()])

    # distance_matrix: se não recomputar, tentar fatiar a existente
    if "distance_matrix" in parent.graph and not recompute_distmat:
        try:
            # precisamos do mapeamento node_id -> índice
            labels = parent.graph.get("residue_labels")
            if labels:
                pos = {nid: i for i, nid in enumerate(labels)}
                idx = [pos[n] for n in node_list if n in pos]
                dm = parent.graph["distance_matrix"]
                child.graph["distance_matrix"] = dm[np.ix_(idx, idx)]
        except Exception as e:
            log.debug(f"Não consegui fatiar distance_matrix: {e}")

    # rótulos auxiliares
    child.graph["residue_labels"] = list(child.nodes())
    if "water_labels" in parent.graph:
        # por consistência, filtra se alguma água estiver presente no subgrafo
        child.graph["water_labels"] = [n for n in parent.graph["water_labels"] if n in child.nodes()]
    if "water_positions" in parent.graph:
        child.graph["water_positions"] = parent.graph["water_positions"]  # manter como está (não depende só do subgrafo)


# --------- funções públicas (estilo graphein) ---------
def extract_subgraph_from_node_list(
    g: nx.Graph,
    node_list: Optional[List[str]],
    filter_dataframe: bool = True,
    update_coords: bool = True,
    recompute_distmat: bool = False,
    inverse: bool = False,
    return_node_list: bool = False,
) -> Union[nx.Graph, List[str]]:
    if node_list:
        if inverse:
            node_list = [n for n in g.nodes() if n not in node_list]
        if return_node_list:
            return node_list

        log.debug(f"Creating subgraph from nodes: {node_list}.")
        parent = g
        sub = parent.subgraph(node_list).copy()

        # opcional: filtrar DataFrames e recomputar artefatos
        if filter_dataframe and "pdb_df" in parent.graph:
            sub.graph["pdb_df"] = _filter_df_by_nodes(parent.graph["pdb_df"], node_list)
            sub.graph["pdb_df"] = sub.graph["pdb_df"].reset_index(drop=True)

        # Atualizar coords dos nós (preferir atributo 'coords', senão 'centroid')
        if update_coords:
            coords = []
            for _, d in sub.nodes(data=True):
                if "coords" in d and d["coords"] is not None:
                    coords.append(np.asarray(d["coords"], dtype=float))
                elif "centroid" in d and d["centroid"] is not None:
                    coords.append(np.asarray(d["centroid"], dtype=float))
                else:
                    # fallback: se pai tinha coords por nó na mesma ordem
                    coords.append(np.zeros(3, dtype=float))
            sub.graph["coords"] = np.vstack(coords) if len(coords) else np.zeros((0, 3), dtype=float)

        # Recomputar distmat se pedido e se houver pdb_df; se não, deixamos para o caller
        if recompute_distmat:
            try:
                from graphein.protein.edges.distance import compute_distmat
                if not filter_dataframe and "pdb_df" not in sub.graph and "pdb_df" in parent.graph:
                    log.warning("Recomputing distmat sem filtrar pdb_df; copiando parent.pdb_df filtrado")
                    sub.graph["pdb_df"] = _filter_df_by_nodes(parent.graph["pdb_df"], node_list).reset_index(drop=True)
                sub.graph["dist_mat"] = compute_distmat(sub.graph["pdb_df"])
            except Exception as e:
                log.debug(f"Falha ao recomputar dist_mat: {e}")

        # Propagar artefatos de nível-grafo (inclui dssp_df!)
        _carry_graph_level(parent, sub, node_list, filter_dataframe, update_coords, recompute_distmat)

        return sub

    return node_list if return_node_list else g


def extract_subgraph_from_point(
    g: nx.Graph,
    centre_point: Union[np.ndarray, Tuple[float, float, float]],
    radius: float,
    filter_dataframe: bool = True,
    update_coords: bool = True,
    recompute_distmat: bool = False,
    inverse: bool = False,
    return_node_list: bool = False,
) -> Union[nx.Graph, List[str]]:
    node_list: List[str] = []
    cp = np.asarray(centre_point, dtype=float)

    for n, d in g.nodes(data=True):
        coords = d.get("coords", d.get("centroid", None))
        if coords is None:
            continue
        dist = np.linalg.norm(np.asarray(coords, dtype=float) - cp)
        if dist < radius:
            node_list.append(n)

    node_list = list(set(node_list))
    log.debug(f"Found {len(node_list)} nodes in the spatial point-radius subgraph.")

    return extract_subgraph_from_node_list(
        g,
        node_list,
        filter_dataframe=filter_dataframe,
        inverse=inverse,
        return_node_list=return_node_list,
        recompute_distmat=recompute_distmat,
        update_coords=update_coords,
    )



def extract_subgraph_from_atom_types(
    g: nx.Graph,
    atom_types: List[str],
    filter_dataframe: bool = True,
    update_coords: bool = True,
    recompute_distmat: bool = False,
    inverse: bool = False,
    return_node_list: bool = False,
) -> Union[nx.Graph, List[str]]:
    node_list: List[str] = [
        n for n, d in g.nodes(data=True) if _node_attr(d, "atom_type") in set(atom_types)
    ]

    node_list = list(dict.fromkeys(node_list))
    log.debug(f"[subgraph] atom types: {len(node_list)} nós")

    return extract_subgraph_from_node_list(
        g,
        node_list,
        filter_dataframe=filter_dataframe,
        inverse=inverse,
        return_node_list=return_node_list,
        update_coords=update_coords,
        recompute_distmat=recompute_distmat,
    )


def extract_subgraph_from_residue_types(
    g: nx.Graph,
    residue_types: List[str],
    filter_dataframe: bool = True,
    update_coords: bool = True,
    recompute_distmat: bool = False,
    inverse: bool = False,
    return_node_list: bool = False,
) -> Union[nx.Graph, List[str]]:
    residue_types = set(residue_types)
    node_list: List[str] = []
    for n, d in g.nodes(data=True):
        rname = _node_attr(d, "residue_name", "resname")
        if rname in residue_types:
            node_list.append(n)

    node_list = list(dict.fromkeys(node_list))
    log.debug(f"[subgraph] residue types: {len(node_list)} nós")

    return extract_subgraph_from_node_list(
        g,
        node_list,
        filter_dataframe=filter_dataframe,
        inverse=inverse,
        return_node_list=return_node_list,
        update_coords=update_coords,
        recompute_distmat=recompute_distmat,
    )


def extract_subgraph_from_chains(
    g: nx.Graph,
    chains: List[str],
    filter_dataframe: bool = True,
    update_coords: bool = True,
    recompute_distmat: bool = False,
    inverse: bool = False,
    return_node_list: bool = False,
) -> Union[nx.Graph, List[str]]:
    chains = set(chains)
    node_list: List[str] = []
    for n, d in g.nodes(data=True):
        cid = _node_attr(d, "chain_id", "chain")
        if cid in chains:
            node_list.append(n)

    node_list = list(dict.fromkeys(node_list))
    log.debug(f"[subgraph] chains: {len(node_list)} nós")

    return extract_subgraph_from_node_list(
        g,
        node_list,
        filter_dataframe=filter_dataframe,
        update_coords=update_coords,
        recompute_distmat=recompute_distmat,
        inverse=inverse,
        return_node_list=return_node_list,
    )


def extract_subgraph_by_sequence_position(
    g: nx.Graph,
    sequence_positions: List[int],
    filter_dataframe: bool = True,
    update_coords: bool = True,
    recompute_distmat: bool = False,
    inverse: bool = False,
    return_node_list: bool = False,
) -> Union[nx.Graph, List[str]]:
    seqset = set(int(x) for x in sequence_positions)
    node_list: List[str] = []
    for n, d in g.nodes(data=True):
        rnum = _node_attr(d, "residue_number", "resseq")
        if rnum in seqset:
            node_list.append(n)

    node_list = list(dict.fromkeys(node_list))
    log.debug(f"[subgraph] sequence positions: {len(node_list)} nós")

    return extract_subgraph_from_node_list(
        g,
        node_list,
        filter_dataframe=filter_dataframe,
        inverse=inverse,
        return_node_list=return_node_list,
        recompute_distmat=recompute_distmat,
        update_coords=update_coords,
    )


def extract_subgraph_by_bond_type(
    g: nx.Graph,
    bond_types: List[str],
    filter_dataframe: bool = True,
    update_coords: bool = True,
    recompute_distmat: bool = False,
    inverse: bool = False,
    return_node_list: bool = False,
) -> Union[nx.Graph, List[str]]:
    bond_types = set(bond_types)
    node_list: List[str] = []

    for u, v, d in g.edges(data=True):
        kinds = _ensure_set(d.get("kind"))
        if kinds & bond_types:
            node_list.append(u)
            node_list.append(v)

    node_list = list(dict.fromkeys(node_list))
    log.debug(f"[subgraph] bond types: {len(node_list)} nós")

    # Atualiza anotação de arestas (mantendo só os tipos pedidos / removendo, se inverse=True)
    for _, _, d in g.edges(data=True):
        kinds = _ensure_set(d.get("kind"))
        if not inverse:
            kinds = {k for k in kinds if k in bond_types}
        else:
            kinds = {k for k in kinds if k not in bond_types}
        d["kind"] = kinds if len(kinds) != 1 else next(iter(kinds), None)

    return extract_subgraph_from_node_list(
        g,
        node_list,
        filter_dataframe=filter_dataframe,
        inverse=inverse,
        return_node_list=return_node_list,
        recompute_distmat=recompute_distmat,
        update_coords=update_coords,
    )


def extract_subgraph_from_secondary_structure(
    g: nx.Graph,
    ss_elements: List[str],
    inverse: bool = False,
    filter_dataframe: bool = True,
    recompute_distmat: bool = False,
    update_coords: bool = True,
    return_node_list: bool = False,
) -> Union[nx.Graph, List[str]]:
    node_list: List[str] = []

    for n, d in g.nodes(data=True):
        if "ss" not in d:
            raise ProteinGraphConfigurationError(
                f"Secondary structure not set for node {n}. "
                "Execute uma etapa que preencha 'ss' (DSSP)."
            )
        if d["ss"] in set(ss_elements):
            node_list.append(n)

    node_list = list(dict.fromkeys(node_list))
    log.debug(f"[subgraph] secondary structure: {len(node_list)} nós")

    return extract_subgraph_from_node_list(
        g,
        node_list,
        inverse=inverse,
        return_node_list=return_node_list,
        filter_dataframe=filter_dataframe,
        recompute_distmat=recompute_distmat,
        update_coords=update_coords,
    )


def extract_surface_subgraph(
    g: nx.Graph,
    rsa_threshold: float = 0.2,
    inverse: bool = False,
    filter_dataframe: bool = True,
    recompute_distmat: bool = False,
    update_coords: bool = True,
    return_node_list: bool = False,
) -> Union[nx.Graph, List[str]]:
    node_list: List[str] = []

    for n, d in g.nodes(data=True):
        if "rsa" not in d:
            raise ProteinGraphConfigurationError(
                f"RSA not set para {n}. Rode anotação via DSSP."
            )
        if d["rsa"] is not None and float(d["rsa"]) >= float(rsa_threshold):
            node_list.append(n)

    node_list = list(dict.fromkeys(node_list))
    log.debug(f"[subgraph] surface (rsa≥{rsa_threshold}): {len(node_list)} nós")

    return extract_subgraph_from_node_list(
        g,
        node_list,
        inverse=inverse,
        return_node_list=return_node_list,
        filter_dataframe=filter_dataframe,
        recompute_distmat=recompute_distmat,
        update_coords=update_coords,
    )


def extract_k_hop_subgraph(
    g: nx.Graph,
    central_node: str,
    k: int,
    k_only: bool = False,
    filter_dataframe: bool = True,
    update_coords: bool = True,
    recompute_distmat: bool = False,
    inverse: bool = False,
    return_node_list: bool = False,
) -> Union[nx.Graph, List[str]]:
    neighbours: Dict[int, Union[List[str], set]] = {0: [central_node]}

    for i in range(1, int(k) + 1):
        neighbours[i] = set()
        for node in neighbours[i - 1]:
            neighbours[i].update(g.neighbors(node))
        neighbours[i] = list(set(neighbours[i]))

    if k_only:
        node_list = neighbours[int(k)]
    else:
        node_list = list({v for values in neighbours.values() for v in values})

    log.debug(f"[subgraph] k-hop (k={k}, only={k_only}): {len(node_list)} nós")

    return extract_subgraph_from_node_list(
        g,
        node_list,
        filter_dataframe=filter_dataframe,
        inverse=inverse,
        return_node_list=return_node_list,
        recompute_distmat=recompute_distmat,
        update_coords=update_coords,
    )


def extract_interface_subgraph(
    g: nx.Graph,
    interface_list: Optional[List[str]] = None,
    chain_list: Optional[List[str]] = None,
    filter_dataframe: bool = True,
    update_coords: bool = True,
    recompute_distmat: bool = False,
    inverse: bool = False,
    return_node_list: bool = False,
) -> Union[nx.Graph, List[str]]:
    node_list: List[str] = []

    for u, v in g.edges():
        u_chain = _node_attr(g.nodes[u], "chain_id", "chain")
        v_chain = _node_attr(g.nodes[v], "chain_id", "chain")
        if u_chain is None or v_chain is None:
            continue

        # filtros
        if chain_list is not None:
            if u_chain in chain_list and v_chain in chain_list and u_chain != v_chain:
                node_list.extend((u, v))
        if interface_list is not None:
            case_1 = f"{u_chain}{v_chain}"
            case_2 = f"{v_chain}{u_chain}"
            if case_1 in interface_list or case_2 in interface_list:
                node_list.extend((u, v))
        if chain_list is None and interface_list is None and u_chain != v_chain:
            node_list.extend((u, v))

    node_list = list(dict.fromkeys(node_list))
    log.debug(f"[subgraph] interface: {len(node_list)} nós")

    return extract_subgraph_from_node_list(
        g,
        node_list,
        filter_dataframe=filter_dataframe,
        inverse=inverse,
        return_node_list=return_node_list,
        recompute_distmat=recompute_distmat,
        update_coords=update_coords,
    )


def extract_subgraph(
    g: nx.Graph,
    node_list: Optional[List[str]] = None,
    sequence_positions: Optional[List[int]] = None,
    chains: Optional[List[str]] = None,
    residue_types: Optional[List[str]] = None,
    atom_types: Optional[List[str]] = None,
    bond_types: Optional[List[str]] = None,
    centre_point: Optional[Union[np.ndarray, Tuple[float, float, float]]] = None,
    radius: Optional[float] = None,
    ss_elements: Optional[List[str]] = None,
    rsa_threshold: Optional[float] = None,
    k_hop_central_node: Optional[str] = None,
    k_hops: Optional[int] = None,
    k_only: Optional[bool] = None,
    filter_dataframe: bool = True,
    update_coords: bool = True,
    recompute_distmat: bool = False,
    inverse: bool = False,
    return_node_list: bool = False,
) -> Union[nx.Graph, List[str]]:
    """Função agregadora com a mesma API do Graphein."""
    if node_list is None:
        node_list = []

    if sequence_positions is not None:
        node_list += extract_subgraph_by_sequence_position(
            g, sequence_positions, return_node_list=True
        )
    if chains is not None:
        node_list += extract_subgraph_from_chains(
            g, chains, return_node_list=True
        )
    if residue_types is not None:
        node_list += extract_subgraph_from_residue_types(
            g, residue_types, return_node_list=True
        )
    if atom_types is not None:
        node_list += extract_subgraph_from_atom_types(
            g, atom_types, return_node_list=True
        )
    if bond_types is not None:
        node_list += extract_subgraph_by_bond_type(
            g, bond_types, return_node_list=True
        )
    if centre_point is not None and radius is not None:
        node_list += extract_subgraph_from_point(
            g, centre_point, radius, return_node_list=True
        )
    if ss_elements is not None:
        node_list += extract_subgraph_from_secondary_structure(
            g, ss_elements, return_node_list=True
        )
    if rsa_threshold is not None:
        node_list += extract_surface_subgraph(
            g, rsa_threshold, return_node_list=True
        )
    if k_hop_central_node is not None and k_hops and k_only is not None:
        node_list += extract_k_hop_subgraph(
            g, k_hop_central_node, k_hops, k_only, return_node_list=True
        )

    # unique, preservando ordem de primeira ocorrência
    seen, merged = set(), []
    for n in node_list:
        if n not in seen:
            seen.add(n)
            merged.append(n)

    return extract_subgraph_from_node_list(
        g,
        merged,
        filter_dataframe=filter_dataframe,
        inverse=inverse,
        return_node_list=return_node_list,
        recompute_distmat=recompute_distmat,
        update_coords=update_coords,
    )
