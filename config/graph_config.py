from functools import partial
from graphein.protein.config import ProteinGraphConfig
from graphein.protein.edges.distance import add_distance_threshold
from graphein.protein.features.nodes.dssp import rsa, secondary_structure
from graphein.protein.config import DSSPConfig

def make_graph_config(centroid_threshold):
    return ProteinGraphConfig(
        edge_construction_functions=[partial(add_distance_threshold, long_interaction_threshold=0, threshold=centroid_threshold)],
        graph_metadata_functions=[rsa, secondary_structure],
        dssp_config=DSSPConfig(),
        granularity="centroids"
    )
