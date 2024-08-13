from graphein.protein.config import ProteinGraphConfig
from graphein.protein.graphs import construct_graph
from graphein.protein.visualisation import plotly_protein_structure_graph
from graphein.protein.edges.distance import add_distance_threshold, compute_distmat
from graphein.protein.features.nodes.dssp import rsa, secondary_structure
from graphein.protein.config import DSSPConfig
from graphein.protein.subgraphs import extract_surface_subgraph, extract_subgraph, extract_subgraph_from_secondary_structure
from functools import partial
import networkx as nx
import matplotlib.pyplot as plt
from utils import check_identity, compute_atom_distance, build_contact_map, find_contact_residues, coefficient_of_variation, check_identity_same_chain, check_neighborhoods, create_sphere_residue, add_sphere_residues, create_subgraph_with_neighbors, check_cross_positions, graph_message_passing, check_similarity
from itertools import combinations, compress
import plotly.graph_objects as go
import numpy as np
from itertools import product
import pandas as pd
from pathlib import Path
from os import path
from typing import Callable, Any, Union, Optional 


class AssociatedGraph:
    def __init__(self, graphA, molA_path, graphB, molB_path, output_path, path_full_subgraph, run_name, association_mode = "identity",  interface_list = None, centroid_threshold = 10):
        if interface_list:
            self.interface_list = interface_list
        else:
            self.interface_list = None
            
        self.graphA = graphA
        self.graphB = graphB
        self.molA_path = molA_path
        self.molB_path = molB_path
        self.output_path = output_path
        self.run_name = run_name
        self.association_mode = association_mode
        Path(self.output_path).mkdir(parents=True, exist_ok=True)
        self.path_full_subgraph = path_full_subgraph
        
        self.associated_graph = self._construct_graph(self.graphA, self.molA_path, self.graphB, self.molB_path, self.association_mode )
        
    
    def _construct_graph(self, graphA, molA_path, graphB, molB_path, association_mode):
        #Create a contact map and a list with residue names order as the contact map
        #We do not need to do this, since graphein already build internally the distance matrix considering the centroid
        #This matrix can be obtained from: graph.graph["pdb_df"]
        #The contact matrix can be obtained from: graph.graph["dist_mat"]
        #Usually this dist_mat is not updated in related to the pdb_df after graph subsets, so it is important to double check it 
        #Instead of using the dist_mat we can build it again using compute_distmat(graph.graph["pdb_df"])
        contact_map1, residue_map1, residue_map1_all = build_contact_map(molA_path) #_all means with resname
        contact_map2, residue_map2, residue_map2_all = build_contact_map(molB_path)
        
        #Run the cartesian product to create the association graphs
        #This can be substituted for just getting the combinations of node names, no need to build the original graphs!
        M = nx.cartesian_product(graphA, graphB)
        
        if association_mode == 'identity':
            #Filter the pair of nodes based on identity, same chain and neighborhoods
            select_ident = [i for i in list(M) if check_identity(i) and check_identity_same_chain(i) and check_neighborhoods(i, contact_map1, residue_map1_all, contact_map2, residue_map2_all)]
        
        elif association_mode == 'similarity':
            message_passing_molA = graph_message_passing(graphA, 'resources/atchley_aa.csv', use_degree=False, norm_features=False)
            message_passing_molB = graph_message_passing(graphB, 'resources/atchley_aa.csv', use_degree=False, norm_features=False)
            #Filter the pair of nodes 
            select_ident = [i for i in list(M) if check_similarity(i, message_passing_molA, message_passing_molB, threshold=0.95) and check_identity_same_chain(i)]
        else:
            print(f'Mode {association_mode} is not a valid mode')
            quit 
            
        #Create the association graphs
        paired_graphs = [list(pair) for pair in combinations(select_ident, 2)]

        #Filter out pairs of pairs (edges) based on cross positions
        #For instance, if an association node has a pair with same positions 169 and 169 and its paired association node has a pair with different positions (169 and 170, for instance), it does not make sense to connect these nodes
        paired_graphs_filtered = [i for i in paired_graphs if check_cross_positions(i)]

        #To build the edges between association nodes, we first calculate the d1 distance, then d2 distance and then the d1d2 ratio
        #calculate d1 distance
        d1 = [find_contact_residues(contact_map1, residue_map1, (i[0][0].split(':')[0],int(i[0][0].split(':')[2])),  (i[1][0].split(':')[0],int(i[1][0].split(':')[2]))) for i in paired_graphs_filtered]
        #calculate d2 distance
        d2 = [find_contact_residues(contact_map2, residue_map2, (i[0][1].split(':')[0],int(i[0][1].split(':')[2])),  (i[1][1].split(':')[0],int(i[1][1].split(':')[2]))) for i in paired_graphs_filtered]
        #calculate d1d2 ratio distance using coefficient of variation
        d1d2 = [coefficient_of_variation(a,b) for a,b in zip(d1,d2)]
        #generate the pair of connected nodes indicating the edges
        new_nodes_edges = [paired_graphs_filtered[n] for n,m in enumerate(d1) if d1[n] < 10 and d2[n] < 10 and d1[n] > 0 and d2[n] > 0 and d1d2[n] < 15 and (paired_graphs_filtered[n][0][0] != paired_graphs_filtered[n][1][1] and paired_graphs_filtered[n][0][1] != paired_graphs_filtered[n][1][0])]

        # Build the new graph
        G_sub = nx.Graph()
        # Add nodes
        for sublist in new_nodes_edges:
            for edge in sublist:
                G_sub.add_node(edge[0])
                G_sub.add_node(edge[1])
        # Add edges
        for sublist in new_nodes_edges:
            G_sub.add_edge(sublist[0], sublist[1])

        # Remove nodes with no edges
        G_sub.remove_nodes_from(list(nx.isolates(G_sub)))

        # Add chain as attribute to color the nodes 
        for nodes in G_sub.nodes:
            if nodes[0].startswith('A') and nodes[1].startswith('A'):
                G_sub.nodes[nodes]['chain_id'] = 'red'
            elif nodes[0].startswith('C') and nodes[1].startswith('C'):
                G_sub.nodes[nodes]['chain_id'] = 'blue'
            else:
                G_sub.nodes[nodes]['chain_id'] = None
        
        return G_sub 
    
    def draw_graph(self, show = True, save = True):
        if not show and not save:
            print("You are not saving or viewing the graph. Please leave at least one of the parameters as true.")
        else:
            node_colors = [self.associated_graph.nodes[node]['chain_id'] for node in self.associated_graph.nodes]
            # Draw the full cross-reactive subgraph
            nx.draw(self.associated_graph, with_labels=True, node_color=node_colors, node_size=50, font_size=6)
            
            if show:
                plt.show()
            if save:
                plt.savefig(self.path_full_subgraph)
                plt.clf()
    
                print(f"GraphAssociated's plot saved in {self.path_full_subgraph}")
  
    def grow_subgraph_bfs(self):
        # Build all possible common TCR interface pMHC subgraphs centered at the peptide nodes 
        count_pep_nodes = 0
        G_sub = self.associated_graph
        for nodes in G_sub.nodes:
            if nodes[0].startswith('C') and nodes[1].startswith('C'): #i.e. peptide nodes
                count_pep_nodes += 1
                #print(nodes)
                bfs_subgraph = create_subgraph_with_neighbors(self.graphA, self.graphB, G_sub, nodes, 20)
                for nodes2 in bfs_subgraph.nodes:
                    if nodes2[0].startswith('A') and nodes2[1].startswith('A'):
                        bfs_subgraph.nodes[nodes2]['chain_id'] = 'red'
                    elif nodes2[0].startswith('C') and nodes2[1].startswith('C'):
                        bfs_subgraph.nodes[nodes2]['chain_id'] = 'blue'
                    else:
                        bfs_subgraph.nodes[nodes2]['chain_id'] = None
                #check whether nodes meet the requirements to be a TCR interface
                number_peptide_nodes = len([i for i in bfs_subgraph.nodes if i[0].startswith('C')])
                if bfs_subgraph.number_of_nodes() >= 14 and nx.diameter(bfs_subgraph) >= 3 and number_peptide_nodes >=3:
                    node_colors = [bfs_subgraph.nodes[node]['chain_id'] for node in bfs_subgraph.nodes]
                    nx.draw(bfs_subgraph, with_labels=True, node_color=node_colors)
                    plt.savefig(path.join(self.output_path,f'plot_bfs_{nodes[0]}_{self.run_name}.png'))
                    plt.clf()

                    #write PDB with the common subgraphs as spheres
                    get_node_names = list(bfs_subgraph.nodes())
                    node_names_molA = [i[0] for i in get_node_names]
                    node_names_molB = [i[1] for i in get_node_names]

                    #create PDB with spheres
                    add_sphere_residues(node_names_molA, self.molA_path, node_names_molB, self.molB_path, self.output_path, nodes[0])
                    #add_sphere_residues(node_names_molB, path_protein2, path_protein2.split('.pdb')[0]+'_spheres.pdb')
                else:
                    print(f"The subgraph centered at the {nodes[0]} node does not satisfies the requirements")
        if count_pep_nodes == 0:
            print(f'No peptide nodes were found in the association graph. No subgraph will be generated.')
        pass
        
    
class Graph:
    def __init__(self, graph_path, config = None):
        if config:
            self.config = config
        else:
            self.config = ProteinGraphConfig(edge_construction_functions=[partial(add_distance_threshold, long_interaction_threshold=0, threshold=centroid_threshold)],
                            graph_metadata_functions=[rsa, secondary_structure], 
                            dssp_config=DSSPConfig(),
                            granularity="centroids")

            
        self.graph = construct_graph(config=config, path=graph_path)
        
        self.subgraphs = {}
    
    def get_subgraph(self, name:str):
        if name not in self.subgraphs.keys():
            print(f"Can't find {name} in subgraph")
        else:
            return self.subgraphs[name]
    
    def create_subgraph(self, name: str, node_list: list = [], return_node_list: bool = False, **args):
        if name in self.subgraphs.keys():
            print(f"You already have this subgraph created. Use graph.delete_subraph({name}) before creating it again.")
        elif not node_list:
            self.subgraphs[name] =  extract_subgraph(g = self.graph, **args)
            print(f"Subgraph {name} created with success!")
        elif node_list:
            self.subgraphs[name] = self.graph.subgraph(node_list)
        
        if return_node_list:
            return self.subgraphs[name].nodes
        
    def delete_subraph(self, name: str):
        if name not in  self.subgraphs.keys():
            del self.subgraphs[name]
        else:
            print(f"{name} isn't in.subgraphs")
            
    def filter_subgraph(self, 
            subgraph_name: str,
            filter_func: Union[Callable[..., Any], str],
            name: Union[str, None] = None, 
            return_node_list: bool = False):
           
        nodes = [i for i in self.subgraphs[subgraph_name].nodes if filter_func(i)]
        if name:
            self.subgraphs[name] = self.subgraphs[subgraph_name].subgraph(nodes)
        else:
            self.subgraphs[subgraph_name] = self.subgraphs[subgraph_name].subgraph(nodes)
        
        if return_node_list:
            return self.subgraphs[name].nodes if name != None else self.subgraphs[subgraph_name].nodes
        
    def join_subgraph(self, name: str, graphs_name: list, return_node_list: bool = False):
        if name in self.subgraphs.keys():
            print(f"You already have this subgraph created. Use graph.delete_subraph({name}) before creating it again.")
        elif set(graphs_name).issubset(self.subgraphs.keys()):
            
            self.subgraphs[name] = self.graph.subgraph([node for i in graphs_name for node in self.subgraphs[i].nodes])
            if return_node_list:
                return self.subgraphs[name].nodes
        else:
            print(f"Some of your subgraph isn't in the subgraph list")
        
        
        
        
    
        

