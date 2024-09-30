import networkx as nx
import requests
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import pandas as pd
import torch.nn.functional as F
from torch_geometric.data import HeteroData
import tqdm
import torch_geometric.transforms as T
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import SAGEConv, to_hetero
import torch.nn.functional as F
from torch import Tensor
import pickle



from torch_geometric.nn import SAGEConv, to_hetero
import torch.nn.functional as F
from torch import Tensor
class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, 64)
    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        return x
# Our final classifier applies the dot-product between source and destination
# node embeddings to derive edge-level predictions:
class Classifier(torch.nn.Module):
    def forward(self, x_drug: Tensor, x_disorder: Tensor, edge_label_index: Tensor) -> Tensor:
        # Convert node embeddings to edge-level representations:
        edge_feat_drug = x_drug[edge_label_index[0]]
        edge_feat_disorder = x_disorder[edge_label_index[1]]
        # Apply dot-product to get a prediction per supervision edge:
        return (edge_feat_drug * edge_feat_disorder).sum(dim=-1)


class Model(torch.nn.Module):
    def __init__(self, hidden_channels, data):
        super().__init__()
        # Since the dataset does not come with rich features, we also learn two
        # embedding matrices for users and movies:
        self.disorder_lin = torch.nn.Linear(768, hidden_channels)
        self.drug_lin = torch.nn.Linear(768, hidden_channels)
        self.protein_lin = torch.nn.Linear(768, hidden_channels)
        self.pathway_lin = torch.nn.Linear(768, hidden_channels)
        self.gene_lin = torch.nn.Linear(768, hidden_channels)
        self.signature_lin = torch.nn.Linear(768, hidden_channels)
        self.tissue_lin = torch.nn.Linear(768, hidden_channels)
        self.side_effect_lin = torch.nn.Linear(768, hidden_channels)
        self.phenotype_lin = torch.nn.Linear(768, hidden_channels)
        self.go_lin = torch.nn.Linear(768, hidden_channels)
        
        # Instantiate homogeneous GNN:
        self.gnn = GNN(hidden_channels)
        # Convert GNN model into a heterogeneous variant:
        self.gnn = to_hetero(self.gnn, metadata=data.metadata())
        self.classifier = Classifier()
    def forward(self, data: HeteroData) -> Tensor:
        x_dict = {
          "drug": self.drug_lin(data["drug"].x),
          "disorder": self.disorder_lin(data["disorder"].x),
          "protein": self.protein_lin(data["protein"].x),
          "gene": self.gene_lin(data["gene"].x),
          "pathway": self.pathway_lin(data["pathway"].x),
          "signature": self.signature_lin(data["signature"].x),
          "tissue": self.tissue_lin(data["tissue"].x),
          "side_effect": self.side_effect_lin(data["side_effect"].x),
          "phenotype": self.phenotype_lin(data["phenotype"].x),
          "go": self.go_lin(data["go"].x),
            

        } 
        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        pred = self.classifier(
            x_dict["drug"],
            x_dict["disorder"],
            data["drug", "DrugHasIndication", "disorder"].edge_label_index,
        )
        return pred

    def return_embeddings(self, data: HeteroData): 

        x_dict = {
          "drug": self.drug_lin(data["drug"].x),
          "disorder": self.disorder_lin(data["disorder"].x),
          "protein": self.protein_lin(data["protein"].x),
          "gene": self.gene_lin(data["gene"].x),
          "pathway": self.pathway_lin(data["pathway"].x),
          "signature": self.signature_lin(data["signature"].x),
          "tissue": self.tissue_lin(data["tissue"].x),
          "side_effect": self.side_effect_lin(data["side_effect"].x),
          "phenotype": self.phenotype_lin(data["phenotype"].x),
          "go": self.go_lin(data["go"].x),
            

        } 
        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        
        return x_dict


def load_graph():
    
    print('Loading Graph...')
    
    G = pickle.load(open('graph_v3.pkl', 'rb'))
                
    print('Graph loaded!')
                
    
    return G


def create_new_graph(nodes, subgraph): 
    
    print('Creating new graph...')
    
    data = HeteroData()
    
    edges_dic = nx.get_edge_attributes(subgraph, 'edge_name')
    
    node_names = ["disorder","drug","gene","go","pathway","phenotype","protein","side_effect","signature","tissue"]
    nodes['Index'] = None

    # Assuming nodes and edges_dic are defined appropriately

    # Create a dictionary to store category indices
    category_indices = {n: nodes[nodes['Category'] == n].index for n in node_names}

    # Update 'Index' column in nodes DataFrame
    for n, indices in category_indices.items():
        nodes.loc[indices, 'Index'] = range(len(indices))

    # Create a dictionary to map node names to their indices
    node_index_map = dict(zip(nodes['Nodes Name'], nodes['Index']))

    # Create a dictionary to store new edges with updated indices
    new_edges = []
    for (node1, node2), val in edges_dic.items():
        if node1 in node_index_map and node2 in node_index_map:
            index1, index2 = node_index_map[node1], node_index_map[node2]
            new_edges.append((index1, index2, val))
        else:
            print(f"Nodes not found for edge: ({node1}, {node2})")

    new_edges_df = pd.DataFrame(new_edges, columns=['Source', 'Target', 'Value'])
            
    
    for i in node_names: 
        feat = nodes[nodes['Category'] == i]['Description Embedding'].to_numpy()   
        data[i].x = torch.Tensor(np.array([x for x in feat]))    
        index = nodes[nodes['Category'] == i]['Index'].to_numpy()
    
        data['disorder', 'DisorderIsSubtypeOfDisorder', 'disorder'].edge_index = torch.Tensor(new_edges_df.loc[new_edges_df['Value'] == 'DisorderIsSubtypeOfDisorder', ['Source', 'Target']].to_numpy().T).to(torch.int64)
        data['drug', 'DrugHasContraindication', 'disorder'].edge_index = torch.Tensor(new_edges_df.loc[new_edges_df['Value'] == 'DrugHasContraindication', ['Source', 'Target']].to_numpy().T).to(torch.int64)
        data['drug', 'DrugHasIndication', 'disorder'].edge_index = torch.Tensor(new_edges_df.loc[new_edges_df['Value'] == 'DrugHasIndication', ['Source', 'Target']].to_numpy().T).to(torch.int64)
        data['drug', 'DrugHasTarget', 'protein'].edge_index = torch.Tensor(new_edges_df.loc[new_edges_df['Value'] == 'DrugHasTarget', ['Source', 'Target']].to_numpy().T).to(torch.int64)
        data['gene', 'GeneAssociatedWithDisorder', 'disorder'].edge_index = torch.Tensor(new_edges_df.loc[new_edges_df['Value'] == 'GeneAssociatedWithDisorder', ['Source', 'Target']].to_numpy().T).to(torch.int64)
        data['protein', 'ProteinEncodedByGene', 'gene'].edge_index = torch.Tensor(new_edges_df.loc[new_edges_df['Value'] == 'ProteinEncodedByGene', ['Source', 'Target']].to_numpy().T).to(torch.int64)
        data['protein', 'ProteinHasSignature', 'signature'].edge_index = torch.Tensor(new_edges_df.loc[new_edges_df['Value'] == 'ProteinHasSignature', ['Source', 'Target']].to_numpy().T).to(torch.int64)
        data['protein', 'ProteinInPathway', 'pathway'].edge_index = torch.Tensor(new_edges_df.loc[new_edges_df['Value'] == 'ProteinInPathway', ['Source', 'Target']].to_numpy().T).to(torch.int64)
        data['disorder', 'DisorderHasPhenotype', 'phenotype'].edge_index = torch.Tensor(new_edges_df.loc[new_edges_df['Value'] == 'DisorderHasPhenotype', ['Source', 'Target']].to_numpy().T).to(torch.int64)
        data['drug', 'DrugHasSideEffect', 'side_effect'].edge_index = torch.Tensor(new_edges_df.loc[new_edges_df['Value'] == 'DrugHasSideEffect', ['Source', 'Target']].to_numpy().T).to(torch.int64)
        data['go', 'GOIsSubtypeOfGO', 'go'].edge_index = torch.Tensor(new_edges_df.loc[new_edges_df['Value'] == 'GOIsSubtypeOfGO', ['Source', 'Target']].to_numpy().T).to(torch.int64)
        data['gene', 'GeneExpressedInTissue', 'tissue'].edge_index = torch.Tensor(new_edges_df.loc[new_edges_df['Value'] == 'GeneExpressedInTissue', ['Source', 'Target']].to_numpy().T).to(torch.int64)
        data['protein', 'ProteinExpressedInTissue', 'tissue'].edge_index = torch.Tensor(new_edges_df.loc[new_edges_df['Value'] == 'ProteinExpressedInTissue', ['Source', 'Target']].to_numpy().T).to(torch.int64)
        data['protein', 'ProteinHasGOAnnotation', 'go'].edge_index = torch.Tensor(new_edges_df.loc[new_edges_df['Value'] == 'ProteinHasGOAnnotation', ['Source', 'Target']].to_numpy().T).to(torch.int64)
        data['protein', 'ProteinInteractsWithProtein', 'protein'].edge_index = torch.Tensor(new_edges_df.loc[new_edges_df['Value'] == 'ProteinInteractsWithProtein', ['Source', 'Target']].to_numpy().T).to(torch.int64)
        data['side_effect', 'SideEffectSameAsPhenotype', 'phenotype'].edge_index = torch.Tensor(new_edges_df.loc[new_edges_df['Value'] == 'SideEffectSameAsPhenotype', ['Source', 'Target']].to_numpy().T).to(torch.int64)
        data['protein', 'IsIsoformOf', 'protein'].edge_index = torch.Tensor(new_edges_df.loc[new_edges_df['Value'] == 'IsIoformOf', ['Source', 'Target']].to_numpy().T).to(torch.int64)
        data['drug', 'MoleculeSimilarityMolecule', 'drug'].edge_index = torch.Tensor(new_edges_df.loc[new_edges_df['Value'] == 'MoleculeSimilarityMolecule', ['Source', 'Target']].to_numpy().T).to(torch.int64)

    
    data = T.ToUndirected()(data)
    
    
    return data

def initial_candidates(model, nodes, data, diso):
    with torch.no_grad():
        df = pd.DataFrame(columns=['ID', 'Score'])
        drugs = nodes[nodes['Category'] == 'drug']
        id1 = nodes[nodes['Nodes Name'] == diso]['Index'].iloc[0]
        a = np.full((1, len(drugs)), id1).flatten()
        b = torch.arange(0, len(drugs), 1)
#         print(torch.Tensor(a))
#         print(b)
#         print(torch.stack((b, torch.Tensor(a))).long())

        data['drug', 'DrugHasIndication', 'disorder'].edge_label_index = torch.stack((b, torch.Tensor(a))).long()
        pred = torch.sigmoid(model(data))
        df['ID'] = drugs['Nodes Name']
        df['Score'] = pred.detach().numpy()
    return df.sort_values('Score', ascending = False)


def initial_candidates_drug(model, nodes, data, drug):
    with torch.no_grad():
        df = pd.DataFrame(columns=['ID', 'Score'])
        diso = nodes[nodes['Category'] == 'disorder']
        id1 = nodes[nodes['Nodes Name'] == drug]['Index'].iloc[0]
        a = np.full((1, len(diso)), id1).flatten()
        b = torch.arange(0, len(diso), 1)
#         print(torch.Tensor(a))
#         print(b)
#         print(torch.stack((b, torch.Tensor(a))).long())

        data['drug', 'DrugHasIndication', 'disorder'].edge_label_index = torch.stack((torch.Tensor(a), b)).long()
        pred = torch.sigmoid(model(data))
        df['ID'] = diso['Nodes Name']
        df['Score'] = pred.detach().numpy()
    return df.sort_values('Score', ascending = False)




def load_model(path): 
    
    
    G = load_graph()
    
    nodes = pd.read_pickle('nodes_v3.1.pkl')
    
    data = create_new_graph(nodes, G)
    
    print('Loading model...')
    
    model = Model(hidden_channels=128, data = data)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    
    print('Model loaded!')
    
    return model, nodes, data, G


    
    
def create_subgraph(G, diso, num_hops): 
    
    target_node = diso
    k = num_hops

    # Create the subgraph based on the k-hop neighborhood of the target node
    subgraph = nx.ego_graph(G, target_node, radius=k, undirected = True)
    
    return subgraph


def get_candidates(model, nodes, data, G, diso, k = 2, remove_present_links = True): 
    
    predictions = initial_candidates(model, nodes, data, diso)
    
    if k == -1: 
        subgraph = G
    else: 
        subgraph = create_subgraph(G, diso, k)
    
    new_candidates = []
    for i in subgraph.nodes(): 
        if i.startswith('drug'): 
            new_candidates.append(i)
                        
    new_preds = predictions[predictions['ID'].isin(new_candidates)].reset_index()
    
    if remove_present_links: 
        filtered_ids = [i for i in new_preds['ID'] if i not in nx.all_neighbors(G, diso)]
        filtered_preds = new_preds[new_preds['ID'].isin(filtered_ids)]
        return filtered_preds

    else: 
        return new_preds
    
    return filtered_preds


def get_candidates_drug(model, nodes, data, G, drug, k = 2, remove_present_links = True): 
    
    predictions = initial_candidates_drug(model, nodes, data, drug)
    
    if k == -1: 
        subgraph = G
    else: 
        subgraph = create_subgraph(G, drug, k)
    
    new_candidates = []
    for i in subgraph.nodes(): 
        if i.startswith('mondo'): 
            new_candidates.append(i)
                        
    new_preds = predictions[predictions['ID'].isin(new_candidates)].reset_index()
    
    if remove_present_links: 
        filtered_ids = [i for i in new_preds['ID'] if i not in nx.all_neighbors(G, drug)]
        filtered_preds = new_preds[new_preds['ID'].isin(filtered_ids)]
        return filtered_preds

    else: 
        return new_preds
    
    return filtered_preds    
    

def create_new_graph_2(nodes, subgraph): 
    
#     print('Creating new graph...')
    
    data = HeteroData()

    # Assuming 'subgraph' is defined somewhere above this snippet
    edges_dic = nx.get_edge_attributes(subgraph, 'edge_name')
    
    # Define your node categories
    node_names = ["disorder", "drug", "gene", "go", "pathway", "phenotype", "protein", "side_effect", "signature", "tissue"]
    nodes['Index'] = None  # Initialize the Index column to None
    
    # Create a dictionary to store category indices, ensuring each category starts with index 0
    category_indices = {n: nodes[nodes['Category'] == n].index for n in node_names}
    
    # Update 'Index' column in nodes DataFrame, resetting for each category
    for n, indices in category_indices.items():
        nodes.loc[indices, 'Index'] = range(len(indices))
    
    # Create a dictionary to map node names to their indices
    node_index_map = dict(zip(nodes['Nodes Name'], nodes['Index']))
    
    # print(nodes)
    
    # Create a dictionary to store new edges with updated indices
    new_edges = []
    for (node1, node2), val in edges_dic.items():
        if node1 in node_index_map and node2 in node_index_map:
            index1, index2 = node_index_map[node1], node_index_map[node2]
            new_edges.append((index1, index2, val))
        else:
            print(f"Nodes not found for edge: ({node1}, {node2})")

    new_edges_df = pd.DataFrame(new_edges, columns=['Source', 'Target', 'Value'])
            
    
    for i in node_names: 
        
        if i in set(nodes['Category']): 
            feat = nodes[nodes['Category'] == i]['Description Embedding'].to_numpy()
            data[i].x = torch.Tensor(np.array([x for x in feat]))
            index = nodes[nodes['Category'] == i]['Index'].to_numpy()
            data[i].node_id = np.array([x for x in index]) 
        
        else: 
            data[i].x = torch.Tensor(np.array([np.zeros(768)]))
            data[i].node_id = np.array([0]) 
        
    if 'DisorderIsSubtypeOfDisorder' in set(edges_dic.values()): 
        data['disorder', 'DisorderIsSubtypeOfDisorder', 'disorder'].edge_index = torch.Tensor(new_edges_df.loc[new_edges_df['Value'] == 'DisorderIsSubtypeOfDisorder', ['Source', 'Target']].to_numpy().T).to(torch.int64)
        
    else: 
        data['disorder', 'DisorderIsSubtypeOfDisorder', 'disorder'].edge_index = torch.Tensor([[], []]).long()
        
    if 'DrugHasContraindication' in set(edges_dic.values()): 
        data['drug', 'DrugHasContraindication', 'disorder'].edge_index = torch.Tensor(new_edges_df.loc[new_edges_df['Value'] == 'DrugHasContraindication', ['Source', 'Target']].to_numpy().T).to(torch.int64)
        
    else: 
        data['drug', 'DrugHasContraindication', 'disorder'].edge_index = torch.Tensor([[], []]).long()
        
    if 'DrugHasIndication' in set(edges_dic.values()): 
        data['drug', 'DrugHasIndication', 'disorder'].edge_index = torch.Tensor(new_edges_df.loc[new_edges_df['Value'] == 'DrugHasIndication', ['Source', 'Target']].to_numpy().T).to(torch.int64)
        
#         print(edges_dic)
#         print(new_edges)
#         print(torch.Tensor(np.array([k for k,v in new_edges.items() if v == 'DrugHasIndication']).transpose()).to(torch.int64))
#         print('Done')
        
    else: 
        data['drug', 'DrugHasIndication', 'disorder'].edge_index = torch.Tensor([[], []]).long()
    
    if 'DrugHasTarget' in set(edges_dic.values()): 
        data['drug', 'DrugHasTarget', 'protein'].edge_index = torch.Tensor(new_edges_df.loc[new_edges_df['Value'] == 'DrugHasTarget', ['Source', 'Target']].to_numpy().T).to(torch.int64)
     
    else: 
        data['drug', 'DrugHasTarget', 'protein'].edge_index = torch.Tensor([[], []]).long()
        
    if 'GeneAssociatedWithDisorder' in set(edges_dic.values()):   
        data['gene', 'GeneAssociatedWithDisorder', 'disorder'].edge_index = torch.Tensor(new_edges_df.loc[new_edges_df['Value'] == 'GeneAssociatedWithDisorder', ['Source', 'Target']].to_numpy().T).to(torch.int64)
        
    else: 
        data['gene', 'GeneAssociatedWithDisorder', 'disorder'].edge_index = torch.Tensor([[], []]).long()
        
    if 'ProteinEncodedByGene' in set(edges_dic.values()):   
        data['protein', 'ProteinEncodedByGene', 'gene'].edge_index = torch.Tensor(new_edges_df.loc[new_edges_df['Value'] == 'ProteinEncodedByGene', ['Source', 'Target']].to_numpy().T).to(torch.int64)
        
    else: 
        data['protein', 'ProteinEncodedByGene', 'gene'].edge_index = torch.Tensor([[], []]).long()
    
    if 'ProteinHasSignature' in set(edges_dic.values()):   
        data['protein', 'ProteinHasSignature', 'signature'].edge_index = torch.Tensor(new_edges_df.loc[new_edges_df['Value'] == 'ProteinHasSignature', ['Source', 'Target']].to_numpy().T).to(torch.int64)
        
    else: 
        data['protein', 'ProteinHasSignature', 'signature'].edge_index = torch.Tensor([[], []]).long()
        
    if 'ProteinInPathway' in set(edges_dic.values()):   
        data['protein', 'ProteinInPathway', 'pathway'].edge_index = torch.Tensor(new_edges_df.loc[new_edges_df['Value'] == 'ProteinInPathway', ['Source', 'Target']].to_numpy().T).to(torch.int64)
        
    else: 
        data['protein', 'ProteinInPathway', 'pathway'].edge_index = torch.Tensor([[], []]).long()
        
    # if 'molecule_similarity_molecule' in set(edges_dic.values()):   
    #     data['drug', 'molecule_similarity_molecule', 'drug'].edge_index = torch.Tensor(new_edges_df.loc[new_edges_df['Value'] == 'molecule_similarity_molecule', ['Source', 'Target']].to_numpy().T).to(torch.int64)
        
    # else: 
    #     data['drug', 'molecule_similarity_molecule', 'drug'].edge_index = torch.Tensor([[], []]).long()
        
    # if 'protein_similarity_protein' in set(edges_dic.values()):   
    #     data['protein', 'protein_similarity_protein', 'protein'].edge_index = torch.Tensor(new_edges_df.loc[new_edges_df['Value'] == 'protein_similarity_protein', ['Source', 'Target']].to_numpy().T).to(torch.int64)
        
    # else: 
    #     data['protein', 'protein_similarity_protein', 'protein'].edge_index = torch.Tensor([[], []]).long()
        
    # if 'is_isoform_of' in set(edges_dic.values()):   
    #     data['protein', 'is_isoform_of', 'protein'].edge_index = torch.Tensor(new_edges_df.loc[new_edges_df['Value'] == 'molecule_similarity_molecule', ['Source', 'Target']].to_numpy().T).to(torch.int64)
        
    # else: 
    #     data['protein', 'is_isoform_of', 'protein'].edge_index = torch.Tensor([[], []]).long()
        
    
    if 'ProteinInteractsWithProtein' in set(edges_dic.values()):   
        data['protein', 'ProteinInteractsWithProtein', 'protein'].edge_index = torch.Tensor(new_edges_df.loc[new_edges_df['Value'] == 'ProteinInteractsWithProtein', ['Source', 'Target']].to_numpy().T).to(torch.int64)
        
    else: 
        data['protein', 'ProteinInteractsWithProtein', 'protein'].edge_index = torch.Tensor([[], []]).long()


    if 'DisorderHasPhenotype' in set(edges_dic.values()):   
        data['disorder', 'DisorderHasPhenotype', 'phenotype'].edge_index = torch.Tensor(new_edges_df.loc[new_edges_df['Value'] == 'DisorderHasPhenotype', ['Source', 'Target']].to_numpy().T).to(torch.int64)
        
    else: 
        data['disorder', 'DisorderHasPhenotype', 'phenotype'].edge_index = torch.Tensor([[], []]).long()


    if 'DrugHasSideEffect' in set(edges_dic.values()):   
        data['drug', 'DrugHasSideEffect', 'side_effect'].edge_index = torch.Tensor(new_edges_df.loc[new_edges_df['Value'] == 'DrugHasSideEffect', ['Source', 'Target']].to_numpy().T).to(torch.int64)
        
    else: 
        data['drug', 'DrugHasSideEffect', 'side_effect'].edge_index = torch.Tensor([[], []]).long()


    if 'GOIsSubtypeOfGO' in set(edges_dic.values()):   
        data['go', 'GOIsSubtypeOfGO', 'go'].edge_index = torch.Tensor(new_edges_df.loc[new_edges_df['Value'] == 'GOIsSubtypeOfGO', ['Source', 'Target']].to_numpy().T).to(torch.int64)
        
    else: 
        data['go', 'GOIsSubtypeOfGO', 'go'].edge_index = torch.Tensor([[], []]).long()


    if 'GeneExpressedInTissue' in set(edges_dic.values()):   
        data['gene', 'GeneExpressedInTissue', 'tissue'].edge_index = torch.Tensor(new_edges_df.loc[new_edges_df['Value'] == 'GeneExpressedInTissue', ['Source', 'Target']].to_numpy().T).to(torch.int64)
        
    else: 
        data['gene', 'GeneExpressedInTissue', 'tissue'].edge_index = torch.Tensor([[], []]).long()

    if 'ProteinExpressedInTissue' in set(edges_dic.values()):   
        data['protein', 'ProteinExpressedInTissue', 'tissue'].edge_index = torch.Tensor(new_edges_df.loc[new_edges_df['Value'] == 'ProteinExpressedInTissue', ['Source', 'Target']].to_numpy().T).to(torch.int64)
        
    else: 
        data['protein', 'ProteinExpressedInTissue', 'tissue'].edge_index = torch.Tensor([[], []]).long()

    if 'ProteinHasGOAnnotation' in set(edges_dic.values()):   
        data['protein', 'ProteinHasGOAnnotation', 'go'].edge_index = torch.Tensor(new_edges_df.loc[new_edges_df['Value'] == 'ProteinHasGOAnnotation', ['Source', 'Target']].to_numpy().T).to(torch.int64)
        
    else: 
        data['protein', 'ProteinHasGOAnnotation', 'go'].edge_index = torch.Tensor([[], []]).long()

    if 'SideEffectSameAsPhenotype' in set(edges_dic.values()):   
        data['side_effect', 'SideEffectSameAsPhenotype', 'phenotype'].edge_index = torch.Tensor(new_edges_df.loc[new_edges_df['Value'] == 'SideEffectSameAsPhenotype', ['Source', 'Target']].to_numpy().T).to(torch.int64)
        
    else: 
        data['side_effect', 'SideEffectSameAsPhenotype', 'phenotype'].edge_index = torch.Tensor([[], []]).long()

    if 'IsIsoformOf' in set(edges_dic.values()):   
        data['protein', 'IsIsoformOf', 'protein'].edge_index = torch.Tensor(new_edges_df.loc[new_edges_df['Value'] == 'IsIsoformOf', ['Source', 'Target']].to_numpy().T).to(torch.int64)
        
    else: 
        data['protein', 'IsIsoformOf', 'protein'].edge_index = torch.Tensor([[], []]).long()

    if 'MoleculeSimilarityMolecule' in set(edges_dic.values()):   
        data['drug', 'MoleculeSimilarityMolecule', 'drug'].edge_index = torch.Tensor(new_edges_df.loc[new_edges_df['Value'] == 'MoleculeSimilarityMolecule', ['Source', 'Target']].to_numpy().T).to(torch.int64)
        
    else: 
        data['drug', 'MoleculeSimilarityMolecule', 'drug'].edge_index = torch.Tensor([[], []]).long()
    
#     print(data)
    data = T.ToUndirected()(data)
    
    
    return data, nodes
    
    
    
    
    
def initial_candidates_2(model, nodes, data, diso, drug):
    with torch.no_grad():
        drug = nodes[nodes['Nodes Name'] == drug]['Index'].iloc[0]
        diso = nodes[nodes['Nodes Name'] == diso]['Index'].iloc[0]
        
        data['drug', 'DrugHasIndication', 'disorder'].edge_label_index = torch.tensor([[drug], [diso]])
        
        pred = model(data)

    return pred
    
    

    
def obtain_path_list(G, nodes, drug, diso, k, only_directed = False): 
    
    edges_dic = nx.get_edge_attributes(G, 'edge_name')

    path_list = []

    if not only_directed: 

        G = G.to_undirected()

    for path in nx.all_simple_paths(G, source = drug, target= diso, cutoff = k): #Can also apply G.to_undirected(), takes more time but allows for undirected explanations
        
        G_sub = nx.DiGraph()
        for i in range(len(path) -1): 
            if (path[i], path[i+1]) in edges_dic:
                G_sub.add_edge(path[i], path[i+1], edge_name=edges_dic[(path[i], path[i+1])])
            else:
                G_sub.add_edge(path[i+1], path[i], edge_name=edges_dic[(path[i+1], path[i])])

        path_list.append(G_sub)
        
#    print(len(path_list))
    return path_list

# TODO: Paralelize function
# def score_single_path(j, i, nodes, model, drug, diso):
#     data, sub_nodes = create_new_graph_2(nodes[nodes['Nodes Name'].isin(i.nodes())].reset_index(), i)
#     pred = initial_candidates_2(model, sub_nodes, data, diso, drug)
#     score = pred
#     last_score = score[-1].item() if isinstance(score[-1], np.ndarray) else score[-1]
#     return {'ID': j, 'Score': last_score}

# def score_path(path_list, nodes, model, drug, diso):
#     num_processes = multiprocessing.cpu_count()  # Get the number of available CPU cores
#     pool = multiprocessing.Pool(processes=num_processes)
#     func = partial(score_single_path, nodes=nodes, model=model, drug=drug, diso=diso)

#     data_list = list(tqdm.tqdm(pool.imap_unordered(func, enumerate(path_list)), total=len(path_list), desc="Scoring paths"))
#     pool.close()
#     pool.join()

#     path_vals = pd.DataFrame(data_list)
#     return path_vals.sort_values(by='Score', ascending=False)


def score_path(path_list, nodes, model, drug, diso): 
    path_vals = pd.DataFrame(columns = [['ID'], ['Score']])
    data_list = [] 

    for j, i in tqdm.tqdm(enumerate(path_list), total=len(path_list), desc="Scoring paths"):
        data, sub_nodes = create_new_graph_2(nodes[nodes['Nodes Name'].isin(i.nodes())].reset_index(), i)
        pred = initial_candidates_2(model, sub_nodes, data, diso, drug)
#         score = -(np.log(pred) + np.log(1 - pred)) 
        score = pred # Compute the score TODO: try using the pred value itself
        last_score = score[-1].item() if isinstance(score[-1], np.ndarray) else score[-1]  # Extract last score value

        # Append 'j' and the last value of 'score' to the list
        data_list.append({'ID': j, 'Score': last_score})

    # Create DataFrame from the list of dictionaries
    path_vals = pd.DataFrame(data_list)
    
    
    return path_vals.sort_values(by='Score', ascending=False)


def best_explanations(G, nodes, model, drug, diso, k, ind = 0, only_directed = False): 
    
    path_list = obtain_path_list(G, nodes, drug, diso, k, only_directed = only_directed)
    
    if len(path_list) == 0: 
        print('No path of target length')
        return [] , pd.DataFrame(columns = [['ID'], ['Score']])
    
    if len(path_list) >= 1000: 
        print(f'The number of paths is {len(path_list)}. The estimated time is {len(path_list)/3068} mins. Would you like to reduce the length of the paths by 1? (Type Y/N)' )
        # answer = input() #Change this

        answer = 'N'
        
        while answer != 'Y' and answer != 'N': 
            print('Wrong answer, please type Y or N')
            answer = input()
        
        if answer == 'Y': 
            return best_explanations(G, nodes, model, drug, diso, k-1, ind = 0)
        
        else: 
            pass
        
            
    
    score_df = score_path(path_list, nodes, model, drug, diso)
    
    return path_list, score_df


# def plot_explanation(path_list, score_df, ind, fz = 6):
    
#     best_G = path_list[score_df[score_df['ID'] == int(ind)].index.tolist()[0]]
    
#     edges_dic = nx.get_edge_attributes(best_G, 'edge_name')
    
#     edge_labels = {}
    
#     for edge in best_G.edges():
#         if edge in edges_dic:
#             edge_labels[edge] = edges_dic[edge]

#     # Arrange nodes in a straight line
#     pos = nx.shell_layout(best_G)

#     # Draw the graph with edge labels
#     nx.draw(best_G, pos, with_labels=True, node_size=500, node_color='lightblue', font_weight='bold',  font_size=fz)
#     nx.draw_networkx_edge_labels(best_G, pos, edge_labels=edge_labels)
#     plt.show()
    

    
    
def plot_explanation(path_list, score_df, nodes_df, ind, fz=6):
    index_map = dict(zip(nodes_df['Nodes Name'], nodes_df['Display Name']))
    
    best_G = path_list[score_df[score_df['ID'] == int(ind)].index.tolist()[0]]

    edges_dic = nx.get_edge_attributes(best_G, 'edge_name')
    edge_labels = {edge: edges_dic[edge] for edge in best_G.edges() if edge in edges_dic}

    node_types = dict(zip(nodes_df['Nodes Name'], nodes_df['Category']))  # Adjusted for correct node names

    unique_types = list(set(node_types.values()))
    colors = cm.tab10.colors
    type_colors = {node_type: colors[i % len(colors)] for i, node_type in enumerate(unique_types)}

    node_colors = [type_colors.get(node_types.get(node, None), 'lightgray') for node in best_G.nodes()]
    
    pos = nx.shell_layout(best_G)
    renamed_labels = {node: index_map.get(node, node) for node in best_G.nodes()}

    nx.draw(best_G, pos, labels=renamed_labels, with_labels=True, node_size=500, node_color=node_colors, font_weight='bold', font_size=fz)
    nx.draw_networkx_edge_labels(best_G, pos, edge_labels=edge_labels)

    legend_patches = [mpatches.Patch(color=color, label=category) for category, color in type_colors.items()]
    plt.legend(handles=legend_patches, loc='upper left')
    plt.show()


































def fine_tune(model, G_list):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion  = ...
    
    for epoch in range(100): 
        for G_sub, y in G_list: 
            data = create_data_loader(G_sub)
            pred = model(G_sub)
            loss = criterion(pred, y)
            loss.backpropagate()
    
    
    
    
    
    

            

            
            
