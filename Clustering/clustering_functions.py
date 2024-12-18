'''
Functions used in the clustering algorithm
'''

import torch
import numpy as np
import os
import time

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import rdFMCS
from rdkit import DataStructs
morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)  


from rdkit import RDLogger 
RDLogger.DisableLog('rdApp.*')



def find_links(df, threshold):

    '''Generates a tensor to show the links between all elements in the similarity matrix'''

    '''Arguments:
                        df: pandas DataFrame containing cleaned input pandas DataFrame (Pandas DataFrame object)
                        threshold: required Tanimoto similarity for a link between two molecules to be established (float)'''
        
    '''Outputs:
                        large_sim_tensor: pytorch tensor of size [N, 2] containing N links where each element is two indices that have a Tanimoto similarity > the threshold (Pytorch float.32 tensor)'''

    sm_list = df['smiles'].tolist()
    large_link_tensor = torch.empty(0, 2)
    fp_list = []
    batch_size = 64
    fp_batch_size = 524288
    mol_batch_size = 16384
    t_ = time.time()

    print(f'len df: {len(df)}')
    print(f'len sm_list: {len(sm_list)}')

    num_batches = int(len(sm_list)//mol_batch_size) + 1

    for x in range(num_batches):

        start_idx = x * mol_batch_size
        stop_idx = (x+1) * mol_batch_size

        smiles_sub = sm_list[start_idx:stop_idx]
        mols = [Chem.MolFromSmiles(x) for x in smiles_sub]
        fps = morgan_gen.GetFingerprints(mols,numThreads=32)
        fp_list.extend(fps)

        print(f'FP generation batch: {x}/{num_batches} complete, time taken: {time.time()-t_}')
        t_ = time.time()

    print(f'len fp_list: {len(fp_list)}')

        


    if len(fp_list) < fp_batch_size:
        fp_num_batches = 1
    else:
        fp_num_batches = int(len(fp_list) / fp_batch_size)
        if fp_num_batches * fp_batch_size < len(fp_list):
            fp_num_batches +=1

    t_ = time.time()


    for fp_idx in range(fp_num_batches):
        fp_start_idx = int(fp_idx * fp_batch_size)
        fp_stop_idx = int((fp_idx+1) * fp_batch_size)

        sub_fps = fp_list[fp_start_idx:fp_stop_idx]
        fingerprint_arrays = np.array([np.array(fp) for fp in sub_fps])
        fingerprint_tensor = torch.tensor(fingerprint_arrays, dtype=torch.float32).to('cuda')
        link_list = []


        num_batches = int(len(sub_fps) / batch_size)
        if num_batches*batch_size < len(sub_fps):
            num_batches = num_batches +1


        for z in range(num_batches):
            start_idx = int(z * batch_size)
            stop_idx = int((z + 1) * batch_size)


            sub_link_tensor = calculate_links(start_idx, stop_idx, fp_start_idx, fingerprint_tensor, threshold)
            link_list.extend(sub_link_tensor)

            if (z % 10==0):
                print(f'Currently on similarity calcuation {batch_size*z + (fp_idx* fp_batch_size)}/{len(fp_list)}, time taken: {time.time()-t_}')
                t_=time.time()

    
        if len(link_list)>0:
            link_indices = torch.stack(link_list)
        else:
            link_indices = torch.tensor([])

        large_link_tensor = torch.cat((large_link_tensor,link_indices.to('cpu')), dim=0)


    return large_link_tensor

def calculate_links(start_idx, stop_idx, off_set, fingerprint_tensor, threshold):
    
    
    '''Computes chunks of the similarity matrix and then finds links between them.'''

    '''Arguments:
                        start_idx: Gives the starting index for which elements to look at in the fingerprint tensor
                        stop_idx: Gives the end index for which elements to look at in the fingerprint tensor
                        off_set: Tells us how much to offset the link tensor. Since we look at sub-rectangles in the similarity matrix, we need to off-set whatever links we find. E.g.,
                                 if we looked at the top right quartile of a matrix then any links found within this quartile would have to be offset to reflect their actual place in the full matrix.
                        fingerprint_tensor: pytorch tensor containing the fingerprints (Pytorch float.32 tensor)
                        threshold: required Tanimoto similarity for a link between two molecules to be established (float)'''
        
    '''Outputs:
                        sub_link_tensor: pytorch tensor contaiing the indices of where links occur (Pytorch float.32 tensor) '''


    if stop_idx > fingerprint_tensor.shape[0]:
        stop_idx = fingerprint_tensor.shape[0]

    target_fingerprints = fingerprint_tensor[start_idx:stop_idx]  
    dot_products = torch.mm(target_fingerprints, fingerprint_tensor.T)
    target_on_bits = target_fingerprints.sum(dim=1).unsqueeze(1)  
    all_on_bits = fingerprint_tensor.sum(dim=1).unsqueeze(0)
    tanimoto_similarity = dot_products / (target_on_bits + all_on_bits - dot_products)

    diag_tensor = torch.eye(stop_idx-start_idx)
    expanded_tensor = torch.zeros((stop_idx-start_idx), fingerprint_tensor.shape[0])
    expanded_tensor[:(stop_idx-start_idx), :(stop_idx-start_idx)] = diag_tensor
    mask = expanded_tensor==1
    tanimoto_similarity[mask] =0

    sub_link_tensor = torch.nonzero(tanimoto_similarity > threshold)
    sub_link_tensor[:, 0] +=start_idx
    sub_link_tensor += off_set

    
    return sub_link_tensor

def find_clusters(link_tensor):

    '''Single-linkage clustering search algorithm to generate clusters. Makes use of depth-first search in order to find all connected links (i.e., clusters)'''

    '''Arguments:
                        link_tensor: pytorch tensor of size [N, 2] containing N links where each element is two indices that have a Tanimoto similarity > the threshold (Pytorch float.32 tensor)'''
        
    '''Outputs:
                        clusters: nested list of indices corresponding to a cluster. E.g., element 1 of this list contains a list with the indices of all memebers in a cluster.
                                  In other words, this is a list of cluster indices. (list)'''


    graph = {}
    for i, j in link_tensor:
        i, j = int(i.item()), int(j.item())
        
        if i not in graph:
            graph[i] = []
        if j not in graph:
            graph[j] = []
        graph[i].append(j)
        graph[j].append(i)
    
    visited = set()
    clusters = []
    
    for node in graph:
        if node not in visited:
            cluster = iterative_dfs(node, graph, visited)
            clusters.append(cluster)
    
    return clusters

def iterative_dfs(start_node, graph, visited):

    '''Iterative depth-first search algorithm. Used to find all graphs present in our link tensor, i.e., find all clusters. Inspired by https://en.wikipedia.org/wiki/Depth-first_search'''

    '''Arguments:
                        start_node: index of the specific molecule we are looking at, i.e., the index we are looking for connections to (int)
                        graph: dict of all indices and all of their corresponding connected indices (dict)
                        visited: set of indices (nodes) which have already been visited'''
        
    '''Outputs:
                        clusters: list of indices which have been connected together as a cluster (list)'''


    stack = [start_node]
    cluster = []

    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            cluster.append(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    stack.append(neighbor)
    
    return cluster

def find_mcs(molecules):

    '''Function used to find the most common structure within molecules. This function tries to calculate both the 'simple' (most common structure
       based on the Murcko scaffolds of the input molecules) and 'complex' (most common structures among the input molecules)'''

    '''Arguments:
                        molecules: list of RDKit mol objects that survived the sanitisation (list)'''
        
    '''Outputs:
                        backbone: RDKit mol object representing the most common structure (RDKit Mol object)
                        complexity: integer to tell if the simple of complex backbone was chosen (int)
                        complex_similarity: mean Tanimoto similarity between the complex backbone and the input molecules (float)
                        simple_similarity: mean Tanimoto similarity between the simple backbone and the input molecules (float)'''


    mcs_result = rdFMCS.FindMCS(molecules, timeout=3)
    mcs_smiles = mcs_result.smartsString 
    
    mcs_mol = Chem.MolFromSmarts(mcs_smiles)
    scaffold_mcs = get_mean_murcko_scaffold(molecules)

    try:

        morgan_Mols = morgan_gen.GetFingerprints(molecules,numThreads=32)
        morgan_Norm = morgan_gen.GetFingerprint(Chem.MolFromSmiles(Chem.MolToSmiles(mcs_mol)))
        morgan_Scaff = morgan_gen.GetFingerprint(Chem.MolFromSmiles(Chem.MolToSmiles(scaffold_mcs)))

        sim_Norm = np.mean(np.array(DataStructs.BulkTanimotoSimilarity(morgan_Norm, morgan_Mols)))
        sim_Scaff = np.mean(np.array(DataStructs.BulkTanimotoSimilarity(morgan_Scaff, morgan_Mols)))
    
    except:
        return None, 0, 0, 0

    if sim_Scaff >= sim_Norm:
        backbone = scaffold_mcs
        complexity = 1
        complex_similarity = sim_Norm
        simple_similarity = sim_Scaff
    else:
        backbone = mcs_mol
        complexity = 0
        complex_similarity = sim_Norm
        simple_similarity = sim_Scaff

    return backbone, complexity, complex_similarity, simple_similarity
    


def get_mean_murcko_scaffold(mols):

    '''Generates the 'simple' backbone, or rather, the most common structure among the Murcko scaffolds of the input molecules'''

    '''Arguments:
                        mols: list of RDKit mol objects that survived the sanitisation (list)'''
        
    '''Outputs:
                        scaffold_MCS: RDKit mol object that contains the most common structure derived from the Murcko scaffolds'''


    mols = [x for x in mols if x != None]
    scaffold = [MurckoScaffold.GetScaffoldForMol(x) for x in mols]
    scaffold_MCS = rdFMCS.FindMCS(scaffold, threshold=0.66, timeout=3)
    mcs_smiles = scaffold_MCS.smartsString 
    scaffold_MCS = Chem.MolFromSmarts(mcs_smiles)

    return scaffold_MCS


def cluster_row_gen(backbone_constituents):

    '''Function to generate the 'backbone_df' pandas dataframe. This dataframe combines molecular properties of molecules within a cluster.'''

    '''Arguments:
                        backbone_constituents: large pandas DataFrame object that contains all of the constituent molecules and their respective cluster backbone (Pandas DataFrame object)'''
    
    '''Outputs:
                        backbone_df: pandas DataFrame of all of the most common structures (backbones) within clusters (Pandas DataFrame object)'''


    backbone_df = (
        backbone_constituents.groupby('smiles_backbone')
        .agg(
            count=('smiles_backbone', 'size'),
            TransitionEnergies1_MU=('TransitionEnergies1', 'mean'),
            TransitionEnergies1_STD=('TransitionEnergies1', 'std'),  # Standard deviation
            TransitionEnergies2_MU=('TransitionEnergies2', 'mean'),
            TransitionEnergies2_STD=('TransitionEnergies2', 'std'),  # Standard deviation
            OscillatorStrength1_MU=('OscillatorStrength1', 'mean'),
            OscillatorStrength1_STD=('OscillatorStrength1', 'std'),  # Standard deviation
            OscillatorStrength2_MU=('OscillatorStrength2', 'mean'),
            OscillatorStrength2_STD=('OscillatorStrength2', 'std'),  # Standard deviation
            SAS_MU=('SAS', 'mean'),
            SAS_STD=('SAS', 'std'),
            backbone_size_MU =('backbone_size', 'mean'),
            backbone_size_STD = ('backbone_size', 'std'),
            simple=('simple', 'mean'),
            generic_similarity=('generic_similarity', 'mean'),
            simple_similarity=('simple_similarity', 'mean')
            
        )
        .reset_index()
    )

    return backbone_df


def save_backbones(settings, backbone_df):


    '''Saves the dataframe generated by cluster_row_gen function'''

    '''Arguments:
                        settings: settings defined by the corresponding .yml file (dict)
                        backbone_df: pandas DataFrame of all of the most common structures (backbones) within clusters (Pandas DataFrame object)'''

    output_folder = settings['settings']['output_file']
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_path = output_folder + 'Backbones.csv'

    backbone_df.to_csv(output_path, index=False)




def save_constituents(settings, backbone_constituents):

    '''Saves the dataframe generated by cluster_row_gen function'''

    '''Arguments:
                        settings: settings defined by the corresponding .yml file (dict)
                        backbone_constituents: large pandas DataFrame object that contains all of the constituent molecules and their respective cluster backbone (Pandas DataFrame object)'''


    output_folder = settings['settings']['output_file']
    output_path = output_folder + 'Backbone_constituents.csv'
    backbone_constituents.to_csv(output_path, index=False)