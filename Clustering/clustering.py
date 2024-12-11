#!/usr/bin/python3

import yaml
import pandas as pd
from rdkit import Chem
import selfies
import time

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from clustering_functions import find_links, find_clusters, find_mcs, cluster_row_gen, save_backbones, save_constituents

from rdkit import RDLogger 
RDLogger.DisableLog('rdApp.*')


def data_init(interested_path):

    '''Intialisation and cleaning up of input pandas dataframe'''

    '''Arguments:
                        interested_path: string containing the path to the input dataframe (str)'''
        
    '''Outputs:
                        df_new: cleaned version of the input pandas DataFrame (pandas DataFrame object)'''


    df = pd.read_csv(interested_path)
    df['smiles'] = df['canon_smiles']
    smiles_list = df['canon_smiles'].tolist()


    new_constraints = selfies.get_semantic_constraints()
    new_constraints['N'] = 5
    new_constraints['B'] = 4
    selfies.set_semantic_constraints(new_constraints)  # update constraints

    mols = [Chem.MolFromSmiles(x) for x in smiles_list]
    valid_list = [x for x in range(len(mols)) if mols[x] !=None]
    smiles_list = [smiles_list[x] for x in valid_list]
    molecules = [mols[x] for x in valid_list]
    df_proto = df.loc[valid_list].reset_index(drop=True)

    scaffold = [MurckoScaffold.GetScaffoldForMol(x) for x in molecules]
    valid_list2 = []
    for x in range(len(scaffold)):
        mol = scaffold[x]
        try:
            Chem.MolToSmiles(mol)
            valid_list2.append(x)
        except Exception as e:
            pass


    molecules = [mols[x] for x in valid_list2]
    df_proto2 = df_proto.loc[valid_list2].reset_index(drop=True)

    valid_list3 = [x for x in range(len(molecules)) if molecules[x] is not None]
    df_new = df_proto2.loc[valid_list3].reset_index(drop=True)

    return df_new
    

def cluster_df_list_generation(df, threshold):

    '''Takes input pandas dataframes and divides them up into clusters to generate a list of clustered pandas DataFrames'''

    '''Arguments:
                        interested_path: cleaned version of the input pandas DataFrame (pandas DataFrame object)
                        threshold: required Tanimoto similarity for a link between two molecules to be established (float)'''
        
    '''Outputs:
                        large_cluster_list: list of pandas DataFrames. Each pandas DataFrame contains molecules which have been clustered together (list)'''


    large_cluster_list = []        
    link_tensor = find_links(df, threshold)
    all_clusters = find_clusters(link_tensor)
    large_clusters = [cluster for cluster in all_clusters if len(cluster) > 2]

    for x in large_clusters:
        large_cluster_list.append(df.iloc[x].reset_index(drop=True))

    

    return large_cluster_list
    

def find_backbone(large_cluster_list):

    '''Searches for backbones amongst all of the clusters'''

    '''Arguments:
                        large_cluster_list: list of pandas DataFrames. Each pandas DataFrame contains molecules which have been clustered together (list)'''
        
    '''Outputs:
                        large_cluster_list_valid: list of pandas DataFrames. Same as large_cluster_list but dataframes have their corresponding backbones contained
                                                  and other information (list)'''


    counter = 0
    smiles_clusters = [x['canon_smiles'].tolist() for x in large_cluster_list]
    smiles_backbone = []
    large_cluster_list_valid = []
    t_ = time.time()

    for smiles_cluster in smiles_clusters:

        if counter % 25==0:
            print(f'Current cluster: {counter}/{len(smiles_clusters)}, time taken: {time.time()-t_}')
            t_ = time.time()

        molecules = [Chem.MolFromSmiles(sm) for sm in smiles_cluster if Chem.MolFromSmiles(sm) is not None]
        atom_count = [x.GetNumAtoms() for x in molecules]

        mcs_result, simple, sim_Norm, sim_Scaff = find_mcs(molecules)
        if mcs_result != None:
            mcs_smiles_full = Chem.MolToSmiles(mcs_result)
            smiles_backbone.append(mcs_smiles_full)

            backbone_atoms = mcs_result.GetNumAtoms()
            atom_proportions = [backbone_atoms / x if backbone_atoms != 0 else 0 for x in atom_count]

            large_cluster_list[counter]['smiles_backbone'] = mcs_smiles_full
            large_cluster_list[counter]['backbone_size'] = atom_proportions
            large_cluster_list[counter]['simple'] = simple
            large_cluster_list[counter]['generic_similarity'] = sim_Norm
            large_cluster_list[counter]['simple_similarity'] = sim_Scaff
            large_cluster_list_valid.append(large_cluster_list[counter])
        counter +=1

    return large_cluster_list_valid

def df_generation(large_cluster_list):

    '''Generates pandas DataFrames of the backbones and the molecules which constitute those backbones'''

    '''Arguments:
                        large_cluster_list: list of pandas DataFrames. Each pandas DataFrame contains molecules which have been clustered together (list)'''        
    '''Outputs:
                        backbone_df: pandas DataFrame of all of the most common structures (backbones) within clusters (Pandas DataFrame object)
                        backbone_constituents: large pandas DataFrame object that contains all of the constituent molecules and their respective cluster backbone (Pandas DataFrame object)'''


    large_cluster_list = [x[x['smiles_backbone'] != ""] for x in large_cluster_list]
    large_cluster_list = [x.drop_duplicates('canon_smiles') for x in large_cluster_list]
    large_cluster_list = [x for x in large_cluster_list if len(x) > 2]

    backbone_constituents = pd.concat(large_cluster_list)
    backbone_df = cluster_row_gen(backbone_constituents)

    return backbone_df, backbone_constituents



def main():


    settings = yaml.safe_load(open("clustering_settings.yml", "r"))
    interested_path = settings['settings']['input_file']
    threshold = settings['settings']['threshold']
    

    print('Starting')
    t_ = time.time()

    df = data_init(interested_path)
    print(f'data_init complete, time taken: {time.time() - t_}')
    t_ = time.time()

    large_cluster_list = cluster_df_list_generation(df, threshold)
    print(f'cluster_df_list_generation complete, time taken: {time.time() - t_}, number of clusters found: {len(large_cluster_list)}')
    t_ = time.time()

    large_cluster_list = find_backbone(large_cluster_list)    
    print(f'MCS_generation complete, time taken: {time.time() - t_}')
    t_ = time.time()

    backbone_df, backbone_constituents = df_generation(large_cluster_list)
    print(f'generate_backbone_df complete, time taken: {time.time() - t_}')

    save_backbones(settings, backbone_df)
    save_constituents(settings, backbone_constituents)



if __name__ == "__main__":
    main()
    

