'''
Functions to be used in the pipeline
'''

import pandas as pd
import torch
import selfies as sf
import numpy as np

import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from functions_sub import remove_unrecognized_symbols, selfies_to_one_hot



def get_selfie_and_smiles_encodings_for_dataset(file_path):

    '''Returns encoding, alphabet and length of largest molecule in SMILES and SELFIES, given a file containing SMILES molecules.'''

    '''Arguments:
                    file_path: .csv file with molecules. Column name containing the smiles must be 'smiles' (str)'''
    
    '''Outputs:
                    selfies_list: the SELFIES encoding of the SMILES molecules provided (list)
                    selfies_alphabet: the alphabet of the SELFIES encoding (list)
                    largest_selfies_len: the longest SELFIES encoding length (int)
                    smiles_list: a list of the SMILES encodings (list)
                    smiles_alphabet: the alphabet of the SMILES encoding (list)
                    largest_smiles_len: the longest SMILES encoding length (int)'''
                    


    df = pd.read_csv(file_path)
    df = df.dropna()

    new_constraints = sf.get_semantic_constraints()
    new_constraints['N'] = 5
    new_constraints['B'] = 4
    sf.set_semantic_constraints(new_constraints)  # update constraints


    smiles_list = np.asanyarray(df.smiles)
    smiles_list = remove_unrecognized_symbols(smiles_list)

    print('--> Translating SMILES to SELFIES...')
    selfies_list = list(map(sf.encoder, smiles_list))
    all_selfies_symbols = sf.get_alphabet_from_selfies(selfies_list)
    selfies_alphabet = list(all_selfies_symbols)
    selfies_alphabet.insert(0, '[nop]')

    largest_selfies_len = max(sf.len_selfies(s) for s in selfies_list)
    print('Finished translating SMILES to SELFIES.')

    return selfies_list, selfies_alphabet, largest_selfies_len



def selfies_to_lpoints(encoding_list, encoding_alphabet, largest_molecule_len, vae_encoder):

    '''Converts SELFIES to Mus for a given list and encoder'''

    '''Arguments:
                    encoding_list: the SELFIES encoding of the SMILES molecules provided (list)
                    encoding_alphabet: the alphabet of the SELFIES encoding (list)
                    largest_smiles_len: the longest SMILES encoding length (int)
                    vae_encoder: the encoder object (VAEEncoder object)'''
    
    '''Outputs:
                    mus: the mean latent vector (Pytorch float.32 tensor)'''

    vae_encoder.eval()

    mu_list = []

   

    batch_size = 256
    num_batches_train = (len(encoding_list) + batch_size - 1) // batch_size


    for batch_iteration in range(num_batches_train):

        start_idx = batch_iteration * batch_size
        stop_idx = (batch_iteration + 1) * batch_size

        sub_encoding = encoding_list[start_idx: stop_idx]
        data = selfies_to_one_hot(sub_encoding, encoding_alphabet, largest_molecule_len).to('cpu').to(torch.float32)

        inp_flat_one_hot = data.flatten(start_dim=1)
        inp_flat_one_hot = inp_flat_one_hot.squeeze().to(device)
        inp_flat_one_hot = inp_flat_one_hot.unsqueeze(0)
        if inp_flat_one_hot.dim() < 3:
            inp_flat_one_hot = inp_flat_one_hot.unsqueeze(0)


        _, mus_sub, _ = vae_encoder(inp_flat_one_hot)
        mu_list.append(mus_sub.to('cpu'))

    mus = torch.cat(mu_list)

    return mus




    