'''
Functions to be used in the pipeline
'''

import os
import pandas as pd
import torch
import selfies as sf
import numpy as np
import time
import functions_sub

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

    data = selfies_to_one_hot(encoding_list, encoding_alphabet, largest_molecule_len).to('cpu').to(torch.float32)
    mus = torch.empty(0, dtype=torch.float32).to('cpu')

    batch_size = 4096

    inp_flat_one_hot = data.flatten(start_dim=1)
    inp_flat_one_hot = inp_flat_one_hot.unsqueeze(0)


    num_batches_train = int(data.shape[0] / batch_size)
    if num_batches_train < data.shape[0] / batch_size:
        num_batches_train = num_batches_train + 1


    for batch_iteration in range(num_batches_train):

        start_idx = batch_iteration * batch_size
        stop_idx = (batch_iteration + 1) * batch_size
        sub_flat_ = inp_flat_one_hot.squeeze()
        if sub_flat_.dim() < 2:
            sub_flat_ = sub_flat_.unsqueeze(0)
        sub_flat = sub_flat_[start_idx: stop_idx].unsqueeze(0).to(device)

        _, mus_sub, _ = vae_encoder(sub_flat)
        mu_list.append(mus_sub.to('cpu'))

    mus = torch.cat(mu_list)

    return mus



def decode_lpoints(zs, selfies_alphabet, vae_decoder):

    '''Converts latent vectors to SMILES'''

    '''Arguments:
                    zs: latent vectors, could be either Mu or Z. (Pytorch float.32 object)
                    selfies_alphabet: the alphabet of the SELFIES encoding (list)
                    vae_decoder: the decoder object (VAEDecoder object)'''
    
    '''Outputs:
                    final_smiles_list: the SMILES encoding of the input latent vectors'''
    

    selfies_alphabet_np = np.array(selfies_alphabet)
    final_smiles_list = []
    if zs.dim() <2:
        zs = zs.unsqueeze(0)

    batch_size = 4096
    num_batches_train = int(len(zs) / batch_size)

    if num_batches_train < len(zs)/batch_size:
        num_batches_train = num_batches_train + 1



    for batch_iteration in range(num_batches_train):

        start_idx = batch_iteration * batch_size
        stop_idx = (batch_iteration + 1) * batch_size
        batch = zs[start_idx: stop_idx].to(device)

        out_one_hot = vae_decoder(batch)
        seq_tensor = out_one_hot.argmax(2).cpu().numpy()

        continuous_strings = selfies_alphabet_np[seq_tensor]
        continuous_strings = [''.join(row) for row in continuous_strings]
        list_of_continuous_strings = [sf.decoder(s) for s in continuous_strings]
        final_smiles_list.extend(list_of_continuous_strings)


    return final_smiles_list
    
