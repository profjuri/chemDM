'''
Less important/used functions used in the pipeline. Much more messy/redundant but used for convenience
'''

import os
import pandas as pd
import torch
import selfies as sf
import numpy as np
import time

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import RDConfig
from rdkit.Chem import AllChem
import deepchem as dc
fpgen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=4096)


import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def remove_unrecognized_symbols(smiles_list):

    '''Removes blank spaces from the SMILES encodings'''

    '''Arguments:
                    smiles_list: the list of SMILES encodings (list)'''
    
    '''Outputs:
                    cleaned_smiles: the cleaned SMILES encodings list (list)'''

    cleaned_smiles = [smiles.replace('\n', '') for smiles in smiles_list]

    return cleaned_smiles


def selfies_to_all(encoding_list, encoding_alphabet, largest_molecule_len, vae_encoder):

    '''Converts SELFIES to Mus, Zs and log_vars for a given list and encoder'''

    '''Arguments:
                    encoding_list: the SELFIES encoding of the SMILES molecules provided (list)
                    encoding_alphabet: the alphabet of the SELFIES encoding (list)
                    largest_smiles_len: the longest SMILES encoding length (int)
                    vae_encoder: the encoder object (VAEEncoder object)'''
    
    '''Outputs:     
                    zs: a latent vector modified by some distribution defined by the standard deviation (Pytorch float.32 tensor)
                    mus: the mean latent vectors (Pytorch float.32 tensor)
                    log_vars: the natural logarithm of the variance tensor of each latent vector (Pytorch float.32 tensor)'''



    vae_encoder.eval()
    z_list = []
    mu_list = []
    log_var_list = []
    batch_size = 1024 


    data = selfies_to_one_hot(encoding_list, encoding_alphabet, largest_molecule_len)
    inp_flat_one_hot = data.flatten(start_dim=1)
    inp_flat_one_hot = inp_flat_one_hot.unsqueeze(0)

    num_batches_train = (len(encoding_list) + batch_size - 1) // batch_size

    for batch_iteration in range(num_batches_train):

        start_idx = batch_iteration * batch_size
        stop_idx = (batch_iteration + 1) * batch_size
        sub_flat_ = inp_flat_one_hot.squeeze()
        if sub_flat_.dim() < 2:
            sub_flat_ = sub_flat_.unsqueeze(0)
        sub_flat = sub_flat_[start_idx: stop_idx].unsqueeze(0).to(device)

        zs_sub, mus_sub, log_vars_sub = vae_encoder(sub_flat)

        mu_list.append(mus_sub.to('cpu'))
        z_list.append(zs_sub.to('cpu'))
        log_var_list.append(log_vars_sub.to('cpu'))

    mus = torch.cat(mu_list)
    zs = torch.cat(z_list)
    log_vars = torch.cat(log_var_list)
    
    vae_encoder.train()

    return zs, mus, log_vars


def stats(y_test, y_pred):

    '''Statistics function that gives you the mse, mae and r^2'''

    '''Arguments:
                    y_test: the true value of whatever property you're analysing (Pytorch float tensor)
                    y_pred: the prediction value of whatever property you're analysing (Pytorch float tensor)'''
    
    '''Outputs:
                    MSE: mean squared error (float)
                    MAE: mean absolute error (float)
                    r2: the r squared coefficient (float)'''

    ABS_DIF = torch.abs(y_pred - y_test)
    MAE = torch.mean(ABS_DIF)
    MSE = torch.mean((y_pred - y_test)*(y_pred - y_test))

    
    MRE = torch.mean(ABS_DIF/y_test)

    SSR = torch.sum((y_test-y_pred).pow(2))
    SST = torch.sum((y_test-y_test.mean()).pow(2))
    r2 = 1 - SSR/SST

    return MSE, MAE, MRE, r2

def selfies_to_one_hot(encoding_list, encoding_alphabet, largest_molecule_len):

    '''One hot generation'''

    '''Arguments:
                    encoding_list: a list containing the SELFIES (list)
                    encoding_alphabet: the alphabet generated by the function get_selfie_and_smiles_encodings_for_dataset (list)
                    largest_molecule_len: the maximum length of the molecule encodings (int)'''
    
    '''Outputs:
                    one_hot: pytorch tensor of representing the SELFIES contained within the encoding_list (Pytorch float.32 tensor) '''

    alphabet_dict = {letter: index for index, letter in enumerate(encoding_alphabet)}
    integer_encoded = [[alphabet_dict[symbol] for symbol in sf.split_selfies(encoding_list[x])] for x in range(len(encoding_list))]
    max_length = max(len(inner_list) for inner_list in integer_encoded)
    padded_list = [torch.tensor(inner_list + [0] * (max_length - len(inner_list))) for inner_list in integer_encoded]
    padded_tensor = pad_sequence(padded_list, batch_first=True, padding_value=0)

    if padded_tensor.shape[1] < largest_molecule_len:
        extra_padding = torch.zeros(padded_tensor.shape[0],(largest_molecule_len), dtype = torch.int64)
        extra_padding[:, :padded_tensor.shape[1]] = padded_tensor
        padded_tensor = extra_padding

    
    one_hot = torch.nn.functional.one_hot(padded_tensor, num_classes = len(encoding_alphabet)).to(torch.float32)


    return one_hot


def _make_dir(directory):

    '''Makes the directory'''

    '''Arguments:
                    directory: directory path (str)'''
    os.makedirs(directory)


def get_free_memory(device):

    '''Calculate the amount of free memory'''

    '''Arguments:
                    device: the device being used (Pytorch object)'''
    
    '''Outputs:
                    free_mem: the amount of free memory in bytes (float)'''

    total_memory = torch.cuda.get_device_properties(device).total_memory
    reserved_memory = torch.cuda.memory_reserved(device)
    free_mem = total_memory - reserved_memory

    return free_mem

def get_free_ram():

    '''Finds the amount of free ram on the device. Pytorch refers to this as the 'cpu' device '''

    '''Outputs:
                    free_mem: the amount of free ram in bytes (float)'''


    import psutil

    memory_info = psutil.virtual_memory()
    free_mem = memory_info.available
           
    return free_mem

def lpoints_to_sequences(zs, vae_decoder):

    '''Converts latent vectors to sequence tensor. Useful for if you do not care about getting the SMILES/SELFIES output. This function
        removes non-unique latent vectors and so is only used in the brute force code.'''

    '''Arguments:
                    zs: a latent vector modified by some distribution defined by the standard deviation (Pytorch float.32 tensor)
                    vae_decoder: the decoder object (VAEDecoder object)'''
    
    '''Outputs:
                    sequence_tensor: pytorch sequence tensor representing the SELFIES output from the decoder. Sequence numbers correspond to alphabet characters (Pytorch float.32 tensor)'''


    if zs.dim() <2:
        zs = zs.unsqueeze(0)
    zs = zs.to('cpu')
        
    batch_size = 4096
    num_batches_train = int(len(zs) / batch_size)
    if num_batches_train < len(zs)/batch_size:
        num_batches_train = num_batches_train + 1
    seq_list = []
        

    for batch_iteration in range(num_batches_train):
        start_idx = batch_iteration * batch_size
        stop_idx = (batch_iteration + 1) * batch_size
        batch = zs[start_idx: stop_idx].to(device)

        with torch.no_grad():
            out_one_hot = vae_decoder(batch)
        seq_tensor = out_one_hot.argmax(2)        
        seq_list.append(seq_tensor)
    sequence_tensor = torch.cat(seq_list).to(device)
    sequence_tensor = torch.unique(sequence_tensor,dim=0)


    return sequence_tensor


def sequences_to_lpoints(sequence_tensor, encoding_alphabet, vae_encoder):

    '''Converts latent vectors to onehots. Useful for if you do not care about getting the SMILES/SELFIES output. This function
        removes non-unique one-hots and so is only used in the brute force code.'''

    '''Arguments:
                    sequence_tensor: pytorch sequence tensor representing the SELFIES output from the decoder. Sequence numbers correspond to alphabet characters (Pytorch float.32 tensor)
                    encoding_alphabet: the alphabet of the SELFIES encoding (list)
                    vae_encoder: the encoder object (VAEEncoder object)'''
    
    '''Outputs:
                    mus: the mean latent vectors (Pytorch float.32 tensor)'''

    mu_list = []
    batch_size = 1024


    num_batches_train = int(sequence_tensor.shape[0] / batch_size)
    if num_batches_train < sequence_tensor.shape[0] / batch_size:
        num_batches_train = num_batches_train + 1

    vae_encoder.eval()

    for batch_iteration in range(num_batches_train):

        start_idx = batch_iteration * batch_size
        stop_idx = (batch_iteration + 1) * batch_size
        batch = sequence_tensor[start_idx:stop_idx]

        data = torch.nn.functional.one_hot(batch, num_classes = len(encoding_alphabet))
        inp_flat_one_hot = data.flatten(start_dim=1)
        inp_flat_one_hot = inp_flat_one_hot.squeeze().to(torch.float32)


        if inp_flat_one_hot.dim() < 2:
            inp_flat_one_hot = inp_flat_one_hot.unsqueeze(0)
        inp_flat_one_hot = inp_flat_one_hot.unsqueeze(0).to(device)

        _, mus_sub, _ = vae_encoder(inp_flat_one_hot)
        mu_list.append(mus_sub)

    

    mus = torch.cat(mu_list)

    return mus
        

def try_encode(x):

    '''Function to see if a given SMILES can be encoded into a SELFIES'''

    '''Arguments:
                    x: a SMILES encoding (str)'''    
    
    '''Outputs:
                    sf.encoder(x): SELFIES version of the SMILES encoding (str) '''


    try:
        return sf.encoder(x)
    except Exception:
        return None
    

def get_run_log_list(settings):

    '''Function to get the run log list. Used so that the brute force code can be easily stopped and started'''

    '''Arguments:
                    settings: settings defined by the corresponding .yml file (dict)'''   
     
    '''Outputs:
                    run_log_list: list corresponding to the indices that have already been covered by the brute force code (list) '''


    if os.path.exists(settings['data']['save_path'] + '/index_list.txt'):
        log_txt = pd.read_csv(settings['data']['save_path'] + '/index_list.txt')
        run_log_list = log_txt['log']
        run_log_list = run_log_list.astype(int)
        run_log_list = run_log_list.tolist() #.astype(int).tolist()
        
    else:
        run_log_list = [0]


    return run_log_list

def get_run_log_list_fp(settings):

    '''Function to get the fingerprint run log list. Used so that the brute force code can be easily stopped and started.
        This is specifically used for fingerprint generation'''

    '''Arguments:
                    settings: settings defined by the corresponding .yml file (dict)'''
        
    '''Outputs:
                    run_log_list: list corresponding to the indices that have already been covered by the brute force code (list) '''
    

    if os.path.exists(settings['data']['save_path'] + '/fp_index_list.txt'):
        log_txt = pd.read_csv(settings['data']['save_path'] + '/fp_index_list.txt')
        run_log_list = log_txt['log']
        run_log_list = run_log_list.astype(int)
        run_log_list = run_log_list.tolist() #.astype(int).tolist()
        run_log_list.append(max(run_log_list)+1)
        
    else:
        run_log_list = [0]


    return run_log_list

def save_index(index, settings):

    '''Function to save the index of a completed loop in the brute force code'''

    '''Arguments:
                    settings: settings defined by the corresponding .yml file (dict)'''    


    save_path = settings['data']['save_path']
    log_filename = 'index_list.txt'

    log_filepath = os.path.join(save_path, log_filename)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    file_exists = os.path.isfile(log_filepath)

    with open(log_filepath, 'a') as file:
        if not file_exists:
            file.write("log\n")
        file.write(f'{index}\n')


def unique_values_and_first_indices(LIS):

    '''Finds the unique values of a list and the first index in which they appeared'''

    '''Arguments:
                    LIS: list of objects that we want to find the unique values + first indices for (list)'''  
      
    '''Outputs:
                    unique_values: unique version of the input list, no degeneracy (list)
                    first_indices: list of indices where the elements of the unique values first appeared (list) '''
    

    unique_values = []
    first_indices = []
    seen = {}

    for i, num in enumerate(LIS):
        if num not in seen:
            seen[num] = i
            unique_values.append(num)
            first_indices.append(i)
    
    return unique_values, first_indices

def non_none_values_and_indices(LIS):

    '''Finds the values of a list which are not none and the first index in which they appeared'''

    '''Arguments:
                    LIS: list of objects that we want to find the not none values + first indices (list)'''  
      
    '''Outputs:
                    non_none_values: version of the list that does not contain any none values (list)
                    non_none_indices: list of indices where the elements of the not none values appeared (list) '''
    

    non_none_values = []
    non_none_indices = []

    for i, value in enumerate(LIS):
        if value is not None:
            non_none_values.append(value)
            non_none_indices.append(i)
    
    return non_none_values, non_none_indices


def generate_normal_tensor(z_num, lpoint_size, bottom, top):

    ''' Generates a tensor which has a normal distribution with elements below bottom and above top cut out. Centred on 0 and has as standard deviation of 1'''

    '''Arguments:
                    z_num: number of normal distributions you desire, i.e., shape[0] of the normal tensor (int)
                    lpoint_size: size of the normal tensor (shape[1]), should correspond to the size of the latent vecotrs you want to apply this to (int)
                    bottom: minimum value you'd like to have in your distribution. E.g., if I want bottom=1, the elements between [-1, 1] will be cut out. (float)
                    top: maximum value you'd like to have in your distribution. E.g., if I want top=5, the elements below/above [-5, 5] will be cut out (float)'''  
      
    '''Outputs:
                    eps: the final normal distributed tensor (Pytorch float.32 tensor)'''
    

    eps = torch.rand(z_num, lpoint_size)
    eps = eps * (top - bottom)
    eps = eps + bottom
    mask = (torch.rand_like(eps) > 0.5).detach()
    eps[mask] = -eps[mask].detach()
    
    return eps


def sequences_to_smiles(scattered_seq_tensor, selfies_alphabet):
    selfies_alphabet_np = np.array(selfies_alphabet)
    continuous_strings = selfies_alphabet_np[scattered_seq_tensor]
    selfies_list = [''.join(row) for row in continuous_strings]

    smiles_list = [sf.decoder(x) for x in selfies_list]

    return selfies_list, smiles_list