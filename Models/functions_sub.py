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

def selfies_to_zs(encoding_list, encoding_alphabet, largest_molecule_len, vae_encoder):

    '''Converts SELFIES to Zs for a given list and encoder'''

    '''Arguments:
                    encoding_list: the SELFIES encoding of the SMILES molecules provided (list)
                    encoding_alphabet: the alphabet of the SELFIES encoding (list)
                    largest_smiles_len: the longest SMILES encoding length (int)
                    vae_encoder: the encoder object (VAEEncoder object)'''
    
    '''Outputs:
                    zs: a latent vector modified by some distribution defined by the standard deviation (Pytorch float.32 tensor)'''

    z_list = []
    vae_encoder.eval()
    batch_size = 1024 


    data = selfies_to_one_hot(encoding_list, encoding_alphabet, largest_molecule_len)
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

        zs_sub, _, _ = vae_encoder(sub_flat)
        z_list.append(zs_sub.to('cpu'))

    zs = torch.cat(z_list)

    return zs

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


def gen_properties_tensor(my_file):
    
    '''Generates the properties tensor'''

    '''Arguments:
                    my_file: dataframe containing SMILES + 1 other properties column. (Pandas DataFrame)'''
    
    '''Outputs:
                    properties_tensor: Pytorch tensor containing the desired properties (Pytorch float.32 tensor)'''

    properties_df = my_file.drop(columns=['smiles'])
    properties_array = properties_df.to_numpy() 
    properties_tensor = torch.tensor(properties_array,dtype=torch.float32)

    return properties_tensor

def lpoints_to_onehots(zs, selfies_alphabet, vae_decoder):

    '''Converts latent vectors to onehots. Useful for if you do not care about getting the SMILES/SELFIES output. This function
        removes non-unique one-hots and so is only used in the brute force code.'''

    '''Arguments:
                    zs: a latent vector modified by some distribution defined by the standard deviation (Pytorch float.32 tensor)
                    selfies_alphabet: the alphabet of the SELFIES encoding (list)
                    vae_decoder: the decoder object (VAEDecoder object)'''
    
    '''Outputs:
                    data: pytorch tensor of representing the SELFIES output from the decoder (Pytorch float.32 tensor)'''



    data_list = []
    big_data_list = []
    if zs.dim() <2:
        zs = zs.unsqueeze(0)
    zs = zs.to('cpu')
    
    len_max_molec = 79
    DATA_SIZE = zs.shape[0] * len(selfies_alphabet) * len_max_molec
    ONEHOT_SIZE = 4096 * 2 * 4 * len_max_molec * (vae_decoder.decode_RNN.weight_hh_l0.shape[0])
    GPU_USAGE = DATA_SIZE + ONEHOT_SIZE
    FREE_MEM = get_free_memory(device)
    MEM_RATIO = FREE_MEM/(GPU_USAGE)

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


    num_batches_train = int(len(sequence_tensor) / batch_size)
    if num_batches_train < len(sequence_tensor)/batch_size:
        num_batches_train = num_batches_train + 1
    PAUSE_ITERATION = int(0.5*num_batches_train*MEM_RATIO) + 1



    for batch_iteration in range(num_batches_train):
        start_idx = batch_iteration * batch_size
        stop_idx = (batch_iteration + 1) * batch_size
        batch = sequence_tensor[start_idx: stop_idx].to(device)
        data = torch.nn.functional.one_hot(batch, num_classes = len(selfies_alphabet)).to(torch.bool)
        data_list.append(data)

        if batch_iteration % PAUSE_ITERATION == 0:
           
            data = torch.cat(data_list).to('cpu')
            big_data_list.append(data)
            data_list = []

    if len(data_list) > 0:
        data = torch.cat(data_list).to('cpu')
        big_data_list.append(data)
        
    one_hot_tensor = torch.cat(big_data_list).to(torch.float32)


    return one_hot_tensor


def onehots_to_lpoints(data, vae_encoder):

    '''Converts latent vectors to onehots. Useful for if you do not care about getting the SMILES/SELFIES output. This function
        removes non-unique one-hots and so is only used in the brute force code.'''

    '''Arguments:
                    data: pytorch tensor of representing the some SELFIES (Pytorch float.32 tensor)
                    vae_encoder: the encoder object (VAEEncoder object)'''
    
    '''Outputs:
                    mus: the mean latent vectors (Pytorch float.32 tensor)'''

    mu_list = []
    batch_size = 1024



    inp_flat_one_hot = data.flatten(start_dim=1)
    inp_flat_one_hot = inp_flat_one_hot.unsqueeze(0)


    num_batches_train = int(data.shape[0] / batch_size)
    if num_batches_train < data.shape[0] / batch_size:
        num_batches_train = num_batches_train + 1

    vae_encoder.eval()

    for batch_iteration in range(num_batches_train):

        start_idx = batch_iteration * batch_size
        stop_idx = (batch_iteration + 1) * batch_size
        sub_flat_ = inp_flat_one_hot.squeeze()
        if sub_flat_.dim() < 2:
            sub_flat_ = sub_flat_.unsqueeze(0)
        sub_flat = sub_flat_[start_idx: stop_idx].unsqueeze(0).to(device)

        _, mus_sub, _ = vae_encoder(sub_flat)
        mu_list.append(mus_sub)

    

    mus = torch.cat(mu_list)

    return mus


def canonicalize_smiles(mol):

    '''Function to generate canonical SMILES'''
    
    '''Arguments:
                    mol: molecular fingerprint object generated from Chem.MolFromSmiles (rdkit.Chem.rdchem.Mol object)'''    
    
    '''Outputs:
                    Chem.MolToSmiles(mol, canonical=True): canonical SMILES of the molecule (str)'''
    

    if mol:
        return Chem.MolToSmiles(mol, canonical=True)
    else:
        return None  # In case the SMILES is invalid

def fp_generation(smiles_list):

    '''Function to transform a list of SMILES into daylight fingerprints from the Chem package'''
    
    '''Arguments:
                    smiles_list: the list of SMILES encodings (list)'''    
    
    '''Outputs:
                    fp: Either a 0 or daylight fingerprint (ExplicitBitVect) '''


    fp = []
    for i in range(len(smiles_list)):
        try:
            fp.append(fpgen.GetFingerprint(Chem.MolFromSmiles(smiles_list[i])).ToBitString())
        except:
            fp.append(0)

    return fp
        

def make_model(size_feature, width, activation, dropout, n_layers, l_rate, batch_size, loss_fn, size_target):
        
        '''Function to generate the MLP used in network B'''
    
        '''Arguments:
                        size_feature: the size of the number of features inputted into the MLP, e.g., if inputting daylight fp, morgan fp and one-hot encoding, 
                                        the size is then 3 (int)
                        width: the number of nodes in a given layer (int)
                        activation: the pytorch activation function (Torch.nn function)
                        dropout: size of dropout (float)
                        n_layers: number of layers (int)
                        l_rate: learning rate (float)
                        batch_size: batch size for training (int)
                        loss_fn: the pytorch loss function (Torch.nn function)
                        size_target: size of the output, e.g., for 1st and 2nd OS + dE, this is 4'''    
        
        '''Outputs:
                        MLP: the output is the MLP (dc.models.TorchModel object) '''
        
        layers = []
        
        # Input layer
        layers.append(torch.nn.Linear(size_feature, width))
        layers.append(torch.nn.BatchNorm1d(width))
        layers.append(activation)
        layers.append(torch.nn.Dropout(dropout))
        
        # Hidden layers
        for _ in range(n_layers - 1):
                layers.append(torch.nn.Linear(width, width))
                layers.append(torch.nn.BatchNorm1d(width))
                layers.append(activation)
                layers.append(torch.nn.Dropout(dropout))
        
        # Output layer
        layers.append(torch.nn.Linear(width, size_target))
        
        # Create the sequential model    
        return dc.models.TorchModel(torch.nn.Sequential(*layers),loss_fn, learning_rate=dc.models.optimizers.ExponentialDecay(l_rate, 0.90, 300) ,batch_size=batch_size)


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

def save_index_fp(index, settings):

    '''Function to save the index of a completed loop in the fingerprint generation of the brute force code'''

    '''Arguments:
                    settings: settings defined by the corresponding .yml file (dict)'''    


    save_path = settings['data']['save_path']
    log_filename = 'fp_index_list.txt'

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
