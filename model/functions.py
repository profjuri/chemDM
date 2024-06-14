import os
import pandas as pd
import torch
import yaml
import selfies as sf
import numpy as np

import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils.rnn import pad_sequence


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



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
    #selfies_alphabet.insert(1, '.')

    largest_selfies_len = max(sf.len_selfies(s) for s in selfies_list)

    print('Finished translating SMILES to SELFIES.')

    return selfies_list, selfies_alphabet, largest_selfies_len


def remove_unrecognized_symbols(smiles_list):

    '''Removes blank spaces from the SMILES encodings'''

    '''Arguments:
                    smiles_list: the list of SMILES encodings (list)'''
    
    '''Outputs:
                    cleaned_smiles: the cleaned SMILES encodings list (list)'''

    cleaned_smiles = [smiles.replace('\n', '') for smiles in smiles_list]

    return cleaned_smiles


def selfies_to_lpoints(encoding_list, encoding_alphabet, largest_molecule_len, vae_encoder, lpoint_size):


    alphabet_dict = {letter: index for index, letter in enumerate(encoding_alphabet)}


    integer_encoded = [[alphabet_dict[symbol] for symbol in sf.split_selfies(encoding_list[x])] for x in range(len(encoding_list))]
    max_length = max(len(inner_list) for inner_list in integer_encoded)
    padded_list = [torch.tensor(inner_list + [0] * (max_length - len(inner_list))) for inner_list in integer_encoded]
    padded_tensor = pad_sequence(padded_list, batch_first=True, padding_value=0)


    if padded_tensor.shape[1] < largest_molecule_len:
        extra_padding = torch.zeros(padded_tensor.shape[0],(largest_molecule_len - padded_tensor.shape[1]), dtype = torch.int64)
        padded_tensor = torch.cat((padded_tensor, extra_padding),dim=1)

    


    data = torch.nn.functional.one_hot(padded_tensor, num_classes = len(encoding_alphabet)).to(torch.float32)
    mus = torch.empty(0, dtype=torch.float32).to('cpu')

    print('data.device', data.device)

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
        mus_sub = mus_sub.to('cpu')

        mus = torch.cat((mus, mus_sub))
    
    #vae_encoder.train()

    return mus

def get_free_memory(device):
    memory_stats = torch.cuda.memory_stats(device)
    total_memory = memory_stats["allocated_bytes.all.peak"]
    memory_allocated = memory_stats["allocated_bytes.all.current"]
    free_mem = total_memory - memory_allocated

    return free_mem


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
        extra_padding = torch.zeros(padded_tensor.shape[0],(largest_molecule_len - padded_tensor.shape[1]), dtype = torch.int64)
        padded_tensor = torch.cat((padded_tensor, extra_padding),dim=1)
    


    one_hot = torch.nn.functional.one_hot(padded_tensor, num_classes = len(encoding_alphabet))

    return one_hot

def _make_dir(directory):

    '''Makes the directory'''

    '''Arguments:
                    directory: directory path (str)'''
    os.makedirs(directory)


def get_free_memory(device):

    '''Calculate the amount of free memory'''

    '''Arguments:
                    device: thr device being used (Pytorch object)'''
    
    '''Outputs:
                    free_mem: the amount of free memory in bytes (float)'''


    memory_stats = torch.cuda.memory_stats(device)
    total_memory = memory_stats["allocated_bytes.all.peak"]
    memory_allocated = memory_stats["allocated_bytes.all.current"]
    free_mem = total_memory - memory_allocated

    return free_mem

def decode_lpoints(zs, selfies_alphabet, vae_decoder, len_max_molec):


    final_smiles_list = []
    zs = zs.squeeze()

    batch_size = 128
    num_batches_train = int(len(zs) / batch_size)

    if num_batches_train < len(zs)*batch_size:
        num_batches_train = num_batches_train + 1

    for batch_iteration in range(num_batches_train):

        start_idx = batch_iteration * batch_size
        stop_idx = (batch_iteration + 1) * batch_size
        batch = zs[start_idx: stop_idx].to(device)

        if batch.dim() <2:
            batch = batch.unsqueeze(0)


        seq_tensor = torch.empty(0, dtype=torch.float32).to(device)
        hidden = vae_decoder.init_hidden(batch_size=batch.shape[0])

        

        for seq_index in range(len_max_molec):
            vae_decoder.eval()
            with torch.no_grad():

                out_one_hot_line, hidden = vae_decoder(batch.unsqueeze(0), hidden)
                out_one_hot_line = out_one_hot_line.squeeze()

                if out_one_hot_line.dim() < 2:
                    out_one_hot_line = out_one_hot_line.unsqueeze(0)


                out_one_hot_line_arg = torch.argmax(out_one_hot_line, dim = 1).unsqueeze(0)
                seq_tensor = torch.cat((seq_tensor, out_one_hot_line_arg), dim = 0)


        

        sequences = seq_tensor.squeeze().t()
        if sequences.dim() < 2:
            sequences = sequences.unsqueeze(0)
        list_of_continuous_strings = [sf.decoder(''.join([selfies_alphabet[int(i)] for i in row])) for row in sequences]
        final_smiles_list = final_smiles_list + list_of_continuous_strings

    return final_smiles_list

def gen_properties_tensor(my_file):
    properties_df = my_file.drop(columns=['smiles'])
    properties_array = properties_df.to_numpy() 
    properties_tensor = torch.tensor(properties_array,dtype=torch.float32)

    return properties_tensor
