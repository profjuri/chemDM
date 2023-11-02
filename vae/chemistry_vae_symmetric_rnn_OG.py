#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SELFIES: a robust representation of semantically constrained graphs with an
    example application in chemistry (https://arxiv.org/abs/1905.13741)
    by Mario Krenn, Florian Haese, AkshatKuman Nigam, Pascal Friederich,
    Alan Aspuru-Guzik.

    Variational Autoencoder (VAE) for chemistry
        comparing SMILES and SELFIES representation using reconstruction
        quality, diversity and latent space validity as metrics of
        interest

information:
    ML framework: pytorch
    chemistry framework: RDKit

    get_selfie_and_smiles_encodings_for_dataset
        generate complete encoding (inclusive alphabet) for SMILES and
        SELFIES given a data file

    VAEEncoder
        fully connected, 3 layer neural network - encodes a one-hot
        representation of molecule (in SMILES or SELFIES representation)
        to latent space

    VAEDecoder
        decodes point in latent space using an RNN

    latent_space_quality
        samples points from latent space, decodes them into molecules,
        calculates chemical validity (using RDKit's MolFromSmiles), calculates
        diversity
"""

import os
import sys
import time
import random
import pickle
import json

import numpy as np
import pandas as pd
import torch
import yaml
from rdkit import rdBase
from rdkit.Chem import MolFromSmiles
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR
import Levenshtein
import torch.distributions as dist
from scipy.stats import pearsonr
import torch.nn.functional as F
from torch.autograd import gradcheck

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:25"

import datetime
timestr = datetime.datetime.now().strftime("%Y%m%d%H%M%S")  # Generate timestamp string

import selfies as sf
import data_loader


from data_loader import \
    multiple_selfies_to_hot, multiple_smile_to_hot

rdBase.DisableLog('rdApp.error')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_dir(directory):
    os.makedirs(directory)


def save_models(encoder, decoder, epoch, lr_new_enc, KLD, settings, alphabet, num_counts):

    gru_stack_size = settings['encoder']['gru_stack_size']
    gru_neurons_num = settings['encoder']['gru_neurons_num']
    latent_dim = settings['encoder']['latent_dimension']
    save_path = settings['data']['save_path']


    out_dir = str(save_path) + 'stack_size' + str(gru_stack_size)  + 'neurons_num' + str(gru_neurons_num) + '_l_dim' + str(latent_dim) + 'num_count' + str(num_counts) + '/{}'.format(epoch) 
    _make_dir(out_dir)
    torch.save(encoder, '{}/E'.format(out_dir))
    torch.save(decoder, '{}/D'.format(out_dir))


    log_folder = out_dir  # Replace with the desired folder path
    log_filename = 'settings_log.txt'

    # Create the log folder if it doesn't exist
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    # Generate the full file path
    log_filepath = os.path.join(log_folder, log_filename)

    # Open the file in write mode and write the settings
    with open(log_filepath, 'w') as log_file:
        log_file.write("Program Settings:\n")
        log_file.write(json.dumps(settings, indent=4))
        log_file.write(json.dumps(alphabet, indent=4))

    print("Settings log file created successfully at:", log_filepath)


def save_models_epoch_loss(epoch, loss, recon_loss, kld_loss, lr_new_enc, KLD, latent_loss, settings, num_count, score, val_score):

    gru_stack_size = settings['encoder']['gru_stack_size']
    gru_neurons_num = settings['encoder']['gru_neurons_num']
    latent_dim = settings['encoder']['latent_dimension']
    save_path = settings['data']['save_path']



    out_dir = str(save_path) + 'stack_size' + str(gru_stack_size)  + 'neurons_num' + str(gru_neurons_num) + '_l_dim' + str(latent_dim) + 'num_count' + str(num_count) + '/0'
    log_folder = out_dir  # Replace with the desired folder path
    log_filename = 'epoch_loss.txt'

    log_filepath = os.path.join(log_folder, log_filename)

    # Create the log folder if it doesn't exist
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    file_exists = os.path.isfile(log_filepath)
    ratio = recon_loss/kld_loss
    
    with open(log_filepath, 'a') as file:
        if not file_exists:
            file.write("epoch,loss,recon_loss,kld_loss,latent_loss,ratio,score,val_score\n")
        file.write(f'{epoch},{loss},{recon_loss},{kld_loss},{latent_loss},{ratio},{score},{val_score}\n')


class VAEEncoder(nn.Module):

    def __init__(self, in_dimension, gru_stack_size, gru_neurons_num,
                 latent_dimension):
        """
        Recurrent layers to encode molecule to latent space
        """
        super(VAEEncoder, self).__init__()
        self.latent_dimension = latent_dimension
        self.gru_stack_size = gru_stack_size
        self.gru_neurons_num = gru_neurons_num

        # RNN Encoder
        self.encode_RNN = nn.GRU(
            input_size=in_dimension,
            hidden_size=gru_neurons_num,
            num_layers=gru_stack_size,
            batch_first=False)

        self.encode_FC_mu = nn.Sequential(
            nn.Linear(gru_neurons_num, latent_dimension),
        )

        self.encode_FC_log_var = nn.Sequential(
            nn.Linear(gru_neurons_num, latent_dimension),
        )

    @staticmethod
    def reparameterize(mu, log_var):
        """
        Reparameterization trick
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x, hidden=None):
        """
        Pass through the Encoder RNN
        """
        if hidden is None:
            hidden = self.init_hidden(x.size(1))  # Initialize hidden state based on batch size

        # Encode using RNN
        rnn_output, hidden = self.encode_RNN(x, hidden)
        rnn_output_last = rnn_output[-1, :, :]  # Consider the last RNN output

        # Latent space mean
        mu = self.encode_FC_mu(rnn_output_last)

        # Latent space variance
        log_var = self.encode_FC_log_var(rnn_output_last)

        # Reparameterize
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var

    def init_hidden(self, batch_size=1):
        weight = next(self.parameters())
        return weight.new_zeros(self.gru_stack_size, batch_size,
                                self.gru_neurons_num)


class VAEDecoder(nn.Module):

    def __init__(self, latent_dimension, gru_stack_size, gru_neurons_num,
                 out_dimension):
        """
        Through Decoder
        """
        super(VAEDecoder, self).__init__()
        self.latent_dimension = latent_dimension
        self.gru_stack_size = gru_stack_size
        self.gru_neurons_num = gru_neurons_num

        # Simple Decoder
        self.decode_RNN = nn.GRU(
            input_size=latent_dimension,
            hidden_size=gru_neurons_num,
            num_layers=gru_stack_size,
            batch_first=False)

        self.decode_FC = nn.Sequential(
            nn.Linear(gru_neurons_num, out_dimension),
        )

    def init_hidden(self, batch_size=1):
        weight = next(self.parameters())
        return weight.new_zeros(self.gru_stack_size, batch_size,
                                self.gru_neurons_num)

    def forward(self, z, hidden):
        """
        A forward pass throught the entire model.
        """

        # Decode
        l1, hidden = self.decode_RNN(z, hidden)
        decoded = self.decode_FC(l1)  # fully connected layer

        return decoded, hidden


def train_model(selfies_list, properties_tensor_train, properties_tensor_valid, vae_encoder, vae_decoder, data_full_train, data_full_valid,
                data_train, data_valid, num_epochs, batch_size,
                lr_enc, lr_dec, KLD_alpha,
                sample_num, sample_len, alphabet, type_of_encoding, test_count, latent_loss_activation, settings):
    """
    Train the Variational Auto-Encoder
    """


    # initialize an instance of the model
    optimizer_encoder = torch.optim.Adam(vae_encoder.parameters(), lr=lr_enc)
    optimizer_decoder = torch.optim.Adam(vae_decoder.parameters(), lr=lr_dec)
    scheduler = ReduceLROnPlateau(optimizer_encoder, mode='min', factor=0.5, patience=10, verbose=True)
    scheduler2 = ReduceLROnPlateau(optimizer_decoder, mode='min', factor=0.5, patience=10, verbose=True)




    data_train = data_train.clone().detach().to(device)
    data_train_latent = data_full_train.to(device)
    data_valid_latent = data_full_valid.to(device)
    num_batches_train = int(len(data_train) / batch_size)

    quality_valid_list = [0, 0, 0, 0]


    for epoch in range(num_epochs):


        
        data_train = data_train[torch.randperm(data_train.size()[0])]

        start = time.time()
        for batch_iteration in range(num_batches_train):  # batch iterator

            # manual batch iterations
            start_idx = batch_iteration * batch_size
            stop_idx = (batch_iteration + 1) * batch_size
            batch = data_train[start_idx: stop_idx]

            # reshaping for efficient parallelization
            inp_flat_one_hot = batch.flatten(start_dim=1)
            inp_flat_one_hot = inp_flat_one_hot.unsqueeze(0)

            inp_flat_one_hot_l = data_train_latent.flatten(start_dim=1)
            inp_flat_one_hot_l = inp_flat_one_hot_l.unsqueeze(0)

            inp_flat_one_hot_l_val = data_valid_latent.flatten(start_dim=1)
            inp_flat_one_hot_l_val = inp_flat_one_hot_l_val.unsqueeze(0)


            latent_points, mus, log_vars = vae_encoder(inp_flat_one_hot)

            if latent_loss_activation == 1:

                # Latent loss calculation preparation


                latent_points_l, mus_l, log_vars_l = vae_encoder(inp_flat_one_hot_l)
                latent_points_l_val, mus_l_val, log_vars_l_val = vae_encoder(inp_flat_one_hot_l_val)
                latent_points_l_tensor = latent_points_l
                properties_tensor_train = properties_tensor_train
                properties_tensor_valid = properties_tensor_valid
                latent_loss, score, val_score = latent_loss_calc(latent_points_l_tensor, properties_tensor_train, properties_tensor_valid, latent_points_l_val)

            else:
                latent_loss = torch.tensor([0]).to(device)
                score = 0

            latent_points = latent_points.unsqueeze(0)

            hidden = vae_decoder.init_hidden(batch_size=batch_size)

            # decoding from RNN N times, where N is the length of the largest
            # molecule (all molecules are padded)
            out_one_hot = torch.zeros_like(batch, device=device)
            for seq_index in range(batch.shape[1]):
                out_one_hot_line, hidden = vae_decoder(latent_points, hidden)
                out_one_hot[:, seq_index, :] = out_one_hot_line[0]


            # Latent loss calculation preparation

 


            
            # compute ELBO
            loss, recon_loss, kld_loss = compute_elbo(batch, out_one_hot, mus, log_vars, KLD_alpha, test_count) ##switched the out_one_hot and the batch variables around
            loss = loss + latent_loss
            # perform back propogation
            optimizer_encoder.zero_grad()
            optimizer_decoder.zero_grad()
            loss.backward(retain_graph=True)
            
            nn.utils.clip_grad_norm_(vae_decoder.parameters(), 10.0)
            optimizer_encoder.step()
            optimizer_decoder.step()


        scheduler.step(loss)
        scheduler2.step(loss)







        if epoch % 50 == 0:
            save_models(vae_encoder, vae_decoder, epoch, lr_enc, KLD_alpha, settings, alphabet, test_count)
            
                

        save_models_epoch_loss(epoch, loss.item(), recon_loss, kld_loss, lr_enc, KLD_alpha, latent_loss.item(), settings, test_count, score, val_score)






def latent_loss_calc(latents, properties_tensor_train, properties_tensor_valid, latent_points_l_val):

    ###
    latents_array_trans = latents.t() 
    val_trans = latent_points_l_val.t() 
    correlation_coeffs_tensor = torch.empty(0, dtype=torch.float32).to(device)
    correlation_coeffs_tensor_val = torch.empty(0, dtype=torch.float32).to(device)


    for i in range(latents_array_trans.shape[0]):
        x = latents_array_trans[i]
        y = properties_tensor_train

        vx = x - torch.mean(x)
        vy = y - torch.mean(y)

        correlation_coeffs = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        correlation_coeffs_tensor = torch.cat((correlation_coeffs_tensor, correlation_coeffs.unsqueeze(0)))

    score = correlation_coeffs_tensor.mean()

    ###
    for i in range(val_trans.shape[0]):
        x = val_trans[i]
        y = properties_tensor_valid

        vx = x - torch.mean(x)
        vy = y.float() - torch.mean(y.float())


        correlation_coeffs = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        correlation_coeffs_tensor_val = torch.cat((correlation_coeffs_tensor, correlation_coeffs.unsqueeze(0)))

    score_val = correlation_coeffs_tensor_val.mean()




    latent_loss = (score-0.3).pow(2)

    return latent_loss, score, score_val


def compute_elbo(x, x_hat, mus, log_vars, KLD_alpha, num_counts):


    inp = x_hat.reshape(-1, x_hat.shape[2])
    target = x.reshape(-1, x.shape[2]).argmax(1)

    criterion = torch.nn.CrossEntropyLoss()
    recon_loss = criterion(inp, target)
    kld = -0.5 * torch.mean(1. + log_vars - mus.pow(2) - log_vars.exp())
    ratio = recon_loss/(KLD_alpha * kld)



    total_loss = recon_loss + (KLD_alpha * kld) 
    #total_loss = latent_loss



    return total_loss, recon_loss, KLD_alpha * kld


def compute_recon_quality(x, x_hat):
    x_indices = x.reshape(-1, x.shape[2]).argmax(1)
    x_hat_indices = x_hat.reshape(-1, x_hat.shape[2]).argmax(1)

    differences = 1. - torch.abs(x_hat_indices - x_indices)
    differences = torch.clamp(differences, min=0., max=1.).double()
    quality = 100. * torch.mean(differences)
    quality = quality.detach().cpu().numpy()

    return quality


def get_selfie_and_smiles_encodings_for_dataset(file_path):
    """
    Returns encoding, alphabet and length of largest molecule in SMILES and
    SELFIES, given a file containing SMILES molecules.

    input:
        csv file with molecules. Column's name must be 'smiles'.
    output:
        - selfies encoding
        - selfies alphabet
        - longest selfies string
        - smiles encoding (equivalent to file content)
        - smiles alphabet (character based)
        - longest smiles string
    """

    df = pd.read_csv(file_path)
    df = df.dropna()
    print(sf.get_semantic_constraints())
    print(df)

    smiles_list = np.asanyarray(df.smiles)

    smiles_list = remove_unrecognized_symbols(smiles_list)

    smiles_alphabet = list(set(''.join(smiles_list)))
    smiles_alphabet.append(' ')  # for padding

    largest_smiles_len = len(max(smiles_list, key=len))

    print('--> Translating SMILES to SELFIES...')
    selfies_list = list(map(sf.encoder, smiles_list))

    all_selfies_symbols = sf.get_alphabet_from_selfies(selfies_list)
    selfies_alphabet = list(all_selfies_symbols)
    selfies_alphabet.insert(0, '[nop]')
    print(len(selfies_alphabet))

    largest_selfies_len = max(sf.len_selfies(s) for s in selfies_list)

    print('Finished translating SMILES to SELFIES.')

    return selfies_list, selfies_alphabet, largest_selfies_len, \
           smiles_list, smiles_alphabet, largest_smiles_len



def remove_unrecognized_symbols(smiles_list):
    cleaned_smiles = [smiles.replace('\n', '') for smiles in smiles_list]
    return cleaned_smiles


def main():
    content = open('logfile.dat', 'w')
    content.close()
    content = open('results.dat', 'w')
    content.close()

    if os.path.exists("selfies_rnn.yml"):
        settings = yaml.safe_load(open("selfies_rnn.yml", "r"))
    else:
        print("Expected a file settings.yml but didn't find it.")
        return

    print('--> Acquiring data...')
    type_of_encoding = settings['data']['type_of_encoding']
    file_name_smiles = settings['data']['smiles_file']
    latent_loss_activation = settings['data']['latent_loss']
    full_alphabet_set = set(settings['data']['full_alphabet_set'])

    focus_choice = settings['data']['focus_file']
    focus_smiles = settings['data']['focus_smiles']


    print('Finished acquiring data.')

    if type_of_encoding == 0:
        print('Representation: SMILES')
        _, _, _, encoding_list, encoding_alphabet, largest_molecule_len = \
            get_selfie_and_smiles_encodings_for_dataset(file_name_smiles)

        print('--> Creating one-hot encoding...')
        data = multiple_smile_to_hot(encoding_list, largest_molecule_len,
                                     encoding_alphabet)
        
        if focus_choice == 1:
            print('Focusing on specific subset...')
            _, _, _, encoding_list2, _, _ = get_selfie_and_smiles_encodings_for_dataset(focus_smiles)
            
            data_2 = multiple_smile_to_hot(encoding_list2, largest_molecule_len, encoding_alphabet)

        
        print('Finished creating one-hot encoding.')

    elif type_of_encoding == 1:
        print('Representation: SELFIES')
        encoding_list, encoding_alphabet, largest_molecule_len, _, _, _ = \
            get_selfie_and_smiles_encodings_for_dataset(file_name_smiles)
        
        for letter in full_alphabet_set:
            if letter not in encoding_alphabet:
                encoding_alphabet.append(letter)

        print('--> Creating one-hot encoding...')
        data = multiple_selfies_to_hot(encoding_list, largest_molecule_len, encoding_alphabet)
        
        
        if focus_choice == 1:
            print('Focusing on specific subset...')
            encoding_list2, _, _, _, _, _ = get_selfie_and_smiles_encodings_for_dataset(focus_smiles)
            
            data_2 = multiple_selfies_to_hot(encoding_list2, largest_molecule_len, encoding_alphabet)



        print('Finished creating one-hot encoding.')

    else:
        print("type_of_encoding not in {0, 1}.")
        return
    

    len_max_molec = data.shape[1]
    len_alphabet = data.shape[2]
    len_max_mol_one_hot = len_max_molec * len_alphabet



    print(encoding_alphabet)

    num_tests = settings['data']['num_of_tests']

    for i in range(num_tests):

        test_count = i + 1

        data_parameters = settings['data']
        batch_size = data_parameters['batch_size']
        save_path = data_parameters['save_path']
        encoder_parameter = settings['encoder']
        decoder_parameter = settings['decoder']
        training_parameters = settings['training']




        vae_encoder = VAEEncoder(in_dimension=len_max_mol_one_hot, **encoder_parameter).to(device)
        vae_decoder = VAEDecoder(**decoder_parameter, out_dimension=len_alphabet).to(device)




        my_file = pd.read_csv(str(focus_smiles), index_col=None)##The file you want to train on, should contain SMILES reps, latent space reps and properties
        my_file.dropna()

        list_of_properties = []
        props_used = []


        for i in range(round(len(my_file))):



            GapIn = my_file['energy_gap'][i]
            SMILESCodeIn = my_file['smiles'][i]
            list_of_properties.append([SMILESCodeIn, GapIn])


        for i in range(len(list_of_properties)):
            props_used.append(list_of_properties[i][1])

        properties_tensor = torch.tensor(props_used, dtype=torch.float32).to(device)



        print('*' * 15, ': -->', device)

        data = torch.tensor(data, dtype=torch.float).to(device)
        data_2 = torch.tensor(data_2, dtype=torch.float).to(device)

        train_valid_test_size = [0.8, 0.2, 0.0]
        data = data[torch.randperm(data.size()[0])]
        idx_train_val = int(len(data) * train_valid_test_size[0])
        idx_val_test = idx_train_val + int(len(data) * train_valid_test_size[1])

        data_train = data[0:idx_train_val]
        data_valid = data[idx_train_val:idx_val_test]
        data_2_train = data_2[0:idx_train_val]
        data_2_valid = data_2[idx_train_val:idx_val_test]

        properties_tensor_train = properties_tensor[0:idx_train_val]
        properties_tensor_valid = properties_tensor[idx_train_val:idx_val_test]



        print("start training")
        train_model(encoding_list,
                    properties_tensor_train,
                    properties_tensor_valid,
                    **training_parameters,
                    vae_encoder=vae_encoder,
                    vae_decoder=vae_decoder,
                    batch_size=batch_size,
                    data_full_train = data_2_train,
                    data_full_valid = data_2_valid,
                    data_train=data_train,
                    data_valid=data_valid,
                    alphabet=encoding_alphabet,
                    type_of_encoding=type_of_encoding,
                    sample_len=len_max_molec,
                    test_count = test_count,
                    latent_loss_activation = latent_loss_activation,
                    settings=settings 
                    )

        with open('COMPLETED', 'w') as content:
            content.write('exit code: 0')


if __name__ == '__main__':
    try:
        main()
    except AttributeError:
        _, error_message, _ = sys.exc_info()
        print(error_message)
