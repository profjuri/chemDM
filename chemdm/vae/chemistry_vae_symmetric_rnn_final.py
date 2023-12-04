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
import psutil

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


def save_models(encoder, decoder, epoch, lr_new_enc, KLD, settings, alphabet):

    gru_stack_size = settings['encoder']['gru_stack_size']
    gru_neurons_num = settings['encoder']['gru_neurons_num']
    latent_dim = settings['encoder']['latent_dimension']
    save_path = settings['data']['save_path']


    out_dir = str(save_path) + 'stack_size' + str(gru_stack_size)  + 'neurons_num' + str(gru_neurons_num) + '_l_dim' + str(latent_dim) + '/{}'.format(epoch) 
    _make_dir(out_dir)
    torch.save(encoder.state_dict(), '{}/E'.format(out_dir))
    torch.save(decoder.state_dict(), '{}/D'.format(out_dir))

    settings_folder = str(save_path) + 'stack_size' + str(gru_stack_size)  + 'neurons_num' + str(gru_neurons_num) + '_l_dim' + str(latent_dim) + '/settings'

    log_folder = settings_folder
    log_filename = 'settings.yml'

    if not os.path.exists(settings_folder):
        os.makedirs(settings_folder)

        log_filepath = os.path.join(log_folder, log_filename)
        data = {**settings, 'alphabet': alphabet}

        with open(log_filepath, 'w') as file:
            yaml.dump(data, file)

def save_models_epoch_loss(epoch, loss, recon_loss, val_loss, kld_loss, lr_new_enc, KLD, settings):

    gru_stack_size = settings['encoder']['gru_stack_size']
    gru_neurons_num = settings['encoder']['gru_neurons_num']
    latent_dim = settings['encoder']['latent_dimension']
    save_path = settings['data']['save_path']



    out_dir = str(save_path) + 'stack_size' + str(gru_stack_size)  + 'neurons_num' + str(gru_neurons_num) + '_l_dim' + str(latent_dim) + '/0'
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
            file.write("epoch,loss,recon_loss,kld_loss,ratio,val_loss\n")
        file.write(f'{epoch},{loss},{recon_loss},{kld_loss},{ratio},{val_loss}\n')


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


def train_model(vae_encoder, vae_decoder, data_train, data_valid ,num_epochs, batch_size,
                lr_enc, lr_dec, KLD_alpha,
                sample_num, sample_len, alphabet, settings):
    """
    Train the Variational Auto-Encoder
    """


    # initialize an instance of the model
    

    optimizer_encoder = torch.optim.Adam(vae_encoder.parameters(), lr=lr_enc)
    optimizer_decoder = torch.optim.Adam(vae_decoder.parameters(), lr=lr_dec)
    scheduler = ReduceLROnPlateau(optimizer_encoder, mode='min', factor=0.5, patience=10, verbose=True)
    scheduler2 = ReduceLROnPlateau(optimizer_decoder, mode='min', factor=0.5, patience=10, verbose=True)

    data_train = data_train.clone().detach().to(device)
    num_batches_train = int(len(data_train) / batch_size)


    for epoch in range(num_epochs):
        start = time.time()
        for batch_iteration in range(num_batches_train):  # batch iterator


            # manual batch iterations
            start_idx = batch_iteration * batch_size
            stop_idx = (batch_iteration + 1) * batch_size
            batch = data_train[start_idx: stop_idx]

            # reshaping for efficient parallelization
            inp_flat_one_hot = batch.flatten(start_dim=1)
            inp_flat_one_hot = inp_flat_one_hot.unsqueeze(0)

            latent_points, mus, log_vars = vae_encoder(inp_flat_one_hot)
            latent_points = latent_points.unsqueeze(0)

            hidden = vae_decoder.init_hidden(batch_size=batch_size)

            # decoding from RNN N times, where N is the length of the largest
            # molecule (all molecules are padded)
            out_one_hot = torch.zeros_like(batch, device=device)
            for seq_index in range(batch.shape[1]):
                out_one_hot_line, hidden = vae_decoder(latent_points, hidden)
                out_one_hot[:, seq_index, :] = out_one_hot_line[0]


            
            # compute ELBO
            loss, recon_loss, kld_loss = compute_elbo(batch, out_one_hot, mus, log_vars, KLD_alpha) ##switched the out_one_hot and the batch variables around
            loss = loss 
            # perform back propogation
            optimizer_encoder.zero_grad()
            optimizer_decoder.zero_grad()
            loss.backward(retain_graph=True)
            
            nn.utils.clip_grad_norm_(vae_decoder.parameters(), 10.0)
            optimizer_encoder.step()
            optimizer_decoder.step()

            if batch_iteration % 30 ==0:
                print('Batch:', batch_iteration, '/', num_batches_train, 'loss:', loss.item())


        scheduler.step(loss)
        scheduler2.step(loss)


        if epoch % 50 == 0:
            save_models(vae_encoder, vae_decoder, epoch, lr_enc, KLD_alpha, settings, alphabet)
            

        vae_encoder.eval()
        vae_decoder.eval()
        
        inp_flat_one_hot = data_valid.flatten(start_dim=1)
        inp_flat_one_hot = inp_flat_one_hot.unsqueeze(0)
        latent_points, mus, log_vars = vae_encoder(inp_flat_one_hot)
        latent_points = latent_points.unsqueeze(0)

        hidden = vae_decoder.init_hidden(batch_size=data_valid.shape[0])

        out_one_hot = torch.zeros_like(data_valid, device=device)
        for seq_index in range(data_valid.shape[1]):
            out_one_hot_line, hidden = vae_decoder(latent_points, hidden)
            out_one_hot[:, seq_index, :] = out_one_hot_line[0]

        vae_encoder.train()
        vae_decoder.train()

        val_loss = validation_loss(data_valid, out_one_hot, mus)

        save_models_epoch_loss(epoch, loss.item(), recon_loss, val_loss.item(), kld_loss, lr_enc, KLD_alpha, settings)




def compute_elbo(x, x_hat, mus, log_vars, KLD_alpha):


    inp = x_hat.reshape(-1, x_hat.shape[2])
    target = x.reshape(-1, x.shape[2]).argmax(1)

    criterion = torch.nn.CrossEntropyLoss()
    recon_loss = criterion(inp, target)
    kld = -0.5 * torch.mean(1. + log_vars - mus.pow(2) - log_vars.exp())
    ratio = recon_loss/(KLD_alpha * kld)



    total_loss = recon_loss + (KLD_alpha * kld) 


    return total_loss, recon_loss, KLD_alpha * kld

def validation_loss(x, x_hat, mus):


    inp = x_hat.reshape(-1, x_hat.shape[2])
    target = x.reshape(-1, x.shape[2]).argmax(1)
    criterion = torch.nn.CrossEntropyLoss()
    recon_loss = criterion(inp, target)


    return recon_loss


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

def data_init(settings, device):

    file_name_smiles = settings['data']['smiles_file']
    full_alphabet_set = set(settings['data']['full_alphabet_set'])
    torch_seed = settings['data']['torch_seed']

    torch.manual_seed(torch_seed)


    encoding_list, encoding_alphabet, largest_molecule_len, _, _, _ = \
        get_selfie_and_smiles_encodings_for_dataset(file_name_smiles)
        
    for letter in full_alphabet_set:
        if letter not in encoding_alphabet:
            encoding_alphabet.append(letter)

    data = multiple_selfies_to_hot(encoding_list, largest_molecule_len, encoding_alphabet)

    len_max_molec = data.shape[1]
    len_alphabet = data.shape[2]
    len_max_mol_one_hot = len_max_molec * len_alphabet

    data = torch.tensor(data, dtype=torch.float).to(device)

    train_valid_test_size = [0.8, 0.2, 0.0]

    idx_train_val = int(len(data) * train_valid_test_size[0])
    idx_val_test = idx_train_val + int(len(data) * train_valid_test_size[1])

    data_train = data[0:idx_train_val]
    data_valid = data[idx_train_val:idx_val_test]

    return data_train, data_valid, len_max_molec, len_alphabet, len_max_mol_one_hot, encoding_alphabet


def main():
    if os.path.exists("selfies_rnn.yml"):
        settings = yaml.safe_load(open("selfies_rnn.yml", "r"))
    else:
        print("Expected a file settings.yml but didn't find it.")
        return

    data_parameters = settings['data']
    batch_size = data_parameters['batch_size']
    save_path = data_parameters['save_path']
    encoder_parameter = settings['encoder']
    decoder_parameter = settings['decoder']
    training_parameters = settings['training']
    


    torch.cuda.empty_cache()
    data_train, data_valid, len_max_molec, len_alphabet, len_max_mol_one_hot, encoding_alphabet = data_init(settings, device)
    vae_encoder = VAEEncoder(in_dimension=len_max_mol_one_hot, **encoder_parameter).to(device)
    vae_decoder = VAEDecoder(**decoder_parameter, out_dimension=len_alphabet).to(device)


    print("start training")
    train_model(**training_parameters,
                vae_encoder=vae_encoder,
                vae_decoder=vae_decoder,
                batch_size=batch_size,
                data_train=data_train,
                data_valid=data_valid,
                alphabet=encoding_alphabet,
                sample_len=len_max_molec,
                settings=settings 
                )



if __name__ == '__main__':
    try:
        main()
    except AttributeError:
        _, error_message, _ = sys.exc_info()
        print(error_message)
