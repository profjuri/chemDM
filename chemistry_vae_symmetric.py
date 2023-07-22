
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import yaml
from rdkit import rdBase
from rdkit.Chem import MolFromSmiles
from torch import nn
import torch.distributions as dist 
# added this for reducing lerning rate
from torch.optim.lr_scheduler import StepLR

import selfies as sf
from data_loader import \
    multiple_selfies_to_hot, multiple_smile_to_hot

rdBase.DisableLog('rdApp.error')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_dir(directory):
    os.makedirs(directory)


def save_models(encoder, decoder, epoch):
    out_dir = './saved_models/{}'.format(epoch)
    _make_dir(out_dir)
    torch.save(encoder, '{}/E'.format(out_dir))
    torch.save(decoder, '{}/D'.format(out_dir))

class VAEEncoder(nn.Module):

    def __init__(self, in_dimension, layer_1d, layer_2d, layer_3d,
                 latent_dimension):
        """
        Fully Connected layers to encode molecule to latent space
        """
        super(VAEEncoder, self).__init__()
        self.latent_dimension = latent_dimension

        # Reduce dimension up to second last layer of Encoder
        self.encode_nn = nn.Sequential(
            nn.Linear(in_dimension, layer_1d),
            nn.BatchNorm1d(layer_1d),
            nn.ReLU(),
             nn.Dropout(0.2),  # 20% dropout
            nn.Linear(layer_1d, layer_2d),
            nn.BatchNorm1d(layer_2d),
            nn.ReLU(),
             nn.Dropout(0.2),  # 20% dropout
            nn.Linear(layer_2d, layer_3d),
            nn.BatchNorm1d(layer_3d),
            nn.ReLU()
        )

        # Latent space mean
        self.encode_mu = nn.Linear(layer_3d, latent_dimension)

        # Latent space variance
        self.encode_log_var = nn.Linear(layer_3d, latent_dimension)

    @staticmethod
    def reparameterize(mu, log_var):
        """
        This trick is explained well here:
            https://stats.stackexchange.com/a/16338
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        """
        Pass throught the Encoder
        """
        # Get results of encoder network
        h1 = self.encode_nn(x)

        # latent space
        mu = self.encode_mu(h1)
        log_var = self.encode_log_var(h1)

        # Reparameterize
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var

class VAEDecoder(nn.Module):

    def __init__(self, latent_dimension, layer_3d, layer_2d, layer_1d, out_dimension):
        """
        Fully Connected layers to decode molecule from latent space
        """
        super(VAEDecoder, self).__init__()

        # Decoder Layers (mirroring the Encoder)
        self.decode_nn = nn.Sequential(
            nn.Linear(latent_dimension, layer_3d),
            nn.BatchNorm1d(layer_3d),
            nn.ReLU(),
            nn.Dropout(0.2),  # 20% dropout
            nn.Linear(layer_3d, layer_2d),
            nn.BatchNorm1d(layer_2d),
            nn.ReLU(),
            nn.Dropout(0.2),  # 20% dropout
            nn.Linear(layer_2d, layer_1d),
            nn.BatchNorm1d(layer_1d),
            nn.ReLU()
        )
        self.final_layer = nn.Linear(layer_1d, out_dimension)

    def forward(self, z):
        """
        Pass throught the Decoder
        """
        # Get results of decoder network
        h2 = self.decode_nn(z)
        out = torch.sigmoid(self.final_layer(h2))
        return out



def compute_elbo(x, x_reconstructed, mus, log_vars, KLD_alpha):

    
    #recon_loss = nn.BCEWithLogitsLoss()(x_reconstructed, x)
    loss_function = nn.BCELoss()
    recon_loss = loss_function(x_reconstructed, x)

    kld = -0.5 * torch.mean(1. + log_vars - mus.pow(2) - log_vars.exp())

    return recon_loss + KLD_alpha * kld
   

def train_model(vae_encoder, vae_decoder,
                data_train, data_valid, num_epochs, batch_size,
                lr_enc, lr_dec, KLD_alpha,
                sample_num, sample_len, alphabet, type_of_encoding):
    """
    Train the Variational Auto-Encoder
    """

    print('num_epochs: ', num_epochs)
   
    # initialize an instance of the model
    optimizer_encoder = torch.optim.Adam(vae_encoder.parameters(), lr=lr_enc)
    optimizer_decoder = torch.optim.Adam(vae_decoder.parameters(), lr=lr_dec)

    # Create separate learning rate schedulers for each optimizer, 20% decrease every 100 epochs for both here
    scheduler_encoder = StepLR(optimizer_encoder, step_size=500, gamma=0.8)
    scheduler_decoder = StepLR(optimizer_decoder, step_size=500, gamma=0.8)

    data_train = data_train.clone().detach().to(device)
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

            #print("batch", batch[0],batch[1])

            # reshaping for efficient parallelization
            inp_flat_one_hot = batch.flatten(start_dim=1)
            latent_points, mus, log_vars = vae_encoder(inp_flat_one_hot)

            
            # decoding using the VAE Decoder 
            out_one_hot = vae_decoder(latent_points)

            # compute ELBO, item picks the scalar value 
            loss = compute_elbo(inp_flat_one_hot ,out_one_hot , mus, log_vars, KLD_alpha)

            # perform back propogation
            optimizer_encoder.zero_grad()
            optimizer_decoder.zero_grad()
            loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(vae_decoder.parameters(), 0.5)
            optimizer_encoder.step()
            optimizer_decoder.step()

            if batch_iteration % 30 == 0:
                end = time.time()

                
                report = 'Epoch: %d,  Batch: %d / %d,\t(loss: %.4f\t| ' \
                         'ELAPSED TIME: %.5f' \
                         % (epoch, batch_iteration, num_batches_train,
                            loss.item(), end - start)
                   
          
               

                print(report)
                
        
                start = time.time()


        # save model state at every 50th each epoch  
        if epoch % 50 == 0:
            print("models saved")
            save_models(vae_encoder, vae_decoder, epoch)
        #print quality report
        print(report)

        with open('results.dat', 'a') as content:
            content.write(report + '\n')
                 
        scheduler_encoder.step()
        scheduler_decoder.step()
    
        # Print current learning rates
        print("Epoch: {0}, Encoder LR: {1}, Decoder LR: {2}".format(epoch, scheduler_encoder.get_last_lr()[0],scheduler_decoder.get_last_lr()[0]))


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

    smiles_list = np.asanyarray(df.smiles)

    smiles_alphabet = list(set(''.join(smiles_list)))
    smiles_alphabet.append(' ')  # for padding

    largest_smiles_len = len(max(smiles_list, key=len))

    print('--> Translating SMILES to SELFIES...')
    selfies_list = list(map(sf.encoder, smiles_list))

    all_selfies_symbols = sf.get_alphabet_from_selfies(selfies_list)
    all_selfies_symbols.add('[nop]')
    selfies_alphabet = list(all_selfies_symbols)

    largest_selfies_len = max(sf.len_selfies(s) for s in selfies_list)

    print('Finished translating SMILES to SELFIES.')

    print("selfies alphabet:", selfies_alphabet)
    print("smiles alphabet:", smiles_alphabet)

    return selfies_list, selfies_alphabet, largest_selfies_len, \
           smiles_list, smiles_alphabet, largest_smiles_len



def preparefortraining():

    content = open('logfile.dat', 'w')
    content.close()
    content = open('results.dat', 'w')
    content.close()

    if os.path.exists("settings_symm.yml"):
        settings = yaml.safe_load(open("settings_symm.yml", "r"))
    else:
        print("Expected a file settings_symm.yml but didn't find it.")
        return

    print('--> Acquiring data...')
    type_of_encoding = settings['data']['type_of_encoding']
    file_name_smiles = settings['data']['smiles_file']

    print('Finished acquiring data.')

    if type_of_encoding == 0:
        print('Representation: SMILES')
        _, _, _, encoding_list, encoding_alphabet, largest_molecule_len = \
            get_selfie_and_smiles_encodings_for_dataset(file_name_smiles)

        print('--> Creating one-hot encoding...')
        data = multiple_smile_to_hot(encoding_list, largest_molecule_len,
                                     encoding_alphabet)
        print('Finished creating one-hot encoding.')

    elif type_of_encoding == 1:
        print('Representation: SELFIES')
        encoding_list, encoding_alphabet, largest_molecule_len, _, _, _ = \
            get_selfie_and_smiles_encodings_for_dataset(file_name_smiles)

        print('--> Creating one-hot encoding...')
        data = multiple_selfies_to_hot(encoding_list, largest_molecule_len,
                                       encoding_alphabet)
        print('Finished creating one-hot encoding.')

    else:
        print("type_of_encoding not in {0, 1}.")
        return

    len_max_molec = data.shape[1]
    len_alphabet = data.shape[2]
    len_max_mol_one_hot = len_max_molec * len_alphabet

    print(' ')
    print(f"Alphabet has {len_alphabet} letters, "
          f"largest molecule is {len_max_molec} letters.")


    data_parameters = settings['data']
    batch_size = data_parameters['batch_size']

    encoder_parameter = settings['encoder']
    decoder_parameter = settings['decoder']
    training_parameters = settings['training']

    return len_max_molec, len_alphabet, len_max_mol_one_hot, data_parameters, batch_size, encoder_parameter, decoder_parameter, training_parameters, data 





def main():
    content = open('logfile.dat', 'w')
    content.close()
    content = open('results.dat', 'w')
    content.close()

    if os.path.exists("settings_symm.yml"):
        settings = yaml.safe_load(open("settings_symm.yml", "r"))
    else:
        print("Expected a file settings_symm.yml but didn't find it.")
        return

    print('--> Acquiring data...')
    type_of_encoding = settings['data']['type_of_encoding']
    file_name_smiles = settings['data']['smiles_file']

    print('Finished acquiring data.')

    if type_of_encoding == 0:
        print('Representation: SMILES')
        _, _, _, encoding_list, encoding_alphabet, largest_molecule_len = \
            get_selfie_and_smiles_encodings_for_dataset(file_name_smiles)

        print('--> Creating one-hot encoding...')
        data = multiple_smile_to_hot(encoding_list, largest_molecule_len,
                                     encoding_alphabet)
        print('Finished creating one-hot encoding.')

    elif type_of_encoding == 1:
        print('Representation: SELFIES')
        encoding_list, encoding_alphabet, largest_molecule_len, _, _, _ = \
            get_selfie_and_smiles_encodings_for_dataset(file_name_smiles)

        print('--> Creating one-hot encoding...')
        data = multiple_selfies_to_hot(encoding_list, largest_molecule_len,
                                       encoding_alphabet)
        print('Finished creating one-hot encoding.')

    else:
        print("type_of_encoding not in {0, 1}.")
        return

    len_max_molec = data.shape[1]
    len_alphabet = data.shape[2]
    len_max_mol_one_hot = len_max_molec * len_alphabet

    print(' ')
    print(f"Alphabet has {len_alphabet} letters, "
          f"largest molecule is {len_max_molec} letters.")


    data_parameters = settings['data']
    batch_size = data_parameters['batch_size']

    encoder_parameter = settings['encoder']
    decoder_parameter = settings['decoder']
    training_parameters = settings['training']


    vae_encoder = VAEEncoder(in_dimension=len_max_mol_one_hot, **encoder_parameter).to(device)
    vae_decoder = VAEDecoder(out_dimension=len_max_mol_one_hot, **decoder_parameter).to(device)

    #return len_max_molec, len_alphabet, len_max_mol_one_hot, data_parameters, batch_size, encoder_parameter, decoder_parameter, training_parameters, data

    # initialize a random instance for the start
    #print("random start point de/coder")
    #vae_encoder = VAEEncoder(in_dimension=len_max_mol_one_hot, **encoder_parameter).to(device)
    #vae_decoder = VAEDecoder(**decoder_parameter,out_dimension=len(encoding_alphabet)).to(device)

    print('*' * 15, ': -->', device)

    data = torch.tensor(data, dtype=torch.float).to(device)

    train_valid_test_size = [0.9, 0.1, 0.0]
    data = data[torch.randperm(data.size()[0])]
    idx_train_val = int(len(data) * train_valid_test_size[0])
    idx_val_test = idx_train_val + int(len(data) * train_valid_test_size[1])

    data_train = data[0:idx_train_val]
    data_valid = data[idx_train_val:idx_val_test]

    print("start training")
    train_model(**training_parameters,
                vae_encoder=vae_encoder,
                vae_decoder=vae_decoder,
                batch_size=batch_size,
                data_train=data_train,
                data_valid=data_valid,
                alphabet=encoding_alphabet,
                type_of_encoding=type_of_encoding,
                sample_len=len_max_molec)

    with open('COMPLETED', 'w') as content:
        content.write('exit code: 0')
    
if __name__ == '__main__':
    try:
        main()
    except AttributeError:
        _, error_message, _ = sys.exc_info()
        print(error_message)

