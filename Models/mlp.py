#!/usr/bin/env python3

import os
import sys
import pandas as pd
import torch
import yaml
import selfies as sf
import time
import numpy as np

import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from rdkit import Chem

import vae
from functions_sub import stats
from function_fingerprints import gen_fingerprints

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from rdkit import RDLogger 
RDLogger.DisableLog('rdApp.*') 

class PropertyRegressionModel(nn.Module):
    def __init__(self, settings):

        '''Multi layer perceptron'''

        '''Arguments:   --- SETTINGS ---
                        input_dim: the latent space dimension (int)
                        hidden_dim: the size of the first hidden layer (int)
                        prop_pred_activation: the activation function used (str)
                        prop_pred_dropout: the dropout coefficient (float)
                        prop_pred_depth: the number of hidden layers (int)
                        prop_growth_factor: the coefficient each hidden layer number is multiplied by. E.g., hidden = 256, prop_growth_factor = 0.5, second layer = 128 (float)
                        batch_norm: whether batch normalisation will be used. 1 for yes, any other int for no (int)
                        fp_size: size of the daylight and morgan fingerprints (int)'''


        input_dim = settings['model_params']['input_dim']
        hidden_dim = settings['model_params']['hidden_dim']
        prop_pred_activation = settings['model_params']['prop_pred_activation']
        prop_pred_dropout = settings['model_params']['prop_pred_dropout']
        prop_pred_depth = settings['model_params']['prop_pred_depth']
        prop_growth_factor = settings['model_params']['prop_growth_factor']
        batch_norm = settings['hyperparameters']['batch_norm']
        output_dim = int(2 * settings['settings']['num_props'])
        FP_size = settings['model_params']['fp_size']




        super(PropertyRegressionModel, self).__init__()

        self.activation = self._get_activation(prop_pred_activation)
        self.dropout = nn.Dropout(prop_pred_dropout)
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.input_dim = input_dim


        self.linear1 = nn.Linear(300, input_dim)
        self.linear2 = nn.Linear(FP_size, input_dim)
        self.linear3 = nn.Linear(FP_size, input_dim)
        self.linear4 = nn.Linear(64, input_dim)
        self.linear5 = nn.Linear(input_dim*4, input_dim)

        
        hidden_dims = []
        hidden_dims.append(hidden_dim)
        for p_i in range(1, prop_pred_depth):
            hidden_dims.append(int(prop_growth_factor * hidden_dims[p_i - 1]))
            Ln_layer = nn.Linear(hidden_dims[p_i - 1], hidden_dims[p_i])
            self.layers.append(Ln_layer)

            if batch_norm == 1:
                BatchNorm_layer = nn.BatchNorm1d(hidden_dims[p_i])
                self.layers.append(BatchNorm_layer)

            self.layers.append(self.activation)

            if prop_pred_dropout > 0:
                self.layers.append(nn.Dropout(prop_pred_dropout))

            

        self.reg_prop_pred = nn.Linear(hidden_dims[len(hidden_dims)-1], output_dim) 


    def forward(self, x1, x2, x3, x4):

        '''Forward pass through the MLP'''

        '''Arguments:
                        x: transformed latent vectors (Pytorch float tensor)'''

        '''Outputs:
                        reg_prop_out: the predicted property output (Pytorch floattensor)'''


        x1 = self.activation(self.linear1(x1))  
        x2 = self.activation(self.linear2(x2)) 
        x3 = self.activation(self.linear3(x3))
        x4 = self.activation(self.linear4(x4))
        x = torch.stack([x1, x2, x3, x4], dim=1).flatten(1)
        x = self.activation(self.linear5(x))


        for layer in self.layers:
            x = layer(x)

        reg_prop_out = self.reg_prop_pred(x)
        return reg_prop_out

    def _get_activation(self, activation_name):

        '''Gives you the activation layer'''

        '''Arguments:
                        activation_name: the name of the activation functions shown below (str)'''
        
        if activation_name == 'silu':
            return nn.SiLU()
        elif activation_name == 'leaky_relu':
            return nn.LeakyReLU()
        elif activation_name == 'tanh':
            return nn.Tanh()
        elif activation_name == 'leaky_relu':
            return nn.LeakyRelU()
        elif activation_name == 'gelu':
            return nn.GELU()
        elif activation_name == 'elu':
            return nn.ELU()
        else:
            raise ValueError(f"Unknown activation function: {activation_name}")




def save_params(epoch, model, settings):

    '''Save the model object and also the current parameters defining the model'''

    '''Arguments:
                    epoch: the epoch currently being saved (int)
                    model: the multi-layered perceptron object (PropertyRegressionModel object)
                    settings: the settings defined by the .yml file (dict)'''



    save_path = settings['settings']['output_folder']
    out_dir = str(save_path)
    out_dir_epoch = out_dir + '/{}'.format(epoch)
    os.makedirs(out_dir_epoch)
    
    torch.save(model.state_dict(), '{}/model.pt'.format(out_dir_epoch))

    settings_folder = out_dir + '/settings'
    log_folder = settings_folder
    log_filename = 'settings.yml'

    if not os.path.exists(settings_folder):
        os.makedirs(settings_folder)

        log_filepath = os.path.join(log_folder, log_filename)
        data = {**settings}

        with open(log_filepath, 'w') as file:
            yaml.dump(data, file)

        


    
def data_init(settings):

    '''Data initialisation'''

    '''Arguments:
                    settings: settings defined by the corresponding .yml file (dict)'''
    
    '''Outputs:
                    largest_molecule_len: the maximum length of the molecule encodings (int)
                    encoding_alphabet: the alphabet generated by the function get_selfie_and_smiles_encodings_for_dataset (list)
                    encoding_list: encoding_list: a list containing the SELFIES (list)
                    vae_encoder: the encoder object (VAEEncoder object)
                    properties_tensor: the properties being used for prediction (Pytorch float.32 tensor)
                    smiles_list: the list of SMILES encodings (list)'''
    
    smiles_file = settings['settings']['smiles_file']
    vae_file = settings['settings']['vae_file']
    vae_epoch = settings['settings']['vae_epoch']
    num_props = settings['settings']['num_props']
    vae_settings = yaml.safe_load(open(str(vae_file) + "settings/" + "settings.yml", "r"))
    encoding_alphabet = vae_settings['alphabet']
    encoder_parameter = vae_settings['encoder']
    vae_weights_path = str(vae_file) + str(vae_epoch) + "/E.pt"
    properties_list = []

    perm_df = pd.read_csv(str(vae_file) + "settings/PERM_IDX.csv")
    rand_perms = perm_df['PERM_IDX'].tolist()

    new_constraints = sf.get_semantic_constraints()
    new_constraints['N'] = 5
    new_constraints['B'] = 4
    sf.set_semantic_constraints(new_constraints)
    encoding_list, _, _ = vae.get_selfie_and_smiles_encodings_for_dataset(smiles_file)

    encoding_list = [encoding_list[x] for x in rand_perms]
    model_weights = torch.load(vae_weights_path, map_location = device)
    len_max_molec_onehot = model_weights['encode_RNN.weight_ih_l0'].shape[1]
    largest_molecule_len = int(len_max_molec_onehot/len(encoding_alphabet))
    vae_encoder = vae.VAEEncoder(in_dimension=len_max_molec_onehot, **encoder_parameter).to(device)
    vae_encoder.load_state_dict(model_weights)
    vae_encoder.eval()

    df = pd.read_csv(smiles_file)
    smiles_list = df['smiles'].tolist()
    smiles_list = [smiles_list[x] for x in rand_perms]

    for x in range(num_props):
        dE = 'TransitionEnergies' + str(x+1)
        OS = 'OscillatorStrength' + str(x+1)

        dE_list = df[dE].tolist()
        OS_list = df[OS].tolist()
        dE_list = [dE_list[x] for x in rand_perms]
        OS_list = [OS_list[x] for x in rand_perms]
        OS_list = [np.log10(dE_list[i]*OS_list[i]) for i in range(len(OS_list))]

        properties_list.append(dE_list)
        properties_list.append(OS_list)

    properties_tensor = torch.tensor(properties_list).t()

    mask = torch.cat([~torch.any(properties_tensor==0,dim=1)]).squeeze()
    valid_mask = torch.nonzero(mask).squeeze()
    encoding_list = [encoding_list[x] for x in valid_mask]
    smiles_list = [smiles_list[x] for x in valid_mask]
    properties_tensor = properties_tensor[valid_mask]

    mask = torch.cat([~torch.any(properties_tensor.isnan(),dim=1)]).squeeze()
    valid_mask = torch.nonzero(mask).squeeze()
    encoding_list = [encoding_list[x] for x in valid_mask]
    smiles_list = [smiles_list[x] for x in valid_mask]
    properties_tensor = properties_tensor[valid_mask]

    mask = torch.cat([~torch.any(properties_tensor.isinf(),dim=1)]).squeeze()
    valid_mask = torch.nonzero(mask).squeeze()
    encoding_list = [encoding_list[x] for x in valid_mask]
    smiles_list = [smiles_list[x] for x in valid_mask]
    properties_tensor = properties_tensor[valid_mask]
    
    mols = [Chem.MolFromSmiles(x) for x in smiles_list]
    valid_list = []

    for x in range(len(mols)):
        if mols[x] is not None:
            valid_list.append(x)

    mols = [mols[x] for x in valid_list]
    smiles_list = [smiles_list[x] for x in valid_list]
    encoding_list = [encoding_list[x] for x in valid_list]
    properties_tensor = properties_tensor[valid_list]


    return largest_molecule_len, encoding_alphabet, encoding_list, vae_encoder, properties_tensor, smiles_list

def save_r2_loss(epoch, loss, r2, train_r2, settings):

    '''This function saves the epoch, total training loss, trainin reconstruction loss, training kld loss and the total validation loss to a .txt file'''

    '''Arguments:
                    epoch: the epoch currently being saved (int)
                    r2: the r squared value of the validation set (float)
                    train_r2: the r squared value of the training set (float)
                    loss: the current loss of the model (float)
                    settings: settings defined by the .yml file (dict)'''

    out_dir = settings['settings']['output_folder']
    log_folder = out_dir + '/settings'
    log_filename = 'r2_loss.txt'
    log_filepath = os.path.join(log_folder, log_filename)

    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    file_exists = os.path.isfile(log_filepath)

    with open(log_filepath, 'a') as file:
        if not file_exists:
            file.write("epoch,loss,val_r2,train_r2\n")
        file.write(f'{epoch},{loss},{r2},{train_r2}\n')

def train_model(encoding_list, lpoints, morgan_fp_tensor, daylight_fingerprints_tensor, mol2vec_tensor, properties_tensor, optimizer, model, loss_function, scheduler, settings, epoch_full):

    '''Train the multi-layered perceptron'''

    '''Arguments:  
                    encoding_list: a list containing the SELFIES (list)
                    lpoints: the mean latent vectors (Pytorch float.32 tensor)
                    morgan_fp_tensor: pytorch tensor containing morgan fingerprints (Pytorch float.32 tensor)
                    daylight_fingerprints_tensor: pytorch tensor containing daylight fingerprints (Pytorch float.32 tensor)
                    mol2vec_tensor: pytorch tensor containing mol2vec fingerprints (Pytorch float.32 tensor)
                    properties_tensor: the properties being used for prediction (Pytorch float.32 tensor): 
                    optimizer: the optimizer used to modify the weights after a back propagation (Pytorch torch.optim object)
                    model: the multi-layered perceptron object (PropertyRegressionModel object)
                    loss_function: the loss function being used, e.g., MSELoss (str)
                    scheduler: function used to modify the learning rate (torch.optim.lr_scheduler object)
                    settings: the settings defined by the .yml file (dict)
                    epoch_full: a list of integers corresponding to the epochs that the train_model function will run through (list)'''


    batch_size = settings['hyperparameters']['batch_size']
    r2t = []
    r2v = []

    idx_train_val = int(len(encoding_list)*0.8)
    idx_val_test = idx_train_val + int(len(encoding_list) * 0.2)

    train_properties_tensor = properties_tensor[0:idx_train_val]
    valid_properties_tensor = properties_tensor[idx_train_val:idx_val_test]
    data_train = encoding_list[0:idx_train_val]
    data_valid = encoding_list[idx_train_val:idx_val_test]
    
    num_batches_train = int(len(data_train) / batch_size)
    if num_batches_train*batch_size < len(data_train):
        num_batches_train = num_batches_train +1
    num_batches_valid = int(len(data_valid) / batch_size)
    if num_batches_valid*batch_size < len(data_valid):
        num_batches_valid = num_batches_valid +1

    lpoints_train = lpoints[0:idx_train_val]
    lpoints_valid = lpoints[idx_train_val:idx_val_test]
    del lpoints
    morgan_train = morgan_fp_tensor[0:idx_train_val]
    morgan_valid = morgan_fp_tensor[idx_train_val:idx_val_test]
    del morgan_fp_tensor
    daylight_train = daylight_fingerprints_tensor[0:idx_train_val]
    daylight_valid = daylight_fingerprints_tensor[idx_train_val:idx_val_test]
    del daylight_fingerprints_tensor
    mol2vec_train = mol2vec_tensor[0:idx_train_val]
    mol2vec_valid = mol2vec_tensor[idx_train_val:idx_val_test]
    del mol2vec_tensor

    
    time0 = time.time()    
    model.train()


    for epoch in epoch_full:
        sub_r2t = []
        sub_r2v = []

        for batch_iteration in range(num_batches_train):

            start_idx = batch_iteration * batch_size
            stop_idx = (batch_iteration + 1) * batch_size

            batch_lpoint = lpoints_train[start_idx: stop_idx].to(torch.float32).to(device)
            batch_mfp = morgan_train[start_idx: stop_idx].to(torch.float32).to(device)
            batch_dfp = daylight_train[start_idx: stop_idx].to(torch.float32).to(device)
            batch_m2v = mol2vec_train[start_idx: stop_idx].to(torch.float32).to(device)
            batch_props = train_properties_tensor[start_idx: stop_idx].to(torch.float32).to(device)

            optimizer.zero_grad()
            predictions = model(batch_m2v, batch_mfp, batch_dfp, batch_lpoint)

            loss = loss_function(predictions, batch_props)
            loss.backward()
            optimizer.step()

            y_test = batch_props.detach().to('cpu')
            y_test[:, 1::2] = (10**(y_test[:, 1::2])) / y_test[:, ::2]
            y_pred = predictions.detach().to('cpu')
            y_pred[:, 1::2] = (10**(y_pred[:, 1::2])) / y_pred[:, ::2]

            sub_sub_r2 = []
            for x in range(y_test.shape[1]):

                y_test_current = y_test[:, x]
                y_pred_current = y_pred[:, x]

                _, _, _, r2_train = stats(y_test_current, y_pred_current)
                sub_sub_r2.append(r2_train.unsqueeze(0))

            r2_tensor = torch.cat(sub_sub_r2)
            sub_r2t.append(r2_tensor)


  
        for batch_iteration in range(num_batches_valid):

            start_idx = batch_iteration * batch_size
            stop_idx = (batch_iteration + 1) * batch_size

            batch_lpoint = lpoints_valid[start_idx: stop_idx].to(torch.float32).to(device)
            batch_mfp = morgan_valid[start_idx: stop_idx].to(torch.float32).to(device)
            batch_dfp = daylight_valid[start_idx: stop_idx].to(torch.float32).to(device)
            batch_m2v = mol2vec_valid[start_idx: stop_idx].to(torch.float32).to(device)
            batch_props = valid_properties_tensor[start_idx: stop_idx].to(torch.float32).to(device)

            model.eval()
            with torch.no_grad():
                predictions = model(batch_m2v, batch_mfp, batch_dfp, batch_lpoint)
            model.train()

            y_test = batch_props.detach().to('cpu')
            y_test[:, 1::2] = (10**(y_test[:, 1::2])) / y_test[:, ::2]
            y_pred = predictions.detach().to('cpu')
            y_pred[:, 1::2] = (10**(y_pred[:, 1::2])) / y_pred[:, ::2]

            sub_sub_r2 = []
            for x in range(y_test.shape[1]):

                y_test_current = y_test[:, x]
                y_pred_current = y_pred[:, x]

                _, _, _, r2_valid = stats(y_test_current, y_pred_current)
                sub_sub_r2.append(r2_valid.unsqueeze(0))
            r2_tensor = torch.cat(sub_sub_r2)
            sub_r2v.append(r2_tensor)

        


        scheduler.step(loss)

        avgr2_t = sum(sub_r2t)/len(sub_r2t)
        avgr2_v = sum(sub_r2v)/len(sub_r2v)

        print('Epoch:', epoch, 'complete', 'Training r2:', avgr2_t, 'Validation r2:', avgr2_v, 'Time taken for epoch:', time.time() - time0)
        time0 = time.time()

        r2t.append(avgr2_t)
        r2v.append(avgr2_v)

        if epoch % 10 ==0:
            save_params(epoch, model, settings)
        save_r2_loss(epoch, loss, avgr2_v, avgr2_t, settings)


def main():

    if os.path.exists("mlp_settings.yml"):
        settings = yaml.safe_load(open("mlp_settings.yml", "r"))
    else:
        print("Expected a file settings.yml but didn't find it.")
        return
    
    num_epochs = settings['hyperparameters']['epochs']
    FP_size = settings['model_params']['fp_size']
    learning_rate = settings['hyperparameters']['lr']
    weight_choice = settings['hyperparameters']['weight_choice']
    learning_rate_factor = settings['hyperparameters']['learning_rate_factor']
    learning_rate_patience = settings['hyperparameters']['learning_rate_patience']
    loss_choice = settings['hyperparameters']['loss_choice']

    epoch_list = list(range(num_epochs))

    len_max_molec, encoding_alphabet, encoding_list, vae_encoder, properties_tensor, smiles_list = data_init(settings)
    lpoints, morgan_fp_tensor, daylight_fingerprints_tensor, mol2vec_tensor = gen_fingerprints(smiles_list, encoding_list, vae_encoder, encoding_alphabet, len_max_molec, FP_size)

    model = PropertyRegressionModel(settings).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=learning_rate_factor, patience=learning_rate_patience, verbose=True)
    

    if loss_choice == 1:
        loss_function = nn.L1Loss()
    if loss_choice == 2:
        loss_function = nn.HuberLoss()
    else:
        loss_function = nn.MSELoss()


    train_model(encoding_list, 
                lpoints, 
                morgan_fp_tensor, 
                daylight_fingerprints_tensor, 
                mol2vec_tensor, 
                properties_tensor, 
                optimizer, 
                model, 
                loss_function, 
                scheduler, 
                settings, 
                epoch_list)

if __name__ == '__main__':
    try:
        main()
    except AttributeError:
        _, error_message, _ = sys.exc_info()
        print(error_message)