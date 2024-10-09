import os
import pandas as pd
import torch
import yaml
import selfies as sf
import time

import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils.rnn import pad_sequence

import vae
from functions import selfies_to_lpoints, get_selfie_and_smiles_encodings_for_dataset
from functions_sub import stats, selfies_to_zs, gen_properties_tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
                        batch_norm: whether batch normalisation will be used. 1 for yes, any other int for no (int)'''


        input_dim = settings['model_params']['input_dim']
        hidden_dim = settings['model_params']['hidden_dim']
        prop_pred_activation = settings['model_params']['prop_pred_activation']
        prop_pred_dropout = settings['model_params']['prop_pred_dropout']
        prop_pred_depth = settings['model_params']['prop_pred_depth']
        prop_growth_factor = settings['model_params']['prop_growth_factor']
        batch_norm = settings['hyperparameters']['batch_norm']



        super(PropertyRegressionModel, self).__init__()

        self.activation = self._get_activation(prop_pred_activation)
        self.dropout = nn.Dropout(prop_pred_dropout)
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim)) ### initial layer

        
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

            

        self.reg_prop_pred = nn.Linear(hidden_dims[len(hidden_dims)-1], 1)  # For regression tasks, single output node


    def forward(self, x):

        '''Forward pass through the MLP'''

        '''Arguments:
                        x: transformed latent vectors (Pytorch float tensor)'''

        '''Outputs:
                        reg_prop_out: the predicted property output (Pytorch floattensor)'''


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
        elif activation_name == 'leaky_relu':#'sigmoid':
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
                    len_alphabet: the length of the alphabet used (int)
                    len_max_molec_onehot: len_alphabet * largest_molecule_len (int)
                    encoding_alphabet: the alphabet generated by the function get_selfie_and_smiles_encodings_for_dataset (list)
                    encoding_list: encoding_list: a list containing the SELFIES (list)
                    vae_encoder: the encoder object (VAEEncoder object)
                    properties_tensor: the properties being used for prediction (Pytorch float.32 tensor)'''
    
    smiles_file = settings['settings']['smiles_file']
    vae_file = settings['settings']['vae_file']
    vae_epoch = settings['settings']['vae_epoch']
    vae_settings = yaml.safe_load(open(str(vae_file) + "settings/" + "settings.yml", "r"))
    encoding_alphabet = vae_settings['alphabet']
    torch_seed = vae_settings['data']['torch_seed']
    encoder_parameter = vae_settings['encoder']
    torch_seed = vae_settings['data']['torch_seed']
    vae_weights_path = str(vae_file) + str(vae_epoch) + "/E.pt"

    new_constraints = sf.get_semantic_constraints()
    new_constraints['N'] = 5
    new_constraints['B'] = 4
    sf.set_semantic_constraints(new_constraints)
    encoding_list, _, _ = get_selfie_and_smiles_encodings_for_dataset(smiles_file)

    torch.manual_seed(torch_seed)
    rand_perms = torch.randperm(len(encoding_list))
    encoding_list = [encoding_list[x] for x in rand_perms]
    model_weights = torch.load(vae_weights_path, map_location = device)
    len_max_molec_onehot = model_weights['encode_RNN.weight_ih_l0'].shape[1]
    largest_molecule_len = int(len_max_molec_onehot/len(encoding_alphabet))
    vae_encoder = vae.VAEEncoder(in_dimension=len_max_molec_onehot, **encoder_parameter).to(device)
    vae_encoder.load_state_dict(model_weights)
    vae_encoder.eval()

    my_file = pd.read_csv(smiles_file, index_col=None)
    my_file = my_file.dropna()
    properties_tensor = gen_properties_tensor(my_file)
    properties_tensor = properties_tensor[rand_perms]

    mask1 = torch.tensor(properties_tensor == 0)
    mask2 = ~mask1.squeeze()
    mask2_index = torch.nonzero(mask2).squeeze()
    encoding_list = [encoding_list[x] for x in mask2_index]
    properties_tensor = properties_tensor[mask2]
    mask3 = (torch.tensor(properties_tensor.isnan()==True)).squeeze()
    mask4 = ~mask3.squeeze()
    mask4_index = torch.nonzero(mask4).squeeze()
    encoding_list = [encoding_list[x] for x in mask4_index]
    properties_tensor = properties_tensor[mask4]
    
    len_alphabet = len(encoding_alphabet)


    return largest_molecule_len, len_alphabet, len_max_molec_onehot, encoding_alphabet, encoding_list, vae_encoder, properties_tensor

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

def train_model(encoding_list, encoding_alphabet, len_max_molec, vae_encoder, properties_tensor, optimizer, model, loss_function, scheduler, settings, epoch_full):

    '''Train the multi-layered perceptron'''

    '''Arguments:  
                    encoding_list: a list containing the SELFIES (list)
                    encoding_alphabet: the alphabet generated by the function get_selfie_and_smiles_encodings_for_dataset (list)
                    len_max_molec: the maximum length of the molecule encodings (int)
                    vae_encoder: vae_encoder: the encoder object (VAEEncoder object)
                    properties_tensor: the properties being used for prediction (Pytorch float.32 tensor): 
                    optimizer: the optimizer used to modify the weights after a back propagation (Pytorch torch.optim object)
                    model: the multi-layered perceptron object (PropertyRegressionModel object)
                    loss_function: the loss function being used, e.g., MSELoss (str)
                    scheduler: function used to modify the learning rate (torch.optim.lr_scheduler object)
                    settings: the settings defined by the .yml file (dict)
                    epoch_full: a list of integers corresponding to the epochs that the train_model function will run through (list)'''


    batch_size = settings['hyperparameters']['batch_size']
    z_det = settings['settings']['z']
    r2t = []
    r2v = []
    epoch_list = []

    idx_train_val = int(len(encoding_list)*0.8)
    idx_val_test = idx_train_val + int(len(encoding_list) * 0.2)

    train_properties_tensor = properties_tensor[0:idx_train_val]
    valid_properties_tensor = properties_tensor[idx_train_val:idx_val_test]
    data_train = encoding_list[0:idx_train_val]
    data_valid = encoding_list[idx_train_val:idx_val_test]


    num_clusters_train = 20
    num_clusters_valid = 5

    cluster_train_size = int(len(data_train) / num_clusters_train)
    cluster_valid_size = int(len(data_valid) / num_clusters_valid)
    if cluster_train_size < (len(data_train) / num_clusters_train):
        cluster_train_size+=1
    if cluster_valid_size < (len(data_valid) / cluster_valid_size):
        cluster_valid_size+=1 


    model.train()

    if z_det == 0:
        print('mu mode')
        lpoint_function = selfies_to_lpoints
    else:
        print('z mode')
        lpoint_function = selfies_to_zs

    for epoch in epoch_full:
        
        time0 = time.time()
        sub_r2t = []
        sub_r2v = []
        epoch_list.append(epoch)

        for cluster_t in range(num_clusters_train):

            start_idx = cluster_t * cluster_train_size
            stop_idx = (cluster_t + 1) * cluster_train_size
            sub_train = data_train[start_idx: stop_idx]
            sub_prop = train_properties_tensor[start_idx: stop_idx]
            with torch.no_grad():
                mu = lpoint_function(sub_train, encoding_alphabet, len_max_molec, vae_encoder)

            num_batches_train = int(len(sub_train) / batch_size)
            if num_batches_train < (len(sub_train)/batch_size):
                num_batches_train+=1

            for batch_iteration in range(num_batches_train):

                start_idx = batch_iteration * batch_size
                stop_idx = (batch_iteration + 1) * batch_size
                batch_mu = mu[start_idx: stop_idx].to(device)
                batch_props = sub_prop[start_idx: stop_idx].to(device)

                optimizer.zero_grad()
                predictions = model(batch_mu)
                loss = loss_function(predictions, batch_props)
                loss.backward()
                optimizer.step()

                y_test = batch_props.detach().to('cpu')
                y_pred = predictions.detach().to('cpu')

                _, _, _, r2_train = stats(y_test, y_pred)
                sub_r2t.append(r2_train)

  

        for cluster_v in range(num_clusters_valid):

            start_idx = cluster_v * cluster_valid_size
            stop_idx = (cluster_v + 1) * cluster_valid_size
            sub_valid = data_valid[start_idx: stop_idx]
            sub_prop = valid_properties_tensor[start_idx: stop_idx]

            with torch.no_grad():
                mu = lpoint_function(sub_valid, encoding_alphabet, len_max_molec, vae_encoder)
            num_batches_valid = int(len(sub_valid) / batch_size)
            if num_batches_valid < (len(sub_valid)/batch_size):
                num_batches_valid+=1


            for batch_iteration in range(num_batches_valid):

                start_idx = batch_iteration * batch_size
                stop_idx = (batch_iteration + 1) * batch_size
                batch_mu = mu[start_idx: stop_idx].to(device)
                batch_props = sub_prop[start_idx: stop_idx].to(device)

                model.eval()
                with torch.no_grad():
                    predictions = model(batch_mu)
                model.train()

                y_test = batch_props.detach().to('cpu')
                y_pred = predictions.detach().to('cpu')

                _, _, _, r2_valid = stats(y_test, y_pred)

                sub_r2v.append(r2_valid)


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
    epoch_list = list(range(num_epochs))

    len_max_molec, _, _, encoding_alphabet, encoding_list, vae_encoder, properties_tensor = data_init(settings)
    learning_rate = settings['hyperparameters']['lr']
    weight_choice = settings['hyperparameters']['weight_choice']
    learning_rate_factor = settings['hyperparameters']['learning_rate_factor']
    learning_rate_patience = settings['hyperparameters']['learning_rate_patience']

    model = PropertyRegressionModel(settings).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)#, weight_decay=weight_choice)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=learning_rate_factor, patience=learning_rate_patience, verbose=True)
    

    loss_choice = settings['hyperparameters']['loss_choice']
    if loss_choice == 1:
        loss_function = nn.L1Loss()
    if loss_choice == 2:
        loss_function = nn.HuberLoss()
    else:
        loss_function = nn.MSELoss()


    train_model(encoding_list,
                    encoding_alphabet,
                    len_max_molec,
                    vae_encoder,
                    properties_tensor,
                    optimizer, 
                    model, 
                    loss_function,
                    scheduler, 
                    settings,
                    epoch_list)
