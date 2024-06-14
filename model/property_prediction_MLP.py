import os
import pandas as pd
import torch
import yaml
import selfies as sf

import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils.rnn import pad_sequence

import vae
from functions import get_selfie_and_smiles_encodings_for_dataset, remove_unrecognized_symbols, selfies_to_lpoints, get_free_memory, stats, gen_properties_tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    

class PropertyRegressionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, prop_pred_activation, prop_pred_dropout, prop_pred_depth, prop_growth_factor, batch_norm):

        '''Multi layer perceptron'''

        '''Arguments:
                        input_dim: the latent space dimension (int)
                        hidden: the size of the first hidden layer (int)
                        prop_pred_activation: the activation function used (str)
                        prop_pred_dropout: the dropout coefficient (float)
                        prop_pred_depth: the number of hidden layers (int)
                        prop_growth_factor: the coefficient each hidden layer number is multiplied by. E.g., hidden = 256, prop_growth_factor = 0.5, second layer = 128 (float)'''
        

        super(PropertyRegressionModel, self).__init__()

        self.activation = self._get_activation(prop_pred_activation)
        self.dropout = nn.Dropout(prop_pred_dropout)
        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(input_dim, hidden_dim))

        
        hidden_dims = []
        hidden_dims.append(hidden_dim)
        for p_i in range(1, prop_pred_depth):
            hidden_dims.append(int(prop_growth_factor * hidden_dims[p_i - 1]))
            Ln_layer = nn.Linear(hidden_dims[p_i - 1], hidden_dims[p_i])
            self.layers.append(Ln_layer)


            if batch_norm ==1:
                BatchNorm_layer = nn.BatchNorm1d(hidden_dims[p_i])
                self.layers.append(BatchNorm_layer)

            self.layers.append(self.activation)

            if prop_pred_dropout > 0:
                self.layers.append(nn.Dropout(prop_pred_dropout))

            

        self.reg_prop_pred = nn.Linear(hidden_dims[len(hidden_dims)-1], 1)  


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
        elif activation_name == 'sigmoid':
            return nn.Sigmoid()
        elif activation_name == 'tanh':
            return nn.Tanh()
        elif activation_name == 'leaky_relu':
            return nn.LeakyReLU()
        elif activation_name == 'elu':
            return nn.ELU()
        else:
            raise ValueError(f"Unknown activation function: {activation_name}")




def save_params(epoch, model, settings):

    '''Save the model object and also the current parameters defining the model'''

    '''Arguments:
                    epoch: the epoch currently being saved (int)
                    model: the mlp object (PropertyRegressionModel object)
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


def dump_params(epoch, model, settings):

    '''Save the model object and also the current parameters defining the model'''

    '''Arguments:
                    epoch: the epoch currently being saved (int)
                    model: the mlp object (PropertyRegressionModel object)
                    settings: the settings defined by the .yml file (dict)'''



    save_path = settings['settings']['output_folder']

    out_dir = str(save_path)
    out_dir_epoch = out_dir + '/Dump/' + '/{}'.format(epoch)
    os.makedirs(out_dir_epoch)
    
    torch.save(model.state_dict(), '{}/model.pt'.format(out_dir_epoch))
    torch.save(model, '{}/full_model.pt'.format(out_dir_epoch))

    settings_folder = out_dir + '/settings'

    log_folder = settings_folder
    log_filename = 'settings.yml'

    if not os.path.exists(settings_folder):
        os.makedirs(settings_folder)

        log_filepath = os.path.join(log_folder, log_filename)
        data = {**settings}

        with open(log_filepath, 'w') as file:
            yaml.dump(data, file)


def save_r2_loss(epoch, r2, train_r2, loss, mse_valid, mae_valid, mre_valid, bottom_mre, settings):

    '''This function saves the epoch, total training loss, trainin reconstruction loss, training kld loss and the total validation loss to a .txt file'''

    '''Arguments:
                    epoch: the epoch currently being saved (int)
                    r2: the r squared value of the validation set (float)
                    train_r2: the r squared value of the training set (float)
                    loss: the current loss of the model (float)
                    mse_valid: mse of the validation set (float)
                    mae_valid: mae of the validation set (float)
                    mre_valid: mre of the validation set (float)
                    bottom_mre: average mre of the lowest 5 properties in the validation set (float)
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
            file.write("epoch,loss,val_r2,train_r2,mse_valid,mae_valid,mre_valid,bottom_mre\n")
        file.write(f'{epoch},{loss},{r2},{train_r2},{mse_valid},{mae_valid},{mre_valid},{bottom_mre}\n')

    
def data_init(settings):

    '''Data initialisation'''

    '''Arguments:
                    settings: settings defined by the corresponding .yml file (dict)'''
    
    '''Outputs:    
                    largest_molecule_len: the maximum length of the molecule encodings (int)
                    len_alphabet: the length of the alphabet used (int)
                    len_max_mol_one_hot: the length of the maximum one hot representation (int)
                    selfies_alphabet: the alphabet used (list)
                    encoding_list: a list containing the SELFIES (list)
                    vae_encoder: the encoder object (VAEEncoder object) 
                    properties_tensor: the properties being used for prediction (Pytorch float.32 tensor)'''
    
    smiles_file = settings['settings']['smiles_file']
    vae_file = settings['settings']['vae_file']
    vae_epoch = settings['settings']['vae_epoch']
    tail_side = settings['settings']['tail_side']

    vae_settings = yaml.safe_load(open(str(vae_file) + "settings/" + "settings.yml", "r"))
    selfies_alphabet = vae_settings['alphabet']
    torch_seed = vae_settings['data']['torch_seed']

    encoder_parameter = vae_settings['encoder']
    selfies_alphabet = vae_settings['alphabet']
    torch_seed = vae_settings['data']['torch_seed']
    vae_weights_path = str(vae_file) + str(vae_epoch) + "/E.pt"

    new_constraints = sf.get_semantic_constraints()
    new_constraints['N'] = 5
    new_constraints['B'] = 4
    sf.set_semantic_constraints(new_constraints)

    

    encoding_list, _, _, = get_selfie_and_smiles_encodings_for_dataset(smiles_file)
    #selfies_alphabet.append('.')


    torch.manual_seed(torch_seed)
    rand_perms = torch.randperm(len(encoding_list))
    encoding_list = [encoding_list[x] for x in rand_perms]

    model_weights = torch.load(vae_weights_path, map_location = device)
    vae_encoder = vae.VAEEncoder(in_dimension=((model_weights['encode_RNN.weight_ih_l0'].shape[1])), **encoder_parameter).to(device)
    vae_encoder.load_state_dict(model_weights)

    my_file = pd.read_csv(smiles_file, index_col=None)

    properties_tensor = gen_properties_tensor(my_file)
    properties_tensor = properties_tensor[rand_perms]

    nan_tensor = (properties_tensor.squeeze().isnan() == False).nonzero().squeeze().to(torch.long)
    print('nan tensor type:', nan_tensor.dtype)
    encoding_list = [encoding_list[x] for x in nan_tensor]
    properties_tensor = properties_tensor[nan_tensor]

    largest_molecule_len = int((model_weights['encode_RNN.weight_ih_l0'].shape[1])/len(selfies_alphabet))

    val_part = int(0.8 * len(properties_tensor))
    prop_clone = properties_tensor[val_part:].squeeze().clone().detach()

    if tail_side == 0: ### If tail_side is 0 then we will investigate the lowest x molecules
        _, indices = prop_clone.sort()
    else: ### Else, we will look at the top x molecules, e.g., the molecules with the highest HOMO-LUMO gaps
        _, indices = prop_clone.sort(descending=True)
    bottom_indices = indices[:128]

    bottom_encoding = [encoding_list[val_part+x] for x in bottom_indices]
    bottom_props = properties_tensor[val_part+bottom_indices].to(device)


    return largest_molecule_len, len(selfies_alphabet), largest_molecule_len*len(selfies_alphabet), selfies_alphabet, encoding_list, vae_encoder, properties_tensor, bottom_encoding, bottom_props

def train_model(vae_encoder, encoding_list, properties_tensor, selfies_alphabet, len_max_molec, optimizer, model, loss_function, scheduler, bottom_encoding, bottom_props, settings):

    '''Train the multi-layered perceptron'''

    '''Arguments:  
                    vae_encoder: vae_encoder: the encoder object (VAEEncoder object)
                    encoding_list: a list containing the SELFIES (list)
                    properties_tensor: the properties being used for prediction (Pytorch float.32 tensor): 
                    selfies_alphabet: the alphabet used (list)
                    len_max_molec: the maximum length of the molecule encodings (int)
                    optimizer: the optimizer used to modify the weights after a back propagation (Pytorch torch.optim object)
                    model: the multi-layered perceptron object (PropertyRegressionModel object)
                    loss_function: the loss function being used, e.g., MSELoss (str)
                    scheduler: function used to modify the learning rate (torch.optim.lr_scheduler object)
                    settings: the settings defined by the .yml file (dict)'''
    

    batch_size = settings['hyperparameters']['batch_size']
    epochs = settings['hyperparameters']['epochs']
    lpoint_size = settings['model_params']['input_dim']

    num_chunks = 50
    chunk_size = int(len(encoding_list)/num_chunks)    


    total_mem_req = 8*(len(encoding_list)*lpoint_size) + 4*(len(encoding_list))
    free_memory = get_free_memory(device)
    memory_ratio = total_mem_req/free_memory 
    

    

    train_chunk = int(0.8 * num_chunks)
    valid_chunk = int(0.2 * num_chunks)
    
    for epoch in range(epochs):

        total_loss = []
        total_train_r2 = []
        for chunk_iteration in range(train_chunk):


            start_idx = int(chunk_iteration * chunk_size)
            stop_idx = int((chunk_iteration + 1) * chunk_size)

            sub_encoding = encoding_list[start_idx: stop_idx]
            sub_properties = properties_tensor[start_idx: stop_idx]
            mus = selfies_to_lpoints(sub_encoding, selfies_alphabet, len_max_molec, vae_encoder, lpoint_size)


            num_clusters = int(memory_ratio)+1
            cluster_size = int(len(mus) / num_clusters)

            print('num_clusters:', num_clusters)
            print('cluster_size:', cluster_size)
            if num_clusters*cluster_size < len(mus):
                num_clusters = num_clusters+1
            if len(mus) + cluster_size - (num_clusters*cluster_size) == 1:
                num_clusters = num_clusters-1


            print('num_clusters*cluster_size', num_clusters*cluster_size)
            print('len(mus)', len(mus))
            print('num_clusters:', num_clusters)




            for cluster_iteration in range(num_clusters):

                start_idx = int(cluster_iteration * cluster_size)
                stop_idx = int((cluster_iteration + 1) * cluster_size)

                cluster_mu = mus[start_idx: stop_idx].detach().to(device)
                cluster_props = sub_properties[start_idx: stop_idx].to(device)


                num_batches_train = int(cluster_mu.shape[0] / batch_size)
                if num_batches_train*batch_size < cluster_mu.shape[0]:
                    num_batches_train = num_batches_train+1
                



                for batch_iteration in range(num_batches_train):
                    
                    print('Epoch:', epoch, 'Chunk iteration:', chunk_iteration, 'Cluster iteration:', cluster_iteration, 'Batch iteration:', batch_iteration)

                    start_idx = batch_iteration * batch_size
                    stop_idx = (batch_iteration + 1) * batch_size
                    batch_mu = cluster_mu[start_idx: stop_idx]
                    batch_props = cluster_props[start_idx: stop_idx]


                    optimizer.zero_grad()


                
                    predictions = model(batch_mu)


                    loss = loss_function(predictions, batch_props)
                    loss.backward()
                    optimizer.step()

                    #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)


                    total_loss.append(loss)

                    y_pred = predictions.squeeze().detach().to('cpu')
                    y_test = batch_props.squeeze().detach().to('cpu')
                    _, _, _,train_r2 = stats(y_test, y_pred)

                    total_train_r2.append(train_r2)
                    


        total_valid_r2 = []
        total_valid_mse = []
        total_valid_mae = []
        total_valid_mre = []

        #model.eval()
        with torch.no_grad():
            for chunk_iteration in range(valid_chunk):


                start_idx = int((train_chunk+chunk_iteration) * chunk_size)
                stop_idx = int((train_chunk+chunk_iteration+1) * chunk_size)

                sub_encoding = encoding_list[start_idx: stop_idx]
                sub_properties = properties_tensor[start_idx: stop_idx]
                mus = selfies_to_lpoints(sub_encoding, selfies_alphabet, len_max_molec, vae_encoder, lpoint_size)

                num_clusters = int(memory_ratio) + 1
                cluster_size = int(len(mus) / num_clusters)

                print('num_clusters:', num_clusters)
                print('cluster_size:', cluster_size)


                if num_clusters*cluster_size < len(mus):
                    num_clusters = num_clusters+1
                if len(mus) + cluster_size - (num_clusters*cluster_size) == 1:
                    num_clusters = num_clusters-1

                print('num_clusters*cluster_size', num_clusters*cluster_size)
                print('len(mus)', len(mus))
                print('num_clusters:', num_clusters)

                for cluster_iteration in range(num_clusters):

                    print('Epoch:', epoch, 'Chunk iteration:', chunk_iteration, 'Cluster iteration:', cluster_iteration)


                    start_idx = int(cluster_iteration * cluster_size)
                    stop_idx = int((cluster_iteration + 1) * cluster_size)

                    cluster_mu = mus[start_idx: stop_idx].to(device)
                    cluster_props = sub_properties[start_idx: stop_idx].to(device)

                    predictions = model(cluster_mu)

                    y_pred = predictions.squeeze().detach().to('cpu')
                    y_test = cluster_props.squeeze().detach().to('cpu')

  
                    sub_mse_valid, sub_mae_valid, sub_mre_valid, valid_r2 = stats(y_test, y_pred)

                    total_valid_mse.append(sub_mse_valid)
                    total_valid_mae.append(sub_mae_valid)
                    total_valid_mre.append(sub_mre_valid)
                    total_valid_r2.append(valid_r2)


            bottom_mus = selfies_to_lpoints(bottom_encoding, selfies_alphabet, len_max_molec, vae_encoder, lpoint_size)
            bottom_mus = bottom_mus.to(device)
            bottom_predictions = model(bottom_mus)
            bottom_mre = torch.mean(torch.abs((bottom_props -bottom_predictions))/bottom_props)

            print('bottom_predictions:', bottom_predictions)
            print('bottom_props:', bottom_props)


        #model.train()
        if epoch % 50 == 0:
            save_params(epoch, model, settings)
            
        
        avg_loss = sum(total_loss) / len(total_loss)
        train_r2 = sum(total_train_r2) / len(total_train_r2)
        r2_valid = sum(total_valid_r2) / len(total_valid_r2)
        mse_valid = sum(total_valid_mse) / len(total_valid_mse)
        mae_valid = sum(total_valid_mae) / len(total_valid_mae)
        mre_valid = sum(total_valid_mre) / len(total_valid_mre)

        scheduler.step(loss)
        
###     

        if bottom_mre < 0.4:
            dump_params(epoch, model, settings)
        
        save_r2_loss(epoch, r2_valid, train_r2, avg_loss, mse_valid, mae_valid, mre_valid, bottom_mre, settings)





def main():
    if os.path.exists("perceptron.yml"):
        settings = yaml.safe_load(open("perceptron.yml", "r"))
        print(settings)
    else:
        print("Expected a file settings.yml but didn't find it.")
        return
    
    model_params = settings['model_params']
    
    learning_rate = settings['hyperparameters']['lr']
    loss_choice = settings['hyperparameters']['loss_choice']
    learning_rate_factor = settings['hyperparameters']['learning_rate_factor']
    learning_rate_patience = settings['hyperparameters']['learning_rate_patience']
    weight_choice = settings['hyperparameters']['weight_choice']




    if loss_choice == 1:
        loss_function = nn.L1Loss()
    if loss_choice == 2:
        loss_function = nn.HuberLoss()
    else:
        loss_function = nn.MSELoss()
    

    len_max_molec, len_alphabet, len_max_mol_one_hot, encoding_alphabet, encoding_list, vae_encoder, properties_tensor, bottom_encoding, bottom_props = data_init(settings)

    print('properties tensor is nan:', torch.sum(properties_tensor.isnan()*1))


###

    model = PropertyRegressionModel(**model_params).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)#, weight_decay=weight_choice)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=learning_rate_factor, patience=learning_rate_patience, verbose=True)

###
    train_model(vae_encoder,
                encoding_list,
                properties_tensor,
                encoding_alphabet,
                len_max_molec,
                optimizer, 
                model, 
                loss_function,
                scheduler,
                bottom_encoding, 
                bottom_props,
                settings)
