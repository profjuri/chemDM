import os
import pandas as pd
import torch
import yaml


import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

import chemistry_vae_symmetric_rnn_final

class PropertyRegressionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, prop_pred_activation, prop_pred_dropout, prop_pred_depth, prop_growth_factor):

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

        self.layers.append(nn.Linear(input_dim, hidden_dim)) ### initial layer

        
        hidden_dims = []
        hidden_dims.append(hidden_dim)
        # Add the rest of the hidden layers
        for p_i in range(1, prop_pred_depth):
            hidden_dims.append(int(prop_growth_factor * hidden_dims[p_i - 1]))
            Ln_layer = nn.Linear(hidden_dims[p_i - 1], hidden_dims[p_i])
            self.layers.append(Ln_layer)

            #BatchNorm_layer = nn.BatchNorm1d(hidden_dims[p_i])
            #self.layers.append(BatchNorm_layer)

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


def stats(y_test, y_pred):

    '''Statistics function that gives you the mse, mae and r^2'''

    '''Arguments:
                    y_test: the true value of whatever property you're analysing (Pytorch float tensor)
                    y_pred: the prediction value of whatever property you're analysing (Pytorch float tensor)'''
    
    '''Outputs:
                    MSE: mean squared error (float)
                    MAE: mean absolute error (float)
                    r2: the r squared coefficient (float)'''

    MAE = torch.mean(torch.abs(y_pred - y_test))
    MSE = torch.mean((y_pred - y_test)*(y_pred - y_test))

    SSR = torch.sum((y_test-y_pred).pow(2))
    SST = torch.sum((y_test-y_test.mean()).pow(2))
    r2 = 1 - SSR/SST

    return MSE, MAE, r2

def save_params(epoch, model, settings):

    '''Save the model object and also the current parameters defining the model'''

    '''Arguments:
                    epoch: the epoch currently being saved (int)
                    model: the mlp object (PropertyRegressionModel object)
                    settings: the settings defined by the .yml file (dict)'''

    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor])

    save_path = settings['settings']['output_folder']

    out_dir = str(save_path)
    out_dir_epoch = out_dir + '/{}'.format(epoch)
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


def save_r2_loss(epoch, r2, train_r2, loss, settings):

    '''This function saves the epoch, total training loss, trainin reconstruction loss, training kld loss and the total validation loss to a .txt file'''

    '''Arguments:
                    epoch: the epoch currently being saved (int)
                    r2: the r squared value of the validation set (float)
                    train_r2: the r squared value of the training set (float)
                    loss: the current loss of the model (float)
                    settings: settings defined by the .yml file (dict)'''

    out_dir = settings['settings']['output_folder']
    log_folder = out_dir + '/settings'  # Replace with the desired folder path
    log_filename = 'r2_loss.txt'

    log_filepath = os.path.join(log_folder, log_filename)

    # Create the log folder if it doesn't exist
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    file_exists = os.path.isfile(log_filepath)

    
    with open(log_filepath, 'a') as file:
        if not file_exists:
            file.write("epoch,loss,val_r2,train_r2\n")
        file.write(f'{epoch},{loss},{r2},{train_r2}\n')

    
def data_init(settings, device):

    '''Data initialisation'''

    '''Arguments:
                    settings: settings defined by the corresponding .yml file (dict)
                    device: the device being used to store data (str)'''
    
    '''Outputs:
                    train_properties_tensor: a tensor containing the properties of the training set (Pytorch float tensor)
                    valid_properties_tensor: a tensor containing the properties of the validation set (Pytorch float tensor)
                    lpoints_train: a tensor containing the latent vectors of the training set (Pytorch float tensor)
                    lpoiints_valid: a tensor containing the latent vectors of the validation set (Pytorch float tensor)'''


    smiles_file = settings['settings']['smiles_file']
    vae_file = settings['settings']['vae_file']
    vae_epoch = settings['settings']['vae_epoch']
    
    batch_size = settings['hyperparameters']['batch_size']

    vae_settings = yaml.safe_load(open(str(vae_file) + "settings/" + "settings.yml", "r"))

    encoder_parameter = vae_settings['encoder']
    selfies_alphabet = vae_settings['alphabet']
    torch_seed = vae_settings['data']['torch_seed']
    vae_weights_path = str(vae_file) + str(vae_epoch) + "/E.pt"

    encoding_list, _, largest_molecule_len, _, _, _ = chemistry_vae_symmetric_rnn_final.get_selfie_and_smiles_encodings_for_dataset(smiles_file)
    data = chemistry_vae_symmetric_rnn_final.multiple_selfies_to_hot(encoding_list, largest_molecule_len, selfies_alphabet)
    len_max_molec = data.shape[1]
    len_alphabet = data.shape[2]


    vae_encoder = chemistry_vae_symmetric_rnn_final.VAEEncoder(in_dimension=(len_max_molec*len_alphabet), **encoder_parameter).to(device)
    model_weights = torch.load(vae_weights_path)
    vae_encoder.load_state_dict(model_weights)


    train_valid_test_size = [0.8, 0.2, 0.0]
    idx_train_val = int(len(data) * train_valid_test_size[0])
    idx_val_test = idx_train_val + int(len(data) * train_valid_test_size[1])

    torch.manual_seed(torch_seed)
    data = torch.tensor(data, dtype=torch.float).to(device)
    rand_perms = torch.randperm(data.size()[0])
    data = data[rand_perms]
    inp_flat_one_hot = data.flatten(start_dim=1)
    inp_flat_one_hot = inp_flat_one_hot.unsqueeze(0)


    _, mus, _ = vae_encoder(inp_flat_one_hot)

    lpoints_train = mus[0:idx_train_val].to('cpu')
    lpoints_valid = mus[idx_train_val:idx_val_test].to('cpu')

####

    my_file = pd.read_csv(smiles_file, index_col=None)##The file you want to train on, should contain SMILES reps, latent space reps and properties


    properties_df = my_file.drop(columns=['smiles']) ##drop all smiles from the properties df
    properties_array = properties_df.to_numpy() ##convert the df to numpy array
    properties_tensor = torch.tensor(properties_array,dtype=torch.float32)
    properties_tensor = properties_tensor[rand_perms]


    train_properties_tensor = properties_tensor[0:idx_train_val].to('cpu')
    valid_properties_tensor = properties_tensor[idx_train_val:idx_val_test].to('cpu')

    return train_properties_tensor, valid_properties_tensor, lpoints_train, lpoints_valid

def train_model(lpoints_train, train_properties_tensor, lpoints_valid, valid_properties_tensor, optimizer, model, loss_function, device, scheduler, settings):

    '''Train the multi-layered perceptron'''

    '''Arguments:   
                    lpoints_train: the training set of latent vectors (Pytorch float tensor)
                    train_properties_tensor: the training set of the properties being used for prediction (Pytorch float tensor)
                    lpoints_valid: the latent vector validation set (Pytorch float tensor)
                    valid_properties_tensor: tensor containing the latent vectors of the validation set (Pytorch float tensor)
                    optimizer: the optimizer used to modify the weights after a back propagation (Pytorch torch.optim object)
                    model: the multi-layered perceptron object (PropertyRegressionModel object)
                    loss_function: the loss function being used, e.g., MSELoss (str)
                    device: the device being used to store data (str)
                    scheduler: function used to modify the learning rate (torch.optim.lr_scheduler object)
                    settings: the settings defined by the .yml file (dict)'''

    total_loss = 0.0

    batch_size = settings['hyperparameters']['batch_size']
    epochs = settings['hyperparameters']['epochs']

    num_batches_train = int(len(lpoints_train) / batch_size)
    bestr2 = 0


    for epoch in range(epochs):

        total_loss = 0.0

        for batch_iteration in range(num_batches_train):

            start_idx = batch_iteration * batch_size
            stop_idx = (batch_iteration + 1) * batch_size
            batch_mu = lpoints_train[start_idx: stop_idx].to(device)
            batch_props = train_properties_tensor[start_idx: stop_idx].to(device)

            optimizer.zero_grad()

                
            predictions = model(batch_mu)
            loss = loss_function(predictions, batch_props)

            loss.backward()
            optimizer.step()


            total_loss += loss.item()

        if epoch % 50 == 0:
            save_params(epoch, model, settings)

        _, _, train_r2 = validation(model, device, lpoints_train, train_properties_tensor, batch_size)
        avg_loss = total_loss / num_batches_train

        scheduler.step(loss)


###     

        mse_valid, mae_valid, r2_valid = validation(model, device, lpoints_valid, valid_properties_tensor, batch_size)

        if r2_valid > bestr2:
            bestr2 = r2_valid
        
        save_r2_loss(epoch, r2_valid, train_r2, avg_loss, settings)





def validation(model, device, lpoints, properties_tensor, batch_size):

    '''Function to provide statistical metrics for the validation set'''

    '''Arguments:
                    model: the multi-layered perceptron object (PropertyRegressionModel object)
                    device: the device being used to store data (str)
                    lpoints: the latent vectors you are looking to vlaidate (Pytorch float tensor)
                    properties_tensor: tensor containing the properties of the latent vectors you are interested in (Pytorch float tensor)
                    batch_size: the batch size (int)'''
    
    '''Outputs:
                    mse: mean squared error (float)
                    mae: mean absolute error (float)
                    r2: the r squared coefficient (float)'''

    print('lpoints_valid', lpoints.size())
    print('valid_properties_tensor', properties_tensor.size())

    num_batches_train = int(len(lpoints) / batch_size)
    print(num_batches_train)
    if num_batches_train*batch_size < len(lpoints):
        num_batches_train = num_batches_train+1
        print(num_batches_train)


    model.eval()
    preds_tensor = torch.empty(0, dtype=torch.float32).to('cpu')

    for batch_iteration in range(num_batches_train):

        start_idx = batch_iteration * batch_size
        stop_idx = (batch_iteration + 1) * batch_size
        batch_mu = lpoints[start_idx: stop_idx].to(device)
        batch_props = properties_tensor[start_idx: stop_idx].to(device)

        
        with torch.no_grad():
            test_predictions = model(batch_mu.squeeze()).to('cpu')
        test_predictions = test_predictions.squeeze()


        preds_tensor = torch.cat((preds_tensor, test_predictions))


    y_pred = preds_tensor.squeeze().detach().to('cpu')
    y_test = properties_tensor.squeeze().detach().to('cpu')

    print('y_pred', y_pred.size())
    print('y_test', y_test.size())



    mse, mae, r2 = stats(y_test, y_pred)
    model.train()
    return mse.item(), mae.item(), r2.item()





def main():
    if os.path.exists("perceptron.yml"):
        settings = yaml.safe_load(open("perceptron.yml", "r"))
    else:
        print("Expected a file settings.yml but didn't find it.")
        return
    
    model_params = settings['model_params']
    
    learning_rate = settings['hyperparameters']['lr']
    loss_choice = settings['hyperparameters']['loss_choice']
    learning_rate_factor = settings['hyperparameters']['learning_rate_factor']
    learning_rate_patience = settings['hyperparameters']['learning_rate_patience']
    weight_choice = settings['hyperparameters']['weight_choice']


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    if loss_choice == 1:
        loss_function = nn.L1Loss()
    if loss_choice == 2:
        loss_function = nn.HuberLoss()
    else:
        loss_function = nn.MSELoss()
    

    train_properties_tensor, valid_properties_tensor, lpoints_train, lpoints_valid = data_init(settings, device)



###

    model = PropertyRegressionModel(**model_params).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_choice)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=learning_rate_factor, patience=learning_rate_patience, verbose=True)

###

    lpoints_train2 = lpoints_train.detach().to(device)
    train_properties_tensor2 = train_properties_tensor.detach().to(device)
    lpoints_valid2 = lpoints_valid.detach().to(device)
    valid_properties_tensor2 = valid_properties_tensor.detach().to(device)

###
    train_model(lpoints_train2,
                train_properties_tensor2,
                lpoints_valid2, 
                valid_properties_tensor2,
                optimizer, 
                model, 
                loss_function, 
                device, 
                scheduler, 
                settings)
