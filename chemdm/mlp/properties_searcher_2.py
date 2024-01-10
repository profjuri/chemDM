import os
import pandas as pd
import torch
import yaml

import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

import chemistry_vae_symmetric_rnn_final
from chemistry_vae_symmetric_rnn_final import VAEEncoder

class PropertyRegressionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, prop_pred_activation, prop_pred_dropout, prop_pred_depth, prop_growth_factor):

        '''Multi layer perceptron'''

        '''Arguments:
                        input_dim: the latent space dimension
                        hidden: the size of the first hidden layer
                        prop_pred_activation: the activation function used
                        prop_pred_dropout: the dropout coefficient
                        prop_pred_depth: the number of hidden layers
                        prop_growth_factor: the coefficient each hidden layer number is multiplied by. E.g., hidden = 256, prop_growth_factor = 0.5, second layer = 128'''


        super(PropertyRegressionModel, self).__init__()

        self.ls_in = nn.Linear(input_dim, hidden_dim)
        self.activation = self._get_activation(prop_pred_activation)
        self.dropout = nn.Dropout(prop_pred_dropout)
        self.layers = nn.ModuleList()

        
        hidden_dims = []
        hidden_dims.append(hidden_dim)
        # Add the rest of the hidden layers
        for p_i in range(1, prop_pred_depth):
            hidden_dims.append(int(prop_growth_factor * hidden_dims[p_i - 1]))
            hidden_layer = nn.Linear(hidden_dims[p_i - 1], hidden_dims[p_i])
            self.layers.append(hidden_layer)
            BatchNorm_layer = nn.BatchNorm1d(hidden_dims[p_i])  
            self.layers.append(BatchNorm_layer)
            if prop_pred_dropout > 0:
                self.layers.append(nn.Dropout(prop_pred_dropout))

        self.reg_prop_pred = nn.Linear(hidden_dims[len(hidden_dims)-1], 1)  # For regression tasks, single output node


    def forward(self, x):

        '''Forward pass through the MLP'''

        '''Arguments:
                        x: transformed latent vectors'''

        '''Outputs:
                        reg_prop_out: the predicted property output'''

        x = self.ls_in(x)
        x = self.activation(x)


        
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)
            if isinstance(layer, nn.Linear) and self.dropout.p > 0:
                x = self.dropout(x)

        reg_prop_out = self.reg_prop_pred(x)
        return reg_prop_out

    def _get_activation(self, activation_name):

        '''Gives you the activation layer'''

        '''Arguments:
                        activation_name: the name of the activation functions shown below'''



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
                    y_test: the true value of whatever property you're analysing
                    y_pred: the prediction value of whatever property you're analysing'''
    
    '''Outputs:
                    MSE: mean squared error
                    MAE: mean absolute error
                    r2: the r squared coefficient'''

    MAE = torch.abs(y_pred - y_test).sum()
    MSE = ((y_pred - y_test)*(y_pred - y_test)).sum()

    SSR = torch.sum((y_pred-y_test.mean()).pow(2))
    SST = torch.sum((y_test-y_test.mean()).pow(2))

    r2 = 1 - SSR/SST

    return MSE, MAE, r2

def save_params(mse, mae, r2, model, settings):

    '''Save the model object and also the current parameters defining the model'''

    '''Arguments:
                    mse: the mean squared error of the current model predictions
                    mae: the mean absolute error of the current model predictions
                    r2: the r squared value of the current model predictions
                    model: the mlp object
                    settings: the settings defined by the .yml file'''
    
    input_dim = settings['settings']['input_dim']
    lr = settings['hyperparameters']['lr']
    batch_size = settings['hyperparameters']['batch_size']
    hidden_dim = settings['hyperparameters']['hidden_dim']
    prop_hidden_dim = settings['hyperparameters']['prop_hidden_dim']
    prop_pred_activation = settings['hyperparameters']['prop_pred_activation']
    prop_pred_dropout = settings['hyperparameters']['dropout']
    prop_pred_depth = settings['hyperparameters']['depth']
    prop_growth_factor = settings['hyperparameters']['growth']
    loss_choice = settings['hyperparameters']['loss_choice']
    factor_choice = settings['hyperparameters']['learning_rate_factor']
    patience_choice = settings['hyperparameters']['learning_rate_patience']
    epochs = settings['hyperparameters']['epochs']
    weight_choice = settings['hyperparameters']['weight_choice']



    out_dir = settings['settings']['output_folder']
    log_folder = out_dir  # Replace with the desired folder path
    log_filename = 'results.txt'

    log_filepath = os.path.join(log_folder, log_filename)
    #torch.save(model, str(out_dir) + str(r2))
    # Create the log folder if it doesn't exist
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    
    torch.save(model.state_dict(), out_dir + '/' + str(r2) + '.pt')

    file_exists = os.path.isfile(log_filepath)
    
    with open(log_filepath, 'a') as file:
        if not file_exists:
            file.write(" lr, batch_size, input_dim, hidden_dim, prop_hidden_dim, prop_pred_activation, prop_pred_dropout, prop_pred_depth, prop_growth_factor, epochs, loss_choice, factor_choice, patience_choice,weight_choice, mse, mae, r2\n")
        file.write(f'{lr},{batch_size},{input_dim},{hidden_dim},{prop_hidden_dim},{prop_pred_activation},{prop_pred_dropout},{prop_pred_depth},{prop_growth_factor},{epochs},{loss_choice},{factor_choice},{patience_choice},{weight_choice},{mse},{mae},{r2}\n')


def save_r2_loss(epoch, r2, train_r2, loss, settings):

    '''This function saves the epoch, total training loss, trainin reconstruction loss, training kld loss and the total validation loss to a .txt file'''

    '''Arguments:
                    epoch: the epoch currently being saved
                    r2: the r squared value of the validation set
                    train_r2: the r squared value of the training set
                    loss: the current loss of the model
                    settings: settings defined by the .yml file'''

    out_dir = settings['settings']['output_folder']
    log_folder = out_dir  # Replace with the desired folder path
    log_filename = 'r2_loss.txt'


    log_folder = out_dir  # Replace with the desired folder path
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
                    settings: settings defined by the corresponding .yml file
                    device: the device being used to store data'''
    
    '''Outputs:
                    train_properties_tensor: a tensor containing the properties of the training set
                    valid_properties_tensor: a tensor containing the properties of the validation set
                    lpoints_train: a tensor containing the latent vectors of the training set
                    lpoiints_valid: a tensor containing the latent vectors of the validation set'''


    smiles_file = settings['settings']['smiles_file']
    vae_file = settings['settings']['vae_file']
    vae_epoch = settings['settings']['vae_epoch']
    torch_seed = settings['settings']['torch_seed']
    batch_size = settings['hyperparameters']['batch_size']

    vae_settings = yaml.safe_load(open(str(vae_file) + "settings/" + "settings.yml", "r"))

    encoder_parameter = vae_settings['encoder']
    selfies_alphabet = vae_settings['alphabet']
    vae_weights_path = str(vae_file) + str(vae_epoch) + "/E.pt"

    encoding_list, encoding_alphabet, largest_molecule_len, _, _, _ = chemistry_vae_symmetric_rnn_final.get_selfie_and_smiles_encodings_for_dataset(smiles_file)
    data = chemistry_vae_symmetric_rnn_final.multiple_selfies_to_hot(encoding_list, largest_molecule_len, selfies_alphabet)
    len_max_molec = data.shape[1]
    len_alphabet = data.shape[2]


    vae_encoder = VAEEncoder(in_dimension=(len_max_molec*len_alphabet), **encoder_parameter).to(device)
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

    lpoints_train = mus[0:idx_train_val]
    lpoints_valid = mus[idx_train_val:idx_val_test]

####

    my_file = pd.read_csv(smiles_file, index_col=None)##The file you want to train on, should contain SMILES reps, latent space reps and properties


    properties_df = my_file.drop(columns=['smiles']) ##drop all smiles from the properties df
    properties_array = properties_df.to_numpy() ##convert the df to numpy array
    properties_tensor = torch.tensor(properties_array,dtype=torch.float32)
    properties_tensor = properties_tensor[rand_perms]


    train_properties_tensor = properties_tensor[0:idx_train_val]
    valid_properties_tensor = properties_tensor[idx_train_val:idx_val_test]

    return train_properties_tensor, valid_properties_tensor, lpoints_train, lpoints_valid

def train_model(num_batches_train, batch_size, lpoints_train, train_properties_tensor, optimizer, model, loss_function):

    '''Train the multi-layered perceptron'''

    '''Arguments:
                    num_batches_train: the number of batches to be used during training
                    batch_size: the size of each batch
                    lpoints_train: the training set of latent vectors
                    train_properties_tensor: the training set of the properties being used for prediction
                    optimizer: the optimizer used to modify the weights after a back propagation
                    model: the multi-layered perceptron object
                    loss_function: the loss function being used, e.g., MSELoss'''
    
    '''Outputs:
                    model: the modified MLP
                    total_loss: the total trainin loss
                    r2_train: the training r squared
                    loss: the loss on the final batch iteration'''

    total_loss = 0.0
    r2_train = 0.0

    for batch_iteration in range(num_batches_train):

        start_idx = batch_iteration * batch_size
        stop_idx = (batch_iteration + 1) * batch_size
        batch_mu = lpoints_train[start_idx: stop_idx]
        batch_props = train_properties_tensor[start_idx: stop_idx]

        optimizer.zero_grad()

                
        predictions = model(batch_mu)
        loss = loss_function(predictions, batch_props)

        loss.backward()
        optimizer.step()
                
        y_pred = predictions.detach()
        y_test = batch_props.detach()

        mse, mae, r2 = stats(y_test, y_pred) 

        total_loss += loss.item()
        r2_train += r2

    return model, total_loss, r2_train, loss




def valdiation(model, lpoints_valid, valid_properties_tensor):

    '''Function to provide statistical metrics for the validation set'''

    '''Arguments:
                    model: the multi-layered perceptron object
                    lpoints_valid: the latent vector validation set
                    valid_properties_tensor: tensor containing the latent vectors of the validation set'''
    
    '''Outputs:
                    mse: mean squared error
                    mae: mean absolute error
                    r2: the r squared coefficient'''

    with torch.no_grad():
        test_predictions = model(lpoints_valid.squeeze())

    y_pred = test_predictions.detach()
    y_test = valid_properties_tensor.detach()


    mse, mae, r2 = stats(y_test, y_pred)

    return mse, mae, r2





def main():
    if os.path.exists("perceptron.yml"):
        settings = yaml.safe_load(open("perceptron.yml", "r"))
    else:
        print("Expected a file settings.yml but didn't find it.")
        return
    
    input_dim = settings['settings']['input_dim']

    learning_rate = settings['hyperparameters']['lr']
    batch_size = settings['hyperparameters']['batch_size']
    hidden_dim = settings['hyperparameters']['hidden_dim']
    prop_hidden_dim = settings['hyperparameters']['prop_hidden_dim']
    prop_pred_activation = settings['hyperparameters']['prop_pred_activation']
    prop_pred_dropout = settings['hyperparameters']['dropout']
    prop_pred_depth = settings['hyperparameters']['depth']
    prop_growth_factor = settings['hyperparameters']['growth']
    loss_choice = settings['hyperparameters']['loss_choice']
    learning_rate_factor = settings['hyperparameters']['learning_rate_factor']
    learning_rate_patience = settings['hyperparameters']['learning_rate_patience']
    epochs = settings['hyperparameters']['epochs']
    weight_choice = settings['hyperparameters']['weight_choice']



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lambda_choice = 0 

    if loss_choice == 1:
        loss_function = nn.L1Loss()
    if loss_choice == 2:
        loss_function = nn.HuberLoss()
    else:
        loss_function = nn.MSELoss()
    

    train_properties_tensor, valid_properties_tensor, lpoints_train, lpoints_valid = data_init(settings, device)

###

    bestr2 = 0
    r2_list = []
    epoch_list = []

###

    model = PropertyRegressionModel(input_dim, hidden_dim, prop_pred_activation, prop_pred_dropout, prop_pred_depth, prop_growth_factor).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_choice)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=learning_rate_factor, patience=learning_rate_patience, verbose=True)

###

    lpoints_train2 = lpoints_train.detach().to(device)
    train_properties_tensor2 = train_properties_tensor.detach().to(device)
    num_batches_train = int(len(lpoints_train2) / batch_size)

###

    for epoch in range(epochs):

        model, total_loss, r2_train, loss = train_model(num_batches_train, batch_size, lpoints_train2, train_properties_tensor2, optimizer, model, loss_function)

        avg_loss = total_loss / num_batches_train
        avg_r2 = r2_train / num_batches_train

        epoch_list.append(epoch)
        scheduler.step(loss)


###
        mse, mae, r2 = valdiation(model, lpoints_valid, valid_properties_tensor)


        r2_list.append(r2)
        print("Current r2:", r2, "Best r2:", bestr2)
        if r2 > 0.91:
            bestr2 = r2
            save_params(mse, mae, r2, model, settings)
        save_r2_loss(epoch, r2, avg_r2, avg_loss, settings)

