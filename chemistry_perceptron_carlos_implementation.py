import os
import numpy as np
import pandas as pd
import selfies
import glob
import torch
import chemistry_vae_selfies_carlos
import data_loader
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import random

from random import sample
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score





class PropertyRegressionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, prop_hidden_dim, prop_pred_activation, prop_pred_dropout, prop_pred_depth, prop_growth_factor):
        super(PropertyRegressionModel, self).__init__()
        
        self.ls_in = nn.Linear(input_dim, hidden_dim)
        self.activation = self._get_activation(prop_pred_activation)
        self.dropout = nn.Dropout(p=prop_pred_dropout)
        self.layers = nn.ModuleList()
        self.batchnorm_first = nn.BatchNorm1d(hidden_dim)
        self.first_hidden_dim = prop_hidden_dim    


        # Add the first hidden layer manually to match the input_dim and prop_hidden_dim
        first_hidden_layer = nn.Linear(hidden_dim, prop_hidden_dim)
        print("layer #1 has "+str(hidden_dim)+' neurons.')
        self.layers.append(first_hidden_layer)

        #array of layer dimensions which shrink by growth rate.
        self.layer_dims = [int(round(prop_hidden_dim*(prop_growth_factor**n))) for n in range(prop_pred_depth)]

        #array of batchnorm layers
        self.batchnorm_hidden = nn.ModuleList([nn.BatchNorm1d(dim) for dim in self.layer_dims])

        # Add the rest of the hidden layers
        if prop_pred_depth > 1:
            for p_i in range(1, prop_pred_depth):
                hidden_layer = nn.Linear(self.layer_dims[p_i-1], self.layer_dims[p_i])  # Output size matches the input size
                print("layer #"+str(p_i+1)+' has '+str(self.layer_dims[p_i-1])+' neurons.')
                self.layers.append(hidden_layer)

        self.reg_prop_pred = nn.Linear(self.layer_dims[prop_pred_depth-1], 1)  # For regression tasks, single output node
        print("layer #"+str(prop_pred_depth+1)+" has "+str(self.layer_dims[prop_pred_depth-1])+' neurons.')

    def forward(self, x):
        batch_size = x.shape[0]        

        x = self.ls_in(x)
        x = self.batchnorm_first(x)
        x = self.activation(x)     
        x = self.dropout(x)

        for i in range(len(self.layers)):
            x = self.layers[i](x)
            x = self.batchnorm_hidden[i](x)
            x = self.activation(x)
            x = self.dropout(x)

        reg_prop_out = self.reg_prop_pred(x)
        return reg_prop_out

    def _get_activation(self, activation_name):
        if activation_name == 'relu':
            return nn.ReLU()
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
        

def create_onehot_instance(selfie_input,largest_selfies_len,selfies_alphabet_in):

    inttest_hot, arraytest_hot = data_loader.selfies_to_hot(selfie_input,largest_selfies_len, selfies_alphabet_in)
    x = torch.from_numpy(arraytest_hot).flatten().float().unsqueeze(0)
    return x



def create_latent_space_vector(selfie_input,largest_selfies_len,selfies_alphabet_in, vae_encoder, vae_decoder):

    x = create_onehot_instance(selfie_input,largest_selfies_len,selfies_alphabet_in)

    z =set()
    vae_encoder.eval()
    vae_decoder.eval()
    z, mu, log_var = vae_encoder(x)

    return z.unsqueeze(0)


def filter(list_of_properties, largest_selfies_len, selfies_alphabet, vae_encoder, vae_decoder):
    selfie_input_valid = []
    latent_space_valid = []
    props_valid = []
    length_properties = len(list_of_properties)



    for i in range(length_properties):
        
        try:
            selfie_input = selfies.encoder(list_of_properties[i][0])            
            latent_vector = create_latent_space_vector(selfie_input,largest_selfies_len,selfies_alphabet, vae_encoder, vae_decoder)
            selfie_input_valid.append(selfie_input)
            latent_space_valid.append([latent_vector] )
            props_valid.append([float(list_of_properties[i][1]), float(list_of_properties[i][2])] )
            if i%1000:
                print('success',(i/length_properties)*100,"\%")
        except Exception as ve:
            if i%1000:
                print("Skipped, ",(i/length_properties)*100,"\%")
            continue

    return selfie_input_valid, latent_space_valid, props_valid

def make_tensors(train_latent, test_latent, train_prop, test_prop):
    train_latent_space_tensor = torch.stack(train_latent)
    test_latent_space_tensor = torch.stack(test_latent)
    train_properties_tensor = torch.tensor(train_prop)
    test_properties_tensor = torch.tensor(test_prop)

    return train_latent_space_tensor, test_latent_space_tensor, train_properties_tensor, test_properties_tensor

def stats(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return mse, mae, r2

def save_params(input_dim, hidden_dim, prop_hidden_dim, prop_pred_activation, prop_pred_dropout, prop_pred_depth, prop_growth_factor, epochs, counter, mse, mae, r2):

    out_dir = './results/property_hyperparams/'
    log_folder = out_dir  # Replace with the desired folder path
    log_filename = 'results.txt'

    log_filepath = os.path.join(log_folder, log_filename)

    # Create the log folder if it doesn't exist
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    file_exists = os.path.isfile(log_filepath)
    
    with open(log_filepath, 'a') as file:
        if not file_exists:
            file.write("counter, input_dim, hidden_dim, prop_hidden_dim, prop_pred_activation, prop_pred_dropout, prop_pred_depth, prop_growth_factor, epochs, mse, mae, r2\n")
        file.write(f'{counter},{input_dim},{hidden_dim},{prop_hidden_dim},{prop_pred_activation},{prop_pred_dropout},{prop_pred_depth},{prop_growth_factor},{epochs},{mse},{mae},{r2}\n')


def main():

    list_of_properties = []
    energy_gaps = []

    input_dim = 25 ### Size of the latent space
    num_of_cycles = 1000 ### The number of hyperparameter searches you want to do, change accordingly
    counter = 0

###

    my_file = pd.read_csv('./datasets/QM9listprops.csv', index_col=None)##The file you want to train on, should contain SMILES reps, latent space reps and properties
    my_file.dropna()


    for i in range(len(my_file)):

        DipoleIn = my_file['dipole_moment'][i]
        GapIn = my_file['energy_gap'][i]
        SMILESCodeIn = my_file['smiles'][i]
        list_of_properties.append([SMILESCodeIn, GapIn, DipoleIn])

###

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    folder_path = "./datasets/"
    file_name = "SelectedSMILES_QM9.txt"

    full_path = folder_path + file_name

    selfies_list, selfies_alphabet, largest_selfies_len, smiles_list, smiles_alphabet, largest_smiles_len = chemistry_vae_selfies_carlos.get_selfie_and_smiles_encodings_for_dataset(full_path)

    selfies_alphabet = ['[#Branch2]', '[Ring2]', '[Branch2]', '[=Branch2]', '[O]', '[=O]', '[=C]', '[=N]', '[#Branch1]', '[=Branch1]', '[nop]', '[N]', '[Branch1]', '[F]', '[#C]', '[#N]', '[Ring1]', '[C]']

###

    # define source file location
    file_to_load =  "./saved_models_RNN/"
    # training file name encoder
    training_file_nameE = "300/E"
    # training file name decoder
    training_file_nameD = "300/D"
    # load data
    #load_data_trained = file_to_load + training_file_nameE
    # Alphabet has 18 letters, largest molecule is 21 letters. (build this as an output function later ... )
    largest_selfies_len_dataset = largest_selfies_len
    largest_smiles_len_dataset = largest_smiles_len

    #in_dimension = len(selfies_alphabet)*largest_selfies_len
    in_dimension = len(smiles_alphabet)*largest_smiles_len

    # load the trained encoder
    vae_encoder = torch.load(file_to_load + training_file_nameE) #, map_location=torch.device(device="cpu"))
    #print(vae_encoder)

    # load the trained decoder
    vae_decoder = torch.load(file_to_load + training_file_nameD) #, map_location=torch.device(device="cpu"))
    #print(vae_decoder)


    selfies_alphabet = ['[#Branch2]', '[Ring2]', '[Branch2]', '[=Branch2]', '[O]', '[=O]', '[=C]', '[=N]', '[#Branch1]', '[=Branch1]', '[nop]', '[N]', '[Branch1]', '[F]', '[#C]', '[#N]', '[Ring1]', '[C]']

###

    smiles_rep, latent_rep, props_used = filter(list_of_properties, largest_selfies_len_dataset, selfies_alphabet, vae_encoder, vae_decoder)


    latent_space_vectors_valid = [item[0].detach().squeeze(0) for item in latent_rep]
    properties_training = [torch.tensor(property_vector) for property_vector in props_used]  # this will convert lists to tensors


    for i in range(len(properties_training)):
        energy_gaps.append(props_used[i][0])


    train_size = round(len(latent_space_vectors_valid)*0.8)

    latents_train = latent_space_vectors_valid[:train_size]
    latents_test = latent_space_vectors_valid[train_size:]

    props_train = energy_gaps[:train_size]
    props_test = energy_gaps[train_size:]


    train_latent_space_tensor, test_latent_space_tensor, train_properties_tensor, test_properties_tensor = make_tensors(latents_train, latents_test, props_train, props_test)


    train_latent_space_tensor = train_latent_space_tensor.view(-1, input_dim)
    test_latent_space_tensor = test_latent_space_tensor.view(-1, input_dim)

####

    for i in range(num_of_cycles):
            # Sample input dimensions for 100,000 compounds represented by 25 numbers
        hidden_dim = random.choice([32, 64, 128, 256, 512])  # You can increase the hidden dimensions for more complexity
        prop_hidden_dim = int(hidden_dim/2)  # You can adjust this based on the complexity of the task
        prop_pred_activation = random.choice(['relu', 'leaky_relu'])
        prop_pred_dropout = random.choice([0, 0.1, 0.2])
        prop_pred_depth = random.choice([1, 2, 3, 4])
        prop_growth_factor = random.choice([0.4, 0.5, 0.6])
        lr_random = random.choice([0.01, 0.005, 0.001])
        epochs = 2000





        # Instantiate the model with hyperparameters
        model = PropertyRegressionModel(input_dim, hidden_dim, prop_hidden_dim, prop_pred_activation, prop_pred_dropout, prop_pred_depth, prop_growth_factor)

        # Define your loss function (e.g., Mean Squared Error)
        loss_function = nn.MSELoss()

        # Define your optimizer (e.g., Adam)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_random)

###

        for epoch in range(epochs):
            optimizer.zero_grad()
            predictions = model(train_latent_space_tensor)
            loss = loss_function(predictions.squeeze(), train_properties_tensor)
            loss.backward()
            optimizer.step()



###

        with torch.no_grad():
            test_predictions = model(test_latent_space_tensor)
            test_loss = loss_function(test_predictions.squeeze(), test_properties_tensor)
            print(f"Test Loss: {test_loss.item():.4f}")

###


        # Generate predictions for the test set
        y_pred = test_predictions
        y_test = props_test

        mse, mae, r2 = stats(y_pred, y_test)


        save_params(input_dim, hidden_dim, prop_hidden_dim, prop_pred_activation, prop_pred_dropout, prop_pred_depth, prop_growth_factor, epochs, counter, mse, mae, r2)
        counter = counter + 1






    
