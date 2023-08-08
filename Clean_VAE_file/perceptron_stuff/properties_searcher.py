import os
import numpy as np
import pandas as pd
import selfies
import glob
import torch
import chemistry_vae_selfies
import data_loader
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import random
import time
import math


from random import sample
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.optim.lr_scheduler import StepLR





import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau



class PropertyRegressionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, prop_hidden_dim, prop_pred_activation, prop_pred_dropout, prop_pred_depth, prop_growth_factor, var_tnsr):
        super(PropertyRegressionModel, self).__init__()

        self.ls_in = nn.Linear(input_dim, hidden_dim)
        self.activation = self._get_activation(prop_pred_activation)
        self.dropout = nn.Dropout(prop_pred_dropout)
        self.layers = nn.ModuleList()
        self.batchnorm_first = nn.BatchNorm1d(hidden_dim)

        self.variance_tensor = var_tnsr

        #self.batchnorm_hidden = nn.ModuleList([nn.BatchNorm1d(dim) for dim in self.layer_dims])


        # Add the first hidden layer manually to match the input_dim and prop_hidden_dim
        first_hidden_layer = nn.Linear(hidden_dim, prop_hidden_dim)
        self.layers.append(first_hidden_layer)
        if prop_pred_dropout > 0:
            self.layers.append(nn.Dropout(prop_pred_dropout))

        # Add the rest of the hidden layers
        if prop_pred_depth > 1:
            for p_i in range(1, prop_pred_depth):
                hidden_layer = nn.Linear(prop_hidden_dim, prop_hidden_dim)  # Output size matches the input size
                self.layers.append(hidden_layer)
                self.activation = self._get_activation(prop_pred_activation)
                if prop_pred_dropout > 0:
                    self.layers.append(nn.Dropout(prop_pred_dropout))

        self.reg_prop_pred = nn.Linear(prop_hidden_dim, 1)  # For regression tasks, single output node

    def init_weights(self, variance_tensor):
        with torch.no_grad():
            # Scale the initial weights using the variance tensor
            self.ls_in.weight.data = self.ls_in.weight.data * variance_tensor.unsqueeze(0)

    def forward(self, x, variance_tensor=None):
        if variance_tensor is not None:
            # Scale the initial weights using the provided variance tensor
            self.init_weights(variance_tensor)

        x = self.ls_in(x)
        x = self.activation(x)
        x = self.batchnorm_first(x)
        


        for layer in self.layers:
            x = layer(x)
            #x = self.batch_norm(x)  # Add BatchNorm after each linear layer
            x = self.activation(x)  # Apply activation function after BatchNorm
            #x = self.batchnorm_hidden(x)
            if isinstance(layer, nn.Linear) and self.dropout.p > 0:
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
        

def min_max_normalize(tensor):
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    return normalized_tensor

        

def create_onehot_instance(selfie_input,largest_selfies_len,selfies_alphabet_in):

    inttest_hot, arraytest_hot = data_loader.selfies_to_hot(selfie_input,largest_selfies_len, selfies_alphabet_in)
    x = torch.from_numpy(arraytest_hot).flatten().float().unsqueeze(0)
    return x

def create_onehot_instance_many(selfie_input,largest_selfies_len,selfies_alphabet_in):
    one_hot_list = []

    for i in range(len(selfie_input)):
        inttest_hot, arraytest_hot = data_loader.selfies_to_hot(selfie_input[i],largest_selfies_len, selfies_alphabet_in)
        x = torch.from_numpy(arraytest_hot).flatten().float().unsqueeze(0)
        one_hot_list.append(x)

    return one_hot_list




def create_latent_space_vector(selfie_input,largest_selfies_len,selfies_alphabet_in, vae_encoder, vae_decoder):

    x = create_onehot_instance(selfie_input,largest_selfies_len,selfies_alphabet_in)

    z =set()
    vae_encoder.eval()
    vae_decoder.eval()
    z, mu, log_var = vae_encoder(x)

    return z.unsqueeze(0)


def gen_new_latent_space(one_hot,largest_selfies_len,selfies_alphabet_in, vae_encoder, vae_decoder):

    valid_latent_spaces = []
    valid_log_vars = []

    for i in range(len(one_hot)):
        x = one_hot[i]

        z =set()
        vae_encoder.eval()
        vae_decoder.eval()
        z, mu, log_var = vae_encoder(x)
        valid_latent_spaces.append(mu.unsqueeze(0))
        valid_log_vars.append(log_var.unsqueeze(0))

    return valid_latent_spaces, valid_log_vars



def filter(list_of_properties, largest_selfies_len, selfies_alphabet, vae_encoder, vae_decoder):
    selfie_input_valid = []
    latent_space_valid = []
    props_valid = []


    for i in range(len(list_of_properties)):
        try:
            selfie_input = selfies.encoder(list_of_properties[i][0])
            print(selfie_input)
            latent_vector = create_latent_space_vector(selfie_input,largest_selfies_len,selfies_alphabet, vae_encoder, vae_decoder)
            selfie_input_valid.append(selfie_input)
            latent_space_valid.append(latent_vector)
            props_valid.append([float(list_of_properties[i][1]), float(list_of_properties[i][2])] )
        except Exception as ve:
            print(ve)

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

def save_params(input_dim, lr, hidden_dim, prop_hidden_dim, prop_pred_activation, prop_pred_dropout, prop_pred_depth, prop_growth_factor, epochs, counter, mse, mae, r2, model, batch_size):

    out_dir = '../../property_hyperparams/'
    log_folder = out_dir  # Replace with the desired folder path
    log_filename = 'results.txt'

    log_filepath = os.path.join(log_folder, log_filename)
    torch.save(model, '../../property_hyperparams/' + str(r2))
    # Create the log folder if it doesn't exist
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    
    torch.save(model, out_dir + str(r2))

    file_exists = os.path.isfile(log_filepath)
    
    with open(log_filepath, 'a') as file:
        if not file_exists:
            file.write("counter, lr, batch_size, input_dim, hidden_dim, prop_hidden_dim, prop_pred_activation, prop_pred_dropout, prop_pred_depth, prop_growth_factor, epochs, mse, mae, r2\n")
        file.write(f'{counter},{batch_size},{input_dim},{lr},{hidden_dim},{prop_hidden_dim},{prop_pred_activation},{prop_pred_dropout},{prop_pred_depth},{prop_growth_factor},{epochs},{mse},{mae},{r2}\n')


def main():


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    list_of_properties = []

    input_dim = 50 ### Size of the latent space
    num_of_cycles = 1000### The number of hyperparameter searches you want to do, change accordingly
    counter = 0

###

    my_file = pd.read_csv('./datasets/PropsQM9/listprops.csv', index_col=None)##The file you want to train on, should contain SMILES reps, latent space reps and properties
    my_file.dropna()


    for i in range(len(my_file)):

        DipoleIn = my_file['dipole_moment'][i]
        GapIn = my_file['energy_gap'][i]
        SMILESCodeIn = my_file['smiles'][i]
        list_of_properties.append([SMILESCodeIn, GapIn, DipoleIn])

###


    folder_path = "./datasets/"
    file_name = "SelectedSMILES_QM9.txt"

    full_path = folder_path + file_name

    selfies_list, selfies_alphabet, largest_selfies_len, smiles_list, smiles_alphabet, largest_smiles_len = chemistry_vae_selfies.get_selfie_and_smiles_encodings_for_dataset(full_path)

    selfies_alphabet = ["[O]","[N]","[Branch1]","[F]","[=C]","[C]","[=N]","[=Branch1]","[Ring2]","[#Branch1]","[=Branch2]","[Branch2]","[Ring1]","[#Branch2]","[#C]","[nop]","[=O]","[#N]"]

###

    # define source file location
    file_to_load =  "../../selfies_saved_models_50/lr_0.002427375401692564_KLD_0.0364/"
    # training file name encoder
    training_file_nameE = "4950/E"
    # training file name decoder
    training_file_nameD = "4950/D"
    # load data
    #load_data_trained = file_to_load + training_file_nameE
    # Alphabet has 18 letters, largest molecule is 21 letters. (build this as an output function later ... )
    largest_selfies_len_dataset = largest_selfies_len
    largest_smiles_len_dataset = largest_smiles_len

    #in_dimension = len(selfies_alphabet)*largest_selfies_len
    in_dimension = len(smiles_alphabet)*largest_smiles_len

    # load the trained encoder
    vae_encoder = torch.load(file_to_load + training_file_nameE, map_location=torch.device(device="cpu"))
    #print(vae_encoder)

    # load the trained decoder
    vae_decoder = torch.load(file_to_load + training_file_nameD, map_location=torch.device(device="cpu"))
    #print(vae_decoder)
###



    selfies_rep, latent_rep, props_used = filter(list_of_properties, largest_selfies_len_dataset, selfies_alphabet, vae_encoder, vae_decoder)

    selfies_list, selfies_alphabet, largest_selfies_len, smiles_list, smiles_alphabet, largest_smiles_len = chemistry_vae_selfies.get_selfie_and_smiles_encodings_for_dataset(full_path)


    selfies_alphabet = ["[O]","[N]","[Branch1]","[F]","[=C]","[C]","[=N]","[=Branch1]","[Ring2]","[#Branch1]","[=Branch2]","[Branch2]","[Ring1]","[#Branch2]","[#C]","[nop]","[=O]","[#N]"]

    train_size = round(len(latent_rep)*0.8)

    one_hots =  create_onehot_instance_many(selfies_rep, largest_selfies_len, selfies_alphabet)

    energy_gaps = []
    #start_time = time.time()                                                                                                                                                                               


    latent_rep, log_var_rep = gen_new_latent_space(one_hots, largest_selfies_len, selfies_alphabet, vae_encoder, vae_decoder)






    log_vars_valid = [item[0].detach().squeeze(0) for item in log_var_rep]
    latent_space_vectors_valid = [item[0].detach().squeeze(0) for item in latent_rep]
    properties_training = [torch.tensor(property_vector) for property_vector in props_used]  # this will convert lists to tensors                                                                           


    log_var2 = [math.e ** (-exponent) for exponent in log_vars_valid]
    normalised_tensor_list = [min_max_normalize(tensor) for tensor in log_var2]




    for i in range(len(properties_training)):
        energy_gaps.append(props_used[i][0])




    #print(log_vars_valid.size())
    #print(latent_space_vectors_valid.size())



    log_var_rep_train = normalised_tensor_list[:train_size]
    log_var_rep_test = normalised_tensor_list[train_size:]



    latents_train = latent_space_vectors_valid[:train_size]
    latents_test = latent_space_vectors_valid[train_size:]

    props_train = energy_gaps[:train_size]
    props_test = energy_gaps[train_size:]
    #print("latents_train_len:", len(latents_train), "latent:", latents_train)


    
    


    train_latent_space_tensor, test_latent_space_tensor, train_properties_tensor, test_properties_tensor = make_tensors(latents_train, latents_test, props_train, props_test)

    log_var_rep_train = torch.stack(log_var_rep_train)
    log_var_rep_test = torch.stack(log_var_rep_test)





    train_latent_space_tensor = train_latent_space_tensor.view(-1, input_dim)
    test_latent_space_tensor = test_latent_space_tensor.view(-1, input_dim)
    log_var_rep_train_tensor = log_var_rep_train.view(-1, input_dim)
    log_var_rep_test_tensor = log_var_rep_test.view(-1, input_dim)


    print(log_var_rep_train_tensor.size())
    print(train_latent_space_tensor.size())

    train_data_with_variances = torch.cat((train_latent_space_tensor, log_var_rep_train_tensor), dim=1)


####

    for i in range(num_of_cycles):
            # Sample input dimensions for 100,000 compounds represented by 25 numbers
        hidden_dim = 1024 #random.choice([512, 1024, 2048])  # You can increase the hidden dimensions for more complexity
        prop_hidden_dim = int(hidden_dim/2)  # You can adjust this based on the complexity of the task
        prop_pred_activation = 'leaky_relu' #random.choice(['relu', 'leaky_relu'])
        prop_pred_dropout = random.choice([0, 0.1, 0.2])
        prop_pred_depth = random.choice([6,7,8])
        prop_growth_factor = 0.4#random.choice([0.4, 0.5, 0.6])
        lr_random = 0.0005#random.choice([0.001, 0.0005, 0.0001])
        batch_size = random.choice([64, 128, 256])#128  # You can adjust this based on your hardware and dataset size

        epochs = 150
        bestr2 = 0
        r2_list = []
        epoch_list = []

        absolute_average = torch.mean(torch.abs(log_var_rep_train), dim=0)

        var_tnsr = absolute_average**2
        print(var_tnsr.size())


        # Instantiate the model with hyperparameters
        model = PropertyRegressionModel(input_dim, hidden_dim, prop_hidden_dim, prop_pred_activation, prop_pred_dropout, prop_pred_depth, prop_growth_factor, var_tnsr)
        


        # Define your loss function (e.g., Mean Squared Error)
        loss_function = nn.MSELoss()

        # Define your optimizer (e.g., Adam)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_random)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)



        # Split the training data and properties into batches
        train_data_batches = train_latent_space_tensor.split(batch_size)
        train_props_batches = train_properties_tensor.split(batch_size)
        latent_variances_batch = log_var_rep_train_tensor.split(batch_size)
        print(len(latent_variances_batch[2][1]))

        


###

        for epoch in range(epochs):

            optimizer.zero_grad()
            predictions = model(train_latent_space_tensor)
            loss = loss_function(predictions.squeeze(), train_properties_tensor)
            loss.backward()
            optimizer.step()
            

            start_time = time.time()


            total_loss = 0.0

            # Iterate through batches
            for data_batch, props_batch, var_batch in zip(train_data_batches, train_props_batches, latent_variances_batch):

                # Concatenate the latent space tensor and variances tensor for this batch
                data_with_variances_batch = torch.cat((data_batch, var_batch), dim=1)

                optimizer.zero_grad()
                predictions = model(data_batch)  # Pass the combined tensor as input
                loss = loss_function(predictions.squeeze(), props_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                

            

                # Calculate average loss for the epoch
            avg_loss = total_loss / len(train_data_batches)

            end_time = time.time()
            epoch_duration = end_time - start_time

            print(f"Epoch {epoch}/{epochs} - Duration: {epoch_duration:.2f} seconds")
            print('loss:', avg_loss)
            epoch_list.append(epoch)
            scheduler.step(avg_loss)



###

            with torch.no_grad():
                test_predictions = model(test_latent_space_tensor)
            #print('test_predictions length:',len(test_predictions))
            #print('actual predictions length:',len(props_test))
            #test_loss = loss_function(test_predictions.squeeze(), test_properties_tensor)
            #print(f"Test Loss: {test_loss.item():.4f}")

###
            y_pred = test_predictions
            y_test = props_test

            mse, mae, r2 = stats(y_pred, y_test)
            r2_list.append(r2)
            print("Current r2:", r2, "Best r2:", bestr2)
            if r2 > bestr2:
                bestr2 = r2

                save_params(input_dim, lr_random, hidden_dim, prop_hidden_dim, prop_pred_activation, prop_pred_dropout, prop_pred_depth, prop_growth_factor, epochs, counter, mse, mae, r2, model, batch_size)
        counter = counter + 1








    
