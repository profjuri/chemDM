import os
import pandas as pd
import time
import yaml

from chemdm.vae import chemistry_vae_symmetric_rnn_OG

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

class PropertyClassificationModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, prop_hidden_dim, prop_pred_depth, prop_growth_factor, num_classes):
        super(PropertyClassificationModel, self).__init__()

        self.ls_in = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList()
        self.batchnorm_first = nn.BatchNorm1d(hidden_dim)

        # Modify output layer for binary classification (num_classes = 2)
        self.classification = nn.Linear(prop_hidden_dim, num_classes)

    # Modify forward method for binary classification
    def forward(self, x):
        x = self.ls_in(x)
        x = self.batchnorm_first(x)

        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)
            if isinstance(layer, nn.Linear) and self.dropout.p > 0:
                x = self.dropout(x)

        # Output logits for binary classification (no activation)
        class_logits = self.classification(x)
        
        # Apply sigmoid activation to obtain class probabilities
        class_probabilities = torch.sigmoid(class_logits)
        return class_probabilities

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

def stats(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, mae, r2

def save_params(input_dim, lr, hidden_dim, prop_hidden_dim, prop_pred_activation, prop_pred_dropout, prop_pred_depth, prop_growth_factor, epochs, counter, mse, mae, r2, model, batch_size, loss_choice, factor_choice, weight_choice, patience_choice, lambda_choice, settings):
    out_dir = settings['settings']['output_folder']
    log_folder = out_dir  # Replace with the desired folder path
    log_filename = 'results.txt'

    log_filepath = os.path.join(log_folder, log_filename)
    #torch.save(model, str(out_dir) + str(r2))
    # Create the log folder if it doesn't exist
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    
    torch.save(model, out_dir + '/' + str(r2))

    file_exists = os.path.isfile(log_filepath)
    
    with open(log_filepath, 'a') as file:
        if not file_exists:
            file.write("counter, lr, batch_size, input_dim, hidden_dim, prop_hidden_dim, prop_pred_activation, prop_pred_dropout, prop_pred_depth, prop_growth_factor, epochs, loss_choice, factor_choice, patience_choice,weight_choice,lambda_choice, mse, mae, r2\n")
        file.write(f'{counter},{lr},{batch_size},{input_dim},{hidden_dim},{prop_hidden_dim},{prop_pred_activation},{prop_pred_dropout},{prop_pred_depth},{prop_growth_factor},{epochs},{loss_choice},{factor_choice},{patience_choice},{weight_choice},{lambda_choice},{mse},{mae},{r2}\n')

def save_r2_loss(epoch, r2, train_r2, loss, settings):

    out_dir = settings['settings']['output_folder']
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

def main():
    if os.path.exists("classifier.yml"):
        settings = yaml.safe_load(open("classifier.yml", "r"))
    else:
        print("Expected a file settings.yml but didn't find it.")
        return


    input_dim = settings['settings']['input_dim']
    smiles_file = settings['settings']['smiles_file']
    vae_file = settings['settings']['vae_file']
    vae_epoch = settings['settings']['vae_epoch']
    selfies_alphabet = settings['settings']['selfies_alphabet']
    output_folder = settings['settings']['output_folder']
    num_of_cycles = settings['settings']['num_of_cycles']

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

    
    counter = 0

    encoding_list, encoding_alphabet, largest_molecule_len, _, _, _ = chemistry_vae_symmetric_rnn_OG.get_selfie_and_smiles_encodings_for_dataset(smiles_file)


    data = chemistry_vae_symmetric_rnn_OG.multiple_selfies_to_hot(encoding_list, largest_molecule_len, selfies_alphabet)

###

    file_to_load =  str(vae_file)
    training_file_nameE = str(vae_epoch) + "/E"
    training_file_nameD = str(vae_epoch) + "/D"

    vae_encoder = torch.load(file_to_load + training_file_nameE, map_location=device)

###
    train_valid_test_size = [0.8, 0.2, 0.0]
    idx_train_val = int(len(data) * train_valid_test_size[0])
    idx_val_test = idx_train_val + int(len(data) * train_valid_test_size[1])

    data = torch.tensor(data, dtype=torch.float).to(device)

    inp_flat_one_hot = data.flatten(start_dim=1)
    inp_flat_one_hot = inp_flat_one_hot.unsqueeze(0)



    latent_points, mus, log_vars = vae_encoder(inp_flat_one_hot)



    lpoints_train = mus[0:idx_train_val]
    lpoints_valid = mus[idx_train_val:idx_val_test]

    num_batches_train = int(len(lpoints_train) / batch_size)

###

    std = torch.exp(0.5 * log_vars)
    normalised_tensor_list = min_max_normalize(std)

    log_var_rep_train = normalised_tensor_list[0:idx_train_val]
    log_var_rep_test = normalised_tensor_list[idx_train_val:idx_val_test]

####

    my_file = pd.read_csv(smiles_file, index_col=None)##The file you want to train on, should contain SMILES reps, latent space reps and properties


    properties_df = my_file.drop(columns=['smiles']) ##drop all smiles from the properties df
    properties_array = properties_df.to_numpy() ##convert the df to numpy array
    properties_tensor = torch.tensor(properties_array,dtype=torch.float32)
    properties_tensor = torch.stack([(1 - properties_tensor), properties_tensor], dim=1).squeeze()


    train_properties_tensor = properties_tensor[0:idx_train_val]
    valid_properties_tensor = properties_tensor[idx_train_val:idx_val_test]

    print('test:', properties_tensor[1])




    for i in range(num_of_cycles):

        epochs = epochs
        bestr2 = 0
        r2_list = []
        epoch_list = []

        absolute_average = torch.mean(torch.abs(log_var_rep_train), dim=0)


        var_tnsr = absolute_average**2



        # Instantiate the model with hyperparameters
        model = PropertyClassificationModel(input_dim, hidden_dim, prop_hidden_dim, prop_pred_depth, prop_growth_factor, 2).to(device)
        print('model:', model)


        lambda_choice = 0 

        loss_function = nn.BCEWithLogitsLoss()


        print("Memory allocated (bytes):", torch.cuda.memory_allocated())
        print("Memory reserved (bytes):", torch.cuda.memory_reserved())

        

        # Define your optimizer (e.g., Adam)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_choice)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=learning_rate_factor, patience=learning_rate_patience, verbose=True)

        lpoints_train2 = lpoints_train.detach().to(device)
        train_properties_tensor2 = train_properties_tensor.detach().to(device)

        train_data_batches = lpoints_train2.split(batch_size)
        train_props_batches = train_properties_tensor2.split(batch_size)

###

        for epoch in range(epochs):

            start_time = time.time()


            total_loss = 0.0
            r2_train = 0.0

            # Iterate through batches
            for train_data_batch, train_props_batch in zip(train_data_batches, train_props_batches):

                optimizer.zero_grad()
                
                predictions = model(train_data_batch)
                #print('predictions:', predictions, 'train_props_batch:', train_props_batch)
                print(predictions.size(), train_props_batch.squeeze().size())


                loss = loss_function(predictions, train_props_batch)

                print('loss:',loss)

                loss.backward()
                optimizer.step()
                
                y_pred = predictions.detach().cpu().numpy() 
                y_test = train_props_batch.detach().cpu().numpy()

                mse, mae, r2 = stats(y_test, y_pred) 

                total_loss += loss.item()
                r2_train += r2


                # Calculate average loss for the epoch
            avg_loss = total_loss / num_batches_train
            avg_r2 = r2_train / num_batches_train

            end_time = time.time()
            epoch_duration = end_time - start_time

            print(f"Epoch {epoch}/{epochs} - Duration: {epoch_duration:.2f} seconds")
            print('loss:', avg_loss)
            epoch_list.append(epoch)
            scheduler.step(loss)



###

            with torch.no_grad():
                test_predictions = model(lpoints_valid.squeeze())


###
            y_pred = test_predictions.cpu().numpy() 
            y_test = valid_properties_tensor.cpu().numpy() 


            mse, mae, r2 = stats(y_test, y_pred)
            r2_list.append(r2)
            print("Current r2:", r2, "Best r2:", bestr2)
            if r2 > bestr2+0.01:
                bestr2 = r2

                save_params(input_dim, learning_rate, hidden_dim, prop_hidden_dim, prop_pred_activation, prop_pred_dropout, prop_pred_depth, prop_growth_factor, epochs, counter, mse, mae, r2, model, batch_size, loss_choice, learning_rate_factor, weight_choice, learning_rate_patience, lambda_choice, settings)
            save_r2_loss(epoch, r2, avg_r2, avg_loss, settings)
        counter = counter + 1