import torch
import chemistry_vae_symmetric_rnn_final
import properties_searcher_2
import os
import yaml
import numpy as np
import selfies
import psutil
import pandas as pd
import time

from rdkit.Chem import rdFingerprintGenerator
from rdkit import Chem

def data_init(settings, device):
    save_path = settings['vae']['path']
    vae_epoch = settings['vae']['vae_epoch']
    dataset = settings['runs']['dataset']

    

    vae_settings = yaml.safe_load(open(str(save_path) + "settings/" + "settings.yml", "r"))
    encoder_parameter = vae_settings['encoder']
    decoder_parameter = vae_settings['decoder']
    selfies_alphabet = vae_settings['alphabet']

    encoder_weights_path = str(save_path) + str(vae_epoch) + "/E.pt"
    decoder_weights_path = str(save_path) + str(vae_epoch) + "/D.pt"

    encoding_list, _, largest_molecule_len, encoding_smiles, _, _ = chemistry_vae_symmetric_rnn_final.get_selfie_and_smiles_encodings_for_dataset(dataset)
    data = chemistry_vae_symmetric_rnn_final.multiple_selfies_to_hot(encoding_list, largest_molecule_len, selfies_alphabet)
    data = torch.tensor(data, dtype=torch.float).to(device)

    my_file = pd.read_csv(dataset, index_col=None)

    properties_df = my_file.drop(columns=['smiles']) ##drop all smiles from the properties df
    properties_array = properties_df.to_numpy() ##convert the df to numpy array
    properties_tensor = torch.tensor(properties_array,dtype=torch.float32).to('cpu')

    encoder_weights = torch.load(encoder_weights_path)
    decoder_weights = torch.load(decoder_weights_path)


    vae_encoder = chemistry_vae_symmetric_rnn_final.VAEEncoder(in_dimension=(data.shape[1]*data.shape[2]), **encoder_parameter).to(device)
    vae_decoder = chemistry_vae_symmetric_rnn_final.VAEDecoder(**decoder_parameter, out_dimension=len(selfies_alphabet)).to(device)


    vae_encoder.load_state_dict(encoder_weights)
    vae_decoder.load_state_dict(decoder_weights)

    

    

    return vae_encoder, vae_decoder, data, selfies_alphabet, encoding_smiles, properties_tensor



def optimiser(settings, data_train, vae_encoder, device, mlp):

    batch_size = settings['settings']['batch_size']
    num_batches_train = int(len(data_train) / batch_size)
    top_zs = torch.empty(0, dtype=torch.long).to('cpu')


    for batch_iteration in range(num_batches_train):


        z_tensor = torch.empty(0, dtype=torch.float32).to('cpu')

        start_idx = batch_iteration * batch_size
        stop_idx = (batch_iteration + 1) * batch_size
        batch = data_train[start_idx: stop_idx].to(device)

        inp_flat_one_hot = batch.flatten(start_dim=1)
        inp_flat_one_hot = inp_flat_one_hot.unsqueeze(0).to(device)  

        del batch 


        for xo in range(1000):
            vae_encoder.eval()
            with torch.no_grad():
                _,mu,log_var = vae_encoder(inp_flat_one_hot)
            

            std = torch.exp(0.5 * log_var).to(device)
            variation_tensor = torch.randn_like(std)
            z = variation_tensor.mul(2*std).add_(mu).to('cpu')

            del variation_tensor
            del std

            z = z.unsqueeze(0)

            z_tensor = torch.cat((z_tensor, z), dim = 0)


        del inp_flat_one_hot



        for i in range(z_tensor.shape[1]):
            
            current_z = z_tensor[:,i,:].to(device)

            with torch.no_grad():
                predictions = mlp(current_z)
            _, indices = torch.topk(predictions.squeeze(), k=10, largest=False)

            indices = indices.unsqueeze(0)
            top_zs = torch.cat((top_zs, z_tensor[indices, i, :]), dim = 1)
        del z_tensor



    return top_zs

def smiles_generation(top_lvecs, selfies_alphabet, vae_decoder, data_s1,  device, settings):

    final_smiles_list = []
    top_lvecs = top_lvecs.squeeze()

    batch_size = settings['settings']['batch_size']
    num_batches_train = int(len(top_lvecs) / batch_size)

    for batch_iteration in range(num_batches_train):

        start_idx = batch_iteration * batch_size
        stop_idx = (batch_iteration + 1) * batch_size
        batch = top_lvecs[start_idx: stop_idx].to(device)




        seq_tensor = torch.empty(0, dtype=torch.float32).to(device)
        hidden = vae_decoder.init_hidden(batch_size=batch.shape[0])

        

        for seq_index in range(data_s1):
            start_time = time.time()
            vae_decoder.eval()
            with torch.no_grad():

                out_one_hot_line, hidden = vae_decoder(batch.unsqueeze(0), hidden)
                out_one_hot_line = out_one_hot_line.squeeze()
                out_one_hot_line_arg = torch.argmax(out_one_hot_line, dim = 1).unsqueeze(0)
                seq_tensor = torch.cat((seq_tensor, out_one_hot_line_arg), dim = 0)


        sequences = seq_tensor.squeeze().t()
        list_of_continuous_strings = [selfies.decoder(''.join([selfies_alphabet[int(i)] for i in row])) for row in sequences]
        final_smiles_list = final_smiles_list + list_of_continuous_strings

    return final_smiles_list


def comparison(valid_smiles, smiles_list):

    filtered_list = [x for x in smiles_list if x is not None]
    generated_list = list(set(filtered_list))


    fpgen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=4096)

    valid_fp = [fpgen.GetFingerprint(Chem.MolFromSmiles(x)).ToBitString() for x in valid_smiles]
    gen_fp = [fpgen.GetFingerprint(Chem.MolFromSmiles(x)).ToBitString() for x in generated_list]
    gen_fp2 = [fpgen.GetFingerprint(Chem.MolFromSmiles(x)).ToBitString() for x in smiles_list]


    mask = np.in1d(gen_fp, valid_fp)
    similar_fingerprints = mask*1
    sum_duplicates = np.sum(similar_fingerprints)

    common_values = set(valid_fp) & set(gen_fp2)

    indices_list1 = []
    indices_list2 = []

    if common_values:
        for value in common_values:
            index_list1 = next((index for index, element in enumerate(valid_fp) if element == value), None)
            index_list2 = next((index for index, element in enumerate(gen_fp2) if element == value), None)
        
            if index_list1 is not None:
                indices_list1.append(index_list1)
            if index_list2 is not None:
                indices_list2.append(index_list2)


    validation_mask = torch.tensor(indices_list1).squeeze()
    generated_mask = torch.tensor(indices_list2)

    valids_number = sum_duplicates/len(valid_smiles)

    return valids_number, validation_mask, generated_mask

def top_comparison(mlp, validation_mask, generated_mask, top_lvecs, top_smiles, top_properties, device):

    top_lvecs = top_lvecs.squeeze().to(device)

    top_smiles_list = [top_smiles[i] for i in validation_mask]
    top_properties = top_properties[validation_mask]
    prop_list = [i.item() for i in top_properties]

    l_vecs = top_lvecs[generated_mask].squeeze()

    with torch.no_grad():
        predictions = mlp(l_vecs)

    data = {'smiles': top_smiles_list, 'true_value': prop_list, 'prediction': predictions}
    df = pd.DataFrame(data)
    
    df.to_csv('top_mols.csv', index = False)









def main():
    if os.path.exists("brute_force.yml"):
        settings = yaml.safe_load(open("brute_force.yml", "r"))
    else:
        print("Expected a file settings.yml but didn't find it.")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_seed = settings['settings']['seed']
    vae_encoder, vae_decoder, data, selfies_alphabet, smiles_list, properties_tensor = data_init(settings, device)

    mlp = properties_searcher_2.PropertyRegressionModel(64, 256, 'silu', 0.07, 4, 0.65).to(device)
    mlp.load_state_dict(torch.load('../VAE_sandbox/perceptron_folder/mlp/models/0.9402734041213989.pt'))
    
    #shuffle data according to seed

    torch.manual_seed(torch_seed)
    rand_perms = torch.randperm(data.size()[0])
    data = data[rand_perms]
    properties_tensor = properties_tensor[rand_perms]

    data_s1 = data.shape[1]
    shuffled_smiles = [smiles_list[i] for i in rand_perms]


    #split data into train and valid sets

    train_valid_test_size = [0.8, 0.2, 0.0]

    idx_train_val = int(len(data) * train_valid_test_size[0])
    idx_val_test = idx_train_val + int(len(data) * train_valid_test_size[1])

    data_train = data[0:idx_train_val]
    smiles_valid = shuffled_smiles[idx_train_val:idx_val_test]
    properties_valid = properties_tensor[idx_train_val:idx_val_test]

    del data
    del properties_tensor

    _, properties_indices = torch.topk(properties_valid.squeeze(), k=100, largest=False)


    top_valid_smiles = [smiles_valid[i] for i in properties_indices]



    best_lvecs = optimiser(settings, data_train, vae_encoder, device, mlp)

    generated_smiles = smiles_generation(best_lvecs, selfies_alphabet, vae_decoder, data_s1, device, settings)

    smiles_data = {'smiles': generated_smiles}
    df = pd.DataFrame(smiles_data)


    df.to_csv('top_smiles.csv', index = False)

    valids_number, validation_mask, generated_mask = comparison(top_valid_smiles, generated_smiles)

    top_comparison(mlp, validation_mask, generated_mask, best_lvecs, top_valid_smiles, properties_valid, device)



    print('ratio of generated things in the validation set:', valids_number)



    