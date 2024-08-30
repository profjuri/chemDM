import torch
import os
import sys
import yaml
import numpy as np
import selfies
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
import time
import subprocess

import mlp
import vae_old
import functions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import RDConfig

fpgen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=4096)

sys.path.insert(1, '../chem_stuff/rdkit/Contrib/SA_Score')
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer
from rdkit import RDLogger 

torch.set_grad_enabled(False)
RDLogger.DisableLog('rdApp.*') 


def gpu_memory():
    gpu_memory = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert bytes to GB
    gpu_memory_reserved = torch.cuda.memory_reserved() / (1024 ** 3)  # Reserved memory
    gpu_total_memory = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)

    print(f"GPU Memory Usage: {gpu_memory:.2f} GB (Allocated), {gpu_memory_reserved:.2f} GB (Reserved)")


def get_models(settings):

    vae_save_path = settings['vae']['save_path']
    vae_epoch = settings['vae']['epoch']
    vae_settings = yaml.safe_load(open(str(vae_save_path) + "settings/" + "settings.yml", "r"))
    encoder_parameter = vae_settings['encoder']
    decoder_parameter = vae_settings['decoder']
    selfies_alphabet = vae_settings['alphabet']
    torch_seed = vae_settings['data']['torch_seed']

    encoder_dict_path = str(vae_save_path) + str(vae_epoch) + "/E.pt"
    decoder_dict_path = str(vae_save_path) + str(vae_epoch) + "/D.pt"
    encoder_dict = torch.load(encoder_dict_path, map_location = device)
    decoder_dict = torch.load(decoder_dict_path, map_location = device)
    vae_encoder = vae_old.VAEEncoder(in_dimension=(encoder_dict['encode_RNN.weight_ih_l0'].shape[1]), **encoder_parameter)
    vae_decoder = vae_old.VAEDecoder(**decoder_parameter, out_dimension=len(selfies_alphabet))
    vae_encoder.load_state_dict(encoder_dict)
    vae_decoder.load_state_dict(decoder_dict)


    mlp_save_path = settings['mlp']['save_path']
    mlp_epoch = settings['mlp']['epoch']

    mlp_settings = yaml.safe_load(open(str(mlp_save_path) + "settings/" + "settings.yml", "r"))
    model_params = mlp_settings['model_params']
    model_params['input_dim'] = vae_encoder.encode_FC_mu[0].weight.shape[0]
    mlp_model = mlp.PropertyRegressionModel(**model_params)
    state_dict = torch.load(mlp_save_path + '/' + str(mlp_epoch) + '/model.pt', map_location=device)
    mlp_model.load_state_dict(state_dict)

    z_mlp_save_path = settings['z_mlp']['save_path']
    z_mlp_epoch = settings['z_mlp']['epoch']

    z_mlp_settings = yaml.safe_load(open(str(z_mlp_save_path) + "settings/" + "settings.yml", "r"))
    model_params = z_mlp_settings['model_params']
    model_params['input_dim'] = vae_encoder.encode_FC_mu[0].weight.shape[0]
    z_model = mlp.PropertyRegressionModel(**model_params)
    state_dict = torch.load(z_mlp_save_path + '/' + str(z_mlp_epoch) + '/model.pt', map_location=device)
    z_model.load_state_dict(state_dict)


    vae_encoder.eval()
    vae_decoder.eval()
    mlp_model.eval()
    z_model.eval()

    vae_encoder.to(device)
    vae_decoder.to(device)
    mlp_model.to(device)
    z_model.to(device)


    largest_molecule_len = int(vae_encoder.encode_RNN.weight_ih_l0.shape[1]/len(selfies_alphabet))
    lpoint_size = decoder_dict['decode_RNN.weight_ih_l0'].shape[1]

    return vae_encoder, vae_decoder, mlp_model, z_model, selfies_alphabet, largest_molecule_len, lpoint_size

def generate_normal_tensor(z_num, lpoint_size, bottom, top):
    eps = torch.rand(z_num, lpoint_size)
    eps = eps * (top - bottom)
    eps = eps + bottom
    mask = (torch.rand_like(eps) > 0.5).detach()
    eps[mask] = -eps[mask].detach()
    
    return eps

def save_smiles(generated_smiles, predictions, sascores, settings):

    save_path = settings['data']['save_path']
    csv_file_name = 'bf_results.csv'
    csv_file_path = save_path + csv_file_name

    df = pd.DataFrame({'smiles': generated_smiles, 'predictions': predictions, 'SAS': sascores})

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        df.to_csv(csv_file_path, mode='a', header=True, index=False)

        settings_filepath = os.path.join(csv_file_path, 'settings.yml')

        data = {**settings}
        with open(settings_filepath, 'w') as file:
            yaml.dump(data, file)
    else:
        df.to_csv(csv_file_path, mode='a', header=False, index=False)

def threshold_refiner(selfies_list, prediction, condition, imp_props):
     
    mask = condition(prediction)
    threshold_indices = mask.squeeze().nonzero().squeeze().tolist()

    if type(threshold_indices) == int:
        threshold_indices = [threshold_indices]

    top_predictions = prediction[threshold_indices]
    imp_props = imp_props[threshold_indices]
    top_selfies = [selfies_list[x] for x in threshold_indices]

    if len(top_selfies) > 0:
        top_smiles = [selfies.decoder(x) for x in top_selfies]
    else:
        top_smiles = []

    return top_selfies, top_smiles, top_predictions, imp_props


def try_encode(x):
    try:
        return selfies.encoder(x)
    except Exception:
        return None
    

def get_run_log_list(settings):

    if os.path.exists(settings['data']['save_path'] + '/index_list.txt'):
        log_txt = pd.read_csv(settings['data']['save_path'] + '/index_list.txt')
        run_log_list = log_txt['log']
        run_log_list = run_log_list.astype(int)
        run_log_list = run_log_list.tolist() #.astype(int).tolist()
        
    else:
        run_log_list = [0]


    return run_log_list

def save_index(index, settings):


    save_path = settings['data']['save_path']
    log_filename = 'index_list.txt'

    log_filepath = os.path.join(save_path, log_filename)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    file_exists = os.path.isfile(log_filepath)

    with open(log_filepath, 'a') as file:
        if not file_exists:
            file.write("log\n")
        file.write(f'{index}\n')


def unique_values_and_first_indices(LIS):
    unique_values = []
    first_indices = []
    seen = {}

    for i, num in enumerate(LIS):
        if num not in seen:
            seen[num] = i
            unique_values.append(num)
            first_indices.append(i)
    
    return unique_values, first_indices

def non_none_values_and_indices(LIS):
    non_none_values = []
    non_none_indices = []

    for i, value in enumerate(LIS):
        if value is not None:
            non_none_values.append(value)
            non_none_indices.append(i)
    
    return non_none_values, non_none_indices






def bf(vae_encoder, vae_decoder, mlp_model, z_model, selfies_alphabet, largest_molecule_len, lpoint_size, encoding_list, settings):

    time0 = time.time()

    tail_end = settings['data']['tail_end']
    threshold = settings['data']['mu_threshold']
    z_scale = settings['data']['z_scale']
    lower_sigma = settings['data']['lower_sigma']
    upper_sigma = settings['data']['upper_sigma']
    sas_threshold = settings['data']['sas_threshold']
    z_num = settings['data']['z_num']


    eps = generate_normal_tensor(z_num, lpoint_size, lower_sigma, upper_sigma).detach()

    condition = lambda y: ('['+y+']' in set(selfies_alphabet))

    if tail_end == 0:
        condition2 = lambda z: z < threshold
        z_threshold = z_scale * threshold
        condition3 = lambda z: z < z_threshold
    else:
        condition2 = lambda z: z > threshold
        z_threshold = threshold / z_scale
        condition3 = lambda z: z > z_threshold

    condition4 = lambda z: z < sas_threshold

    counter = 0

    time_0 = time.time()

    tensor_memory = (z_num * lpoint_size) * 4
    string_memory = z_num
    max_memory_consumption = 3 * tensor_memory + 2 * string_memory

    run_list = get_run_log_list(settings)
    smiles_index= max(run_list)
    print('Starting SMILES index:', smiles_index)



    while smiles_index < len(encoding_list):


        #smiles_index = get_new_id(settings, username)

        if smiles_index % 100 ==0:
            print('SMILES:', smiles_index, '/', len(encoding_list), 'Time taken:', time.time() - time_0)
            time_0 = time.time()

        with torch.no_grad():

            time_1 = time.time()


            _, mu, log_var = functions.selfies_to_all([encoding_list[smiles_index]], selfies_alphabet, largest_molecule_len, vae_encoder, lpoint_size)
            log_var = log_var
            std = torch.exp(0.5 * log_var)
            std_repeat = std.squeeze().repeat(z_num, 1)
            del std
            mus_repeat = mu.squeeze().repeat(z_num, 1)
            del mu
            zs = eps.mul(std_repeat).add_(mus_repeat).to(device)

            time_2 = time.time()
            #print('Time 2:', time_2 - time_1)
            
            del std_repeat
            del mus_repeat
            del log_var
            z_model.eval()
            pre_list = []
            num = 10
            time1 = time.time()

            '''
            for x in range(num):
                start_idx = x * int(zs.shape[0]/num)
                stop_idx = (x+1) * int(zs.shape[0]/num)
                current_zs = zs[start_idx:stop_idx]
                predictions = z_model(current_zs).detach()
                pre_list.append(predictions)
            predictions = torch.cat(pre_list)
            '''
            
            
            
            predictions= z_model(zs).detach()


            mask = condition3(predictions)

            time_3 = time.time()
            #print('Time 3:', time_3 - time_2)

            del predictions
            torch.cuda.empty_cache()


            threshold_indices = mask.squeeze().nonzero().squeeze().tolist()
            del mask
            if type(threshold_indices) == int:
                threshold_indices = [threshold_indices]

            time_4 = time.time()
            #print('Time 4:', time_4 - time_3)

            if len(threshold_indices) > 0:  

                  
                        
                zs = zs[threshold_indices]

                

                time_5 = time.time()
                #print('Time 5:', time_5 - time_4)

                if zs.shape[0] > 0:

                    scattered_one_hots = functions.lpoints_to_onehots(zs, selfies_alphabet, vae_decoder, largest_molecule_len)
                    del zs
                    lpoints_decoded = functions.onehots_to_lpoints(scattered_one_hots, vae_encoder).to(device)

                    mlp_model.eval()
                    predictions = mlp_model(lpoints_decoded)


                    index_tensor = condition2(predictions).squeeze().nonzero().squeeze().tolist()
                    if type(index_tensor) == int:
                        index_tensor = [index_tensor]


                    if len(index_tensor) > 0:

                        lpoints_decoded = lpoints_decoded[index_tensor]
                        predictions = predictions[index_tensor]
                        scattered_smiles = functions.decode_lpoints(lpoints_decoded, selfies_alphabet, vae_decoder, largest_molecule_len)
                        del lpoints_decoded


                        scattered_smiles, first_indices = unique_values_and_first_indices(scattered_smiles)
                        predictions = predictions[first_indices]


                        scattered_mols = [Chem.MolFromSmiles(x) for x in scattered_smiles]
                        _, non_none_indices = non_none_values_and_indices(scattered_mols)


                        scattered_smiles = [scattered_smiles[x] for x in non_none_indices]
                        predictions = predictions[non_none_indices]

                        try_encode_list = [try_encode(x) for x in scattered_smiles]
                        _, non_none_indices = non_none_values_and_indices(try_encode_list)

                        scattered_smiles = [scattered_smiles[x] for x in non_none_indices]
                        predictions = predictions[non_none_indices]

                        


                        scatterd_selfies = [selfies.encoder(x) for x in scattered_smiles]
                        top_selfies_index = [x for x in range(len(scatterd_selfies)) if all(condition(element) for element in scatterd_selfies[x][1:-1].split("]["))]
                        top_selfies = [scatterd_selfies[x] for x in top_selfies_index]
                        top_smiles = [scattered_smiles[x] for x in top_selfies_index]
                        top_predictions = predictions[top_selfies_index]
                        del predictions


                        top_smiles = [selfies.decoder(x) for x in top_selfies]

                        if len(top_smiles) > 0:


                            sascores = [sascorer.calculateScore(Chem.MolFromSmiles(x)) for x in top_smiles]
                            sascores = torch.tensor(sascores)

                            _, top_smiles, sascores, top_predictions = threshold_refiner(top_selfies, sascores, condition4, top_predictions) ### returns SMILES meeting the SA score threshold


                            if len(top_smiles) > 0:


                                time_6 = time.time()
                                #print('Time 6:', time_6 - time_5)


                                save_smiles(top_smiles, top_predictions.squeeze().tolist(), sascores.squeeze().tolist(), settings)
                    


        save_index(smiles_index, settings)
        smiles_index +=1
        counter +=1

def main():
    settings = yaml.safe_load(open("bf_settings.yml", "r"))
    smiles_path = settings['data']['smiles_path']

    vae_encoder, vae_decoder, mlp_model, z_model, selfies_alphabet, largest_molecule_len, lpoint_size = get_models(settings)
    encoding_list, _, _ = functions.get_selfie_and_smiles_encodings_for_dataset(smiles_path)

    bf(vae_encoder, vae_decoder, mlp_model, z_model, selfies_alphabet, largest_molecule_len, lpoint_size, encoding_list, settings)


