import selfies
import torch
import chemistry_vae_symmetric_rnn_final
import time
import pandas as pd
import os
import yaml
import numpy as np

import deepchem as dc
from rdkit.Chem import rdFingerprintGenerator
from rdkit import Chem

def save_params(deviation, gen_target, gen_input, target_percent, input_percent, new_percent, total_input_fraction, unique_ratio, unique_elements_len, settings, long_ratio):

    out_dir = settings['runs']['output_folder']
    log_folder = out_dir  # Replace with the desired folder path
    log_filename = settings['runs']['name']

    log_filepath = os.path.join(log_folder, log_filename)

    # Create the log folder if it doesn't exist
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    file_exists = os.path.isfile(log_filepath)

    
    with open(log_filepath, 'a') as file:
        if not file_exists:
            file.write("deviation,gen_target,gen_input,target_percent,input_percent,total_input_fraction,unique_ratio,unique_elements_len,long_ratio,new_percent\n")
        file.write(f'{deviation},{gen_target},{gen_input},{target_percent},{input_percent},{total_input_fraction},{unique_ratio},{unique_elements_len},{long_ratio},{new_percent}\n')


def data_init(settings, device):

    save_path = settings['vae']['path']
    vae_epoch = settings['vae']['vae_epoch']

    dataset = settings['runs']['dataset']
    target = settings['runs']['target']

    vae_settings = yaml.safe_load(open(str(save_path) + "settings/" + "settings.yml", "r"))
    encoder_parameter = vae_settings['encoder']
    decoder_parameter = vae_settings['decoder']
    selfies_alphabet = vae_settings['alphabet']

    encoder_weights_path = str(save_path) + str(vae_epoch) + "/E.pt"
    decoder_weights_path = str(save_path) + str(vae_epoch) + "/D.pt"


    _, _, _, target_list, _, _ = chemistry_vae_symmetric_rnn_final.get_selfie_and_smiles_encodings_for_dataset(target)
    encoding_list, _, largest_molecule_len, encoding_smiles, _, _ = chemistry_vae_symmetric_rnn_final.get_selfie_and_smiles_encodings_for_dataset(dataset)
    data = chemistry_vae_symmetric_rnn_final.multiple_selfies_to_hot(encoding_list, largest_molecule_len, selfies_alphabet)
    data = torch.tensor(data, dtype=torch.float).to(device)

    inp_flat_one_hot = data.flatten(start_dim=1)
    inp_flat_one_hot = inp_flat_one_hot.unsqueeze(0) 


    encoder_weights = torch.load(encoder_weights_path)
    decoder_weights = torch.load(decoder_weights_path)


    vae_encoder = chemistry_vae_symmetric_rnn_final.VAEEncoder(in_dimension=(data.shape[1]*data.shape[2]), **encoder_parameter).to(device)
    vae_decoder = chemistry_vae_symmetric_rnn_final.VAEDecoder(**decoder_parameter, out_dimension=len(selfies_alphabet)).to(device)


    vae_encoder.load_state_dict(encoder_weights)
    vae_decoder.load_state_dict(decoder_weights)



    return vae_encoder, vae_decoder, inp_flat_one_hot, target_list, encoding_smiles, encoding_list, data.shape[1], selfies_alphabet



def decode(final_smiles_list, device, vae_decoder, latent_vecs, data_1, selfies_alphabet):

    seq_tensor = torch.empty(0, dtype=torch.float32).to(device)
    hidden = vae_decoder.init_hidden(batch_size=latent_vecs.shape[0])


    for seq_index in range(data_1):

        out_one_hot_line, hidden = vae_decoder(latent_vecs.unsqueeze(0), hidden)
        out_one_hot_line = out_one_hot_line.squeeze()
        out_one_hot_line_arg = torch.argmax(out_one_hot_line, dim = 1).unsqueeze(0)
        seq_tensor = torch.cat((seq_tensor, out_one_hot_line_arg), dim = 0)

    sequences = seq_tensor.squeeze().t()
    list_of_continuous_strings = [selfies.decoder(''.join([selfies_alphabet[int(i)] for i in row])) for row in sequences]
    final_smiles_list = final_smiles_list + list_of_continuous_strings
    
    return final_smiles_list

def gen_numbers(settings):

    start = settings['runs']['start']
    end = settings['runs']['end']
    step = settings['runs']['step']

    numbers = np.arange(start, end + step, step).tolist()
    number_list = [round(x, 2) for x in numbers]

    return number_list

def fingerprint_compare(fpgen, generated_fingerprints, list2):


    fingerprints_2 = [fpgen.GetFingerprint(Chem.MolFromSmiles(x)).ToBitString() for x in list2]

    similar_fingerprints = np.in1d(generated_fingerprints, fingerprints_2)
    similar_fingerprints = similar_fingerprints*1
    sum_duplicates = np.sum(similar_fingerprints)









    ratio = sum_duplicates/len(list2)
    gen_ratio = sum_duplicates/len(generated_fingerprints)

    return ratio, gen_ratio

def gen_list_processing(fpgen, generated_list):

    generated_fingerprints = []

    for x in generated_list:

        try:
            mol = Chem.MolFromSmiles(x)
            fingerprint = fpgen.GetFingerprint(mol).ToBitString()
            generated_fingerprints.append(fingerprint)
        except:
            pass


    #generated_fingerprints = [fpgen.GetFingerprint(Chem.MolFromSmiles(x)).ToBitString() for x in generated_list]
    _, idx = np.unique(generated_fingerprints, return_index=True)
    generated_fingerprints = np.unique(generated_fingerprints)

    un_selfies = [generated_list[x] for x in idx]
    atoms = {"N", "F", "O", "C"}
    counts = [sum(selfie.count(atom) for atom in atoms) for selfie in un_selfies]
    long_smiles = [un_selfies[i] for i in range(len(un_selfies)) if counts[i] > 9]

    long_ratio = len(long_smiles)/len(generated_fingerprints)

    return generated_fingerprints, long_ratio

def remove_unrecognized_symbols(smiles_list):

    '''Removes blank spaces from the SMILES encodings'''

    '''Arguments:
                    smiles_list: the list of SMILES encodings (list)'''
    
    '''Outputs:
                    cleaned_smiles: the cleaned SMILES encodings list (list)'''

    cleaned_smiles = [smiles.replace('\n', '') for smiles in smiles_list]

    return cleaned_smiles



def main():
    if os.path.exists("check_for_long.yml"):
        settings = yaml.safe_load(open("check_for_long.yml", "r"))
    else:
        print("Expected a file settings.yml but didn't find it.")
        return
    
    runs_coeff = settings['runs']['num']
    out_path = settings['runs']['out_path']
    dir_name = settings['runs']['dir_name']

    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae_encoder, vae_decoder, inp_flat_one_hot, target_list, encoding_smiles, encoding_list, data_s1, selfies_alphabet = data_init(settings, device)
    list_of_numbers = gen_numbers(settings)

    runs = runs_coeff * round((len(target_list)+ len(encoding_list))/len(encoding_list))


    for x in range(len(list_of_numbers)):
        deviation = list_of_numbers[x]
        final_smiles_list = []


        for y in range(runs):

            _, mu, log_var = vae_encoder(inp_flat_one_hot)
            std = torch.exp(0.5 * log_var).to(device)

            #variation_tensor = torch.tensor([[random.uniform(-deviation, deviation) for _ in range(std.shape[1])] for _ in range(std.shape[0])], dtype=torch.float32).to(device)
            variation_tensor = torch.randn_like(std)
            latent_vecs = variation_tensor.mul(deviation*std).add_(mu)
            
            #latent_vecs = mu + variation_tensor * std

            start_time = time.time()

            final_smiles_list = decode(final_smiles_list, device, vae_decoder, latent_vecs, data_s1, selfies_alphabet)



        filtered_list = [x for x in final_smiles_list if x is not None]
        generated_list = list(set(filtered_list))
        generated_list = list(filter(None, generated_list))


        #generated_list = remove_unrecognized_symbols(generated_list)
        fpgen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=4096)


        generated_fingerprints, long_ratio = gen_list_processing(fpgen, generated_list)



        gen_target, target_percent = fingerprint_compare(fpgen, generated_fingerprints, target_list)
        gen_input, input_percent = fingerprint_compare(fpgen, generated_fingerprints, encoding_smiles)
        new_percent = 1 - (target_percent + input_percent + long_ratio)
        total_input_fraction = len(encoding_smiles)/(len(target_list)+len(encoding_smiles))
        unique_ratio = len(generated_fingerprints)/len(final_smiles_list)
        unique_elements_len = len(generated_fingerprints)

        save_params(deviation, gen_target, gen_input, target_percent, input_percent, new_percent, total_input_fraction, unique_ratio, unique_elements_len, settings, long_ratio)

        if x % 1 ==0:
            smiles_data = {'smiles': final_smiles_list}
            df = pd.DataFrame(smiles_data)
            log_folder = str(out_path) + str(dir_name)

            if not os.path.exists(log_folder):
                os.makedirs(log_folder)

            current_path = log_folder + '/' + str(x) + '.csv'

            df.to_csv(current_path, index = False)