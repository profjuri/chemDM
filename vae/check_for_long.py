import selfies
import torch
import chemistry_vae_symmetric_rnn_OG
import time
import random
import pandas as pd
import os
import yaml
import numpy as np

import deepchem as dc
import chemfp
import chemfp.bitops
from rdkit.Chem import rdFingerprintGenerator
from rdkit import Chem
from rdkit import Chem
from rdkit import DataStructs
import rdkit

def save_params(deviation, gen_target, gen_input, target_percent, input_percent, new_percent, total_input_fraction, unique_ratio, unique_elements_len, settings, calc_time, long_ratio):

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
            file.write("deviation,gen_target,gen_input,target_percent,input_percent,total_input_fraction,unique_ratio,unique_elements_len,calc_time,long_ratio,new_percent\n")
        file.write(f'{deviation},{gen_target},{gen_input},{target_percent},{input_percent},{total_input_fraction},{unique_ratio},{unique_elements_len},{calc_time},{long_ratio},{new_percent}\n')



def main():
    if os.path.exists("check_for_long.yml"):
        settings = yaml.safe_load(open("check_for_long.yml", "r"))
    else:
        print("Expected a file settings.yml but didn't find it.")
        return
    


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(settings['vae']['vae_epoch'])
    save_path = settings['vae']['path']
    save_name = settings['runs']['save_name']

    file_to_load = save_path
    vae_epoch = settings['vae']['vae_epoch']
    training_file_nameE = str(vae_epoch)+"/E"
    training_file_nameD = str(vae_epoch)+"/D"

    vae_encoder = torch.load(file_to_load + training_file_nameE, map_location=device)
    vae_decoder = torch.load(file_to_load + training_file_nameD, map_location=device)


    selfies_alphabet = settings['vae']['alphabet']
    dataset = settings['runs']['dataset']
    target = settings['runs']['target']

    _, _, _, target_list, _, _ = chemistry_vae_symmetric_rnn_OG.get_selfie_and_smiles_encodings_for_dataset(target)
    encoding_list, encoding_alphabet, largest_molecule_len, encoding_smiles, _, _ = chemistry_vae_symmetric_rnn_OG.get_selfie_and_smiles_encodings_for_dataset(dataset)
    data = chemistry_vae_symmetric_rnn_OG.multiple_selfies_to_hot(encoding_list, largest_molecule_len, selfies_alphabet)
    data = torch.tensor(data, dtype=torch.float).to(device)

    inp_flat_one_hot = data.flatten(start_dim=1)
    inp_flat_one_hot = inp_flat_one_hot.unsqueeze(0) 

    runs_coeff = settings['runs']['num']

    start = settings['runs']['start']
    end = settings['runs']['end']
    step = settings['runs']['step']

    numbers = np.arange(start, end + step, step).tolist()
    my_list = [round(x, 2) for x in numbers]

    print(my_list)

    runs = 10 * round((len(target_list)+ len(encoding_list))/len(encoding_list))
    print('number of runs:', runs)

    for x in range(len(my_list)):
        deviation = my_list[x]

        print('current:', my_list[x])




        final_smiles_list = []




        for x in range(runs):

            z, mu, log_var = vae_encoder(inp_flat_one_hot)
            std = torch.exp(0.5 * log_var).to(device)

            variation_tensor = torch.tensor([[random.uniform(-deviation, deviation) for _ in range(std.shape[1])] for _ in range(std.shape[0])], dtype=torch.float32).to(device)
            latent_vecs = mu + variation_tensor * std



            print('std tnsr 1:', std[1])
            print('var tnsr 1:', variation_tensor[1])
            print('t list len:', len(target_list))
            print('i list len:', len(encoding_list))






            start_time = time.time()

            vae_decoder.eval()
            vae_encoder.eval()
    
            seq_tensor = torch.empty(0, dtype=torch.float32).to(device)
            hidden = vae_decoder.init_hidden(batch_size=latent_vecs.shape[0])

            print('data shape:', data.shape[1])

            for seq_index in range(data.shape[1]):

                out_one_hot_line, hidden = vae_decoder(latent_vecs.unsqueeze(0), hidden)
                out_one_hot_line = out_one_hot_line.squeeze()
                out_one_hot_line_arg = torch.argmax(out_one_hot_line, dim = 1).unsqueeze(0)
                seq_tensor = torch.cat((seq_tensor, out_one_hot_line_arg), dim = 0)

            sequences = seq_tensor.t()
            list_of_continuous_strings = [selfies.decoder(''.join([selfies_alphabet[int(i)] for i in row])) for row in sequences]
            final_smiles_list = final_smiles_list + list_of_continuous_strings

            

            end_time = time.time()
            print('step:', x, 'time taken:', end_time-start_time)

        start_calc = time.time()

        filtered_list = [x for x in final_smiles_list if x is not None]
        unique_elements = list(set(filtered_list))


        fpgen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=4096)
        fingerprints_unique = [fpgen.GetFingerprint(Chem.MolFromSmiles(x)).ToBitString() for x in unique_elements]
        fingerprints_target = [fpgen.GetFingerprint(Chem.MolFromSmiles(x)).ToBitString() for x in target_list]
        fingerprints_og = [fpgen.GetFingerprint(Chem.MolFromSmiles(x)).ToBitString() for x in encoding_smiles]



        _, idx = np.unique(fingerprints_unique, return_index=True)
        fingerprints_unique = np.unique(fingerprints_unique)
        
        un_selfies = [unique_elements[x] for x in idx]
        atoms = {"N", "F", "O", "C"}
        counts = [sum(selfie.count(atom) for atom in atoms) for selfie in un_selfies]
        long_smiles = [un_selfies[i] for i in range(len(un_selfies)) if counts[i] > 9]


        long_ratio = len(long_smiles)/len(fingerprints_unique)

        target_count = 0

        similar_fingerprints = np.in1d(fingerprints_unique, fingerprints_target)
        similar_fingerprints = similar_fingerprints*1
        target_count = np.sum(similar_fingerprints)

        same_count = 0

        similar_fingerprints2 = np.in1d(fingerprints_unique, fingerprints_og)
        similar_fingerprints2 = similar_fingerprints2*1
        same_count = np.sum(similar_fingerprints2)


        gen_target = target_count/len(target_list)
        gen_input = same_count/len(encoding_smiles)
        

        target_percent = target_count/len(fingerprints_unique)
        input_percent = same_count/len(fingerprints_unique)
        new_percent = 1 - (target_percent + input_percent + long_ratio)

        total_input_fraction = len(encoding_smiles)/(len(target_list)+len(encoding_smiles))

        unique_ratio = len(fingerprints_unique)/len(final_smiles_list)
        unique_elements_len = len(fingerprints_unique)

        calc_time = time.time() - start_calc

        save_params(deviation, gen_target, gen_input, target_percent, input_percent, new_percent, total_input_fraction, unique_ratio, unique_elements_len, settings, calc_time, long_ratio)






        

        data2 = {'smiles': final_smiles_list}

        df = pd.DataFrame(data2)

        df.to_csv(str(save_name), index = False)
    

