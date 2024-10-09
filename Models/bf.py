import torch
import os
import sys
import yaml
import numpy as np
import selfies
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
import time

import mlp
import vae
from functions import decode_lpoints, get_selfie_and_smiles_encodings_for_dataset
from functions_sub import selfies_to_all, lpoints_to_onehots, onehots_to_lpoints, canonicalize_smiles, fp_generation, make_model, try_encode, get_run_log_list, save_index, unique_values_and_first_indices, non_none_values_and_indices, generate_normal_tensor, get_free_memory, get_free_ram, save_index_fp, get_run_log_list_fp

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import RDConfig
from rdkit.Chem import AllChem
import deepchem as dc 
fpgen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=4096)
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

from rdkit import RDLogger 
RDLogger.DisableLog('rdApp.*') 

torch.set_grad_enabled(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def get_models(settings):

    '''Gets the models and other items needed to run the algorithm'''

    '''Arguments:
                        settings: dict of settings used (dict)'''
        
    '''Outputs:
                        vae_encoder: the encoder object (VAEEncoder object)
                        vae_decoder: the decoder object (VAEDecoder object) 
                        mlp_model: the multi-layered perceptron object, trained on mus (PropertyRegressionModel object)
                        z_model: the multi-layered perceptron object, trained on zs (PropertyRegressionModel object)
                        encoding_alphabet: the alphabet generated by the function get_selfie_and_smiles_encodings_for_dataset (list)
                        largest_molecule_len: the maximum length of the molecule encodings (int)
                        lpoint_size: size of the normal tensor (shape[1]), should correspond to the size of the latent vecotrs you want to apply this to (int)'''

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

    largest_molecule_len = int(encoder_dict['encode_RNN.weight_ih_l0'].shape[1]/len(selfies_alphabet))
    lpoint_size = decoder_dict['decode_RNN.weight_ih_l0'].shape[1]


    vae_encoder = vae.VAEEncoder(in_dimension=(encoder_dict['encode_RNN.weight_ih_l0'].shape[1]), **encoder_parameter)
    vae_decoder = vae.VAEDecoder(**decoder_parameter, out_dimension=len(selfies_alphabet), seq_len=largest_molecule_len)
    vae_encoder.load_state_dict(encoder_dict)
    vae_decoder.load_state_dict(decoder_dict)


    mlp_save_path = settings['mlp']['save_path']
    mlp_epoch = settings['mlp']['epoch']

    mlp_settings = yaml.safe_load(open(str(mlp_save_path) + "settings/" + "settings.yml", "r"))
    mlp_settings['model_params']['input_dim'] = vae_encoder.encode_FC_mu[0].weight.shape[0]
    mlp_model = mlp.PropertyRegressionModel(mlp_settings)
    state_dict = torch.load(mlp_save_path + '/' + str(mlp_epoch) + '/model.pt', map_location=device)
    mlp_model.load_state_dict(state_dict)

    z_mlp_save_path = settings['z_mlp']['save_path']
    z_mlp_epoch = settings['z_mlp']['epoch']

    z_mlp_settings = yaml.safe_load(open(str(z_mlp_save_path) + "settings/" + "settings.yml", "r"))
    z_mlp_settings['model_params']['input_dim'] = vae_encoder.encode_FC_mu[0].weight.shape[0]
    z_model = mlp.PropertyRegressionModel(z_mlp_settings)
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


    return vae_encoder, vae_decoder, mlp_model, z_model, selfies_alphabet, largest_molecule_len, lpoint_size

def round_to_nearest_even(n):

    '''Rounds a number to the nearest even number'''

    '''Arguments:
                        n: input number (float)'''
        
    '''Outputs:
                        n: even rounded version of input n (int)'''


    return n if n % 2 == 0 else n + 1

def save_smiles(generated_smiles, predictions, sascores, smiles_index, settings):

    '''Saves the generated SMILES and the associated properties to a file'''

    '''Arguments:
                    generated_smiles: generated SMILES list (list)
                    predictions: predictions for the SMILES list, this will nominally be the 1st transition energy (list)
                    sascores: the SA scores of the generated SMILES (list)
                    smiles_index: corresponding index of the SMILES seed that made this specific list of generated SMILES (int)
                    settings: settings defined by the corresponding .yml file (dict) '''
    

    save_path = settings['data']['save_path']
    csv_file_name = 'bf_results.csv'
    csv_file_path = save_path + csv_file_name

    df = pd.DataFrame({'smiles': generated_smiles, 'property': predictions, 'SAS': sascores, 'smiles_index': smiles_index})

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if not os.path.exists(csv_file_path):
        df.to_csv(csv_file_path, mode='a', header=True, index=False)

        settings_filepath = os.path.join(save_path, 'settings.yml')

        data = {**settings}
        with open(settings_filepath, 'w') as file:
            yaml.dump(data, file)
    else:
        df.to_csv(csv_file_path, mode='a', header=False, index=False)

def dump_csvs(df, settings, file_type):

    '''Function to dump 2 different csvs, ideal_mols, ideal_mols_2'''

    '''Arguments:
                        df: pandas DataFrame we want to dump (pandas DataFrame object)
                        settings: settings defined by the corresponding .yml file (dict)
                        file_type: 1, or 2. Corresponds to the file name we want (int) '''



    if file_type ==1:
        loc_file = '/ideal_mols.csv'
    if file_type ==2:
        loc_file = '/ideal_mols_2.csv'

    save_path = settings['data']['save_path']
    csv_file_path = save_path + loc_file

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if not os.path.exists(csv_file_path):
        df.to_csv(csv_file_path, mode='a', header=True, index=False)
    else:
        df.to_csv(csv_file_path, mode='a', header=False, index=False)


def gen_small_tensor_list(TENSOR, RECOMMENDED_TENSOR_SIZE):

    '''Function to split up a tensor into a list of smaller tensors. Used so that we have do not crash due to memory consumption'''

    '''Arguments:
                        TENSOR: input tensor (Pytorch float.32 tensor)
                        RECOMMENDED_TENSOR_SIZE: size of one tensor in the new smaller tensor list. e.g., if this is 10 and the original input tensor is size 100,
                                                    we would get a list of 10 tensors of size 10. (int)'''
        
    '''Outputs:
                        smaller_tensors: list of smaller tensors that should contain every entry in the original tensor (list)'''



    RECOMMENDED_TENSOR_SIZE = int(RECOMMENDED_TENSOR_SIZE)
    TENSOR_RESHAPE = round(TENSOR.shape[0]/RECOMMENDED_TENSOR_SIZE) + 1
    TENSOR_RESHAPE = round_to_nearest_even(TENSOR_RESHAPE)

    part_size = TENSOR.shape[0] // TENSOR_RESHAPE
    remainder = TENSOR.shape[0] % TENSOR_RESHAPE

    # Create a list to hold the smaller tensors
    smaller_tensors = []
    start_index = 0

    for i in range(TENSOR_RESHAPE):
        # Determine the end index for the current part
        end_index = start_index + part_size + (1 if i < remainder else 0)
        smaller_tensors.append(TENSOR[start_index:end_index])
        start_index = end_index
    
    return smaller_tensors



def gen_step(encoding_list, smiles_index, encoding_alphabet, largest_molecule_len, vae_encoder, z_num, eps):

    '''Brute force step 1. Generates the Zs'''

    '''Arguments:
                        encoding_list: a list containing the SELFIES (list)
                        smiles_index: specific SMILES index we are looking at, e.g., 0 for the 1st SMILES in the seed list (int)
                        encoding_alphabet: the alphabet generated by the function get_selfie_and_smiles_encodings_for_dataset (list)
                        largest_molecule_len: the maximum length of the molecule encodings (int)
                        vae_encoder: the encoder object (VAEEncoder object)
                        z_num: number of zs you want, e.g., 10 will give you 10 Zs in the output (int)
                        eps: the final normal distributed tensor (Pytorch float.32 tensor)'''
        
    '''Outputs:
                        zs: latent vectors defined by the distribution in eps and the mus (Pytorch float.32 tensor)'''


    _, mu, log_var = selfies_to_all([encoding_list[smiles_index]], encoding_alphabet, largest_molecule_len, vae_encoder)
    log_var = log_var
    std = torch.exp(0.5 * log_var)
    std_repeat = std.squeeze().repeat(z_num, 1)
    del std
    mus_repeat = mu.squeeze().repeat(z_num, 1)
    del mu
    zs = eps.mul(std_repeat).add_(mus_repeat).to(device)

    time_2 = time.time()
            
    del std_repeat
    del mus_repeat
    del log_var

    return zs


def z_model_step(z_model, zs, condition3):

    '''Brute force step 2. Predicts the properties of the Zs and discriminates based on condition3.'''

    '''Arguments:
                        z_model: the multi-layered perceptron object, trained on zs (PropertyRegressionModel object)
                        zs: latent vectors defined by the distribution in eps and the mus (Pytorch float.32 tensor)
                        condition3: condition that checks whether a number is bigger or lower than whatever the z threshold is (lambda function)'''
        
    '''Outputs:
                        threshold_indices: tensor containing the indices of the zs that met condition3 (Pytorch float.32 tensor)'''


    FREE_GPU = get_free_memory(device)
    RECOMMENDED_Z_SIZE = (FREE_GPU * 0.8) / ( 2 * 4 * (z_model.layers[0].weight.shape[0])) ### 0.8 so we use 80% of the gpu at maximum just to be safe
    sub_z = gen_small_tensor_list(zs, RECOMMENDED_Z_SIZE)

    z_model.eval()
    with torch.no_grad():
        predictions= torch.cat([z_model(x) for x in sub_z])
    mask = condition3(predictions)
    del predictions

    threshold_indices = mask.squeeze().nonzero().squeeze().tolist()
    del mask
    if type(threshold_indices) == int:
        threshold_indices = [threshold_indices]

    return threshold_indices


def mu_gen_step(zs, threshold_indices, selfies_alphabet, vae_decoder, vae_encoder):

    '''Brute force step 3. Takes the 'good' Zs and converts them to Mus.'''

    '''Arguments:
                        zs: latent vectors defined by the distribution in eps and the mus (Pytorch float.32 tensor)
                        threshold_indices: tensor containing the indices of the zs that met condition3 (Pytorch float.32 tensor)
                        selfies_alphabet: the alphabet generated by the function get_selfie_and_smiles_encodings_for_dataset (list)
                        vae_decoder: the decoder object (VAEDecoder object)
                        vae_encoder: the encoder object (VAEEncoder object)'''
        
    '''Outputs:
                        lpoints_decoded: Mu latent vectors that coreespond to the honest to goodness representations of the decoded Zs (Pytorch float.32 tensor)'''
   

    zs = zs[threshold_indices]
    del threshold_indices
    scattered_one_hots = lpoints_to_onehots(zs, selfies_alphabet, vae_decoder)
    
    del zs
    lpoints_decoded = onehots_to_lpoints(scattered_one_hots, vae_encoder).to(device)
    del scattered_one_hots

    return lpoints_decoded

def mu_model_step(mlp_model, mus, condition2):

    '''Brute force step 4. Predicts the properties of the Mus and discriminates based on condition2.'''

    '''Arguments:
                        mlp_model: the multi-layered perceptron object, trained on mus (PropertyRegressionModel object)
                        mus: Mu latent vectors that coreespond to the honest to goodness representations of the decoded Zs (Pytorch float.32 tensor)
                        condition2: condition that checks whether a number is bigger or lower than whatever the mu threshold is (lambda function)'''
        
    '''Outputs:
                        index_tensor: tensor containing the indices of the mus that met condition2 (Pytorch float.32 tensor)
                        predictions: tensor containing the predictions of the mus that met condition2 (Pytorch float.32 tensor)'''


    FREE_GPU = get_free_memory(device)
    RECOMMENDED_Z_SIZE = (FREE_GPU * 0.8) / ( 2 * 4 * (mlp_model.layers[0].weight.shape[0])) ### 0.8 so we use 80% of the gpu at maximum just to be safe
    sub_z = gen_small_tensor_list(mus, RECOMMENDED_Z_SIZE)

    mlp_model.eval()
    with torch.no_grad():
        predictions= torch.cat([mlp_model(x) for x in sub_z])
    mask = condition2(predictions)
    index_tensor = mask.squeeze().nonzero().squeeze().tolist()
    del mask
    if type(index_tensor) == int:
        index_tensor = [index_tensor]

    return index_tensor, predictions

def sanitisation_step(mus, predictions, selfies_alphabet, vae_decoder, index_tensor, condition):

    '''Brute force step 5. Sanitises the molecules generated so far.'''

    '''Arguments:
                        mus: Mu latent vectors that coreespond to the honest to goodness representations of the decoded Zs (Pytorch float.32 tensor)
                        predictions: tensor containing the predictions of the mus that met condition2 (Pytorch float.32 tensor)
                        selfies_alphabet: the alphabet generated by the function get_selfie_and_smiles_encodings_for_dataset (list)
                        vae_decoder: the decoder object (VAEDecoder object)
                        index_tensor: tensor containing the indices of the mus that met condition2 (Pytorch float.32 tensor)
                        condition: condition that checks whether a character is in the SELFIES alphabet (lambda function)'''
        
    '''Outputs:
                        top_smiles: list of SMILES that survived the sanitisation (list)
                        top_selfies: list of SELFIES that survives the sanitisation (list)
                        top_predictions: properties tensor that survived the sanitisation, corresponds to the other two lists (Pytorch float.32 tensor)'''


    mus = mus[index_tensor]
    predictions = predictions[index_tensor]
    del index_tensor
    scattered_smiles = decode_lpoints(mus, selfies_alphabet, vae_decoder)

    del mus
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
    del non_none_indices

    
    scatterd_selfies = [selfies.encoder(x) for x in scattered_smiles]
    top_selfies_index = [x for x in range(len(scatterd_selfies)) if all(condition(element) for element in scatterd_selfies[x][1:-1].split("]["))]
    top_selfies = [scatterd_selfies[x] for x in top_selfies_index]
    top_smiles = [scattered_smiles[x] for x in top_selfies_index]
    top_predictions = predictions[top_selfies_index]
    del predictions
    del scattered_smiles
    del top_selfies_index

    top_smiles = [selfies.decoder(x) for x in top_selfies]

    return top_smiles, top_selfies, top_predictions

def SAS_step(top_smiles, top_selfies, condition4, top_predictions):

    '''Brute force step 6. Calculates the SA score here.'''

    '''Arguments:
                        top_smiles: list of SMILES that survived the sanitisation (list)
                        top_selfies: list of SELFIES that survives the sanitisation (list)
                        condition4: condition that checks whether a number is bigger or lower than whatever the sas threshold is (lambda function)
                        top_predictions: properties tensor that survived the sanitisation, corresponds to the other two lists (Pytorch float.32 tensor)'''
        
    '''Outputs:
                        top_smiles: list of SMILES that met the SA score threshold (list)
                        top_selfies: list of SELFIES that met the SA score threshold (list)
                        top_sas: list of sa scores corresponding the molecules that met the SA threshold (list)
                        top_predictions: properties tensor that that met the SA score threshold, corresponds to the other three lists (Pytorch float.32 tensor)'''


    sascores = [sascorer.calculateScore(Chem.MolFromSmiles(x)) for x in top_smiles]
    sascores = torch.tensor(sascores)
            
    mask = condition4(sascores)
    threshold_indices = mask.squeeze().nonzero().squeeze().tolist()

    if type(threshold_indices) == int:
        threshold_indices = [threshold_indices]

    top_sas = sascores[threshold_indices]
    top_preds = top_predictions[threshold_indices]
    top_selfies = [top_selfies[x] for x in threshold_indices]
    top_smiles = [top_smiles[x] for x in threshold_indices]

    return top_smiles, top_selfies, top_sas, top_preds



def bf(vae_encoder, vae_decoder, mlp_model, z_model, selfies_alphabet, largest_molecule_len, lpoint_size, encoding_list, settings):

    '''Brute force algorithm - NETWORK A. Takes a list of molecular seeds and outputs a bunch of molecules that fit our criteria.'''

    '''Arguments:
                        vae_encoder: the encoder object (VAEEncoder object)
                        vae_decoder: the decoder object (VAEDecoder object) 
                        mlp_model: the multi-layered perceptron object, trained on mus (PropertyRegressionModel object)
                        z_model: the multi-layered perceptron object, trained on zs (PropertyRegressionModel object)
                        selfies_alphabet: the alphabet generated by the function get_selfie_and_smiles_encodings_for_dataset (list)
                        largest_molecule_len: the maximum length of the molecule encodings (int)
                        lpoint_size: number of dimensions in a given latent vector, e.g., 64 for a 64D latent vector (int)
                        encoding_list: a list containing the SELFIES (list)
                        settings: dict of settings used (dict) '''


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
    threshold_condition = lambda z: len(z) == 0

    time_0 = time.time()

    run_list = get_run_log_list(settings)
    smiles_index= max(run_list)
    print('Starting SMILES index:', smiles_index)

    while smiles_index < len(encoding_list):

        if smiles_index % 100 ==0 and smiles_index !=0:
            print('SMILES:', smiles_index, '/', len(encoding_list), 'Time taken:', time.time() - time_0)
            time_0 = time.time()

            

        zs = gen_step(encoding_list, smiles_index, selfies_alphabet, largest_molecule_len, vae_encoder, z_num, eps)
        threshold_indices = z_model_step(z_model, zs, condition3)
        if threshold_condition(threshold_indices):
            save_index(smiles_index, settings)
            smiles_index+=1
            continue


        mus = mu_gen_step(zs, threshold_indices, selfies_alphabet, vae_decoder, vae_encoder)
        index_tensor, predictions = mu_model_step(mlp_model, mus, condition2)
        if threshold_condition(index_tensor):
            save_index(smiles_index, settings)
            smiles_index+=1
            continue

        top_smiles, top_selfies, top_predictions = sanitisation_step(mus, predictions, selfies_alphabet, vae_decoder, index_tensor, condition)
        if threshold_condition(top_smiles):
            save_index(smiles_index, settings)
            smiles_index+=1
            continue
        

        top_smiles, _, sascores, top_predictions = SAS_step(top_smiles, top_selfies, condition4, top_predictions)
        if threshold_condition(top_smiles):
            save_index(smiles_index, settings)
            smiles_index+=1
            continue


        save_smiles(top_smiles, top_predictions.squeeze().tolist(), sascores.squeeze().tolist(), smiles_index, settings)
        del top_smiles
        del top_predictions
        del sascores

        save_index(smiles_index, settings)
        smiles_index +=1


def final_sanitiser(settings):

    '''This is the final sanitisation step. Here, we take the output from NETWORK A and clean it up, i.e., we remove duplicates and invalid molecules 
        then save them to a csv'''

    '''Arguments:
                        settings: dict of settings used (dict)'''
        

    save_path = settings['data']['save_path']
    csv_file_name = 'bf_results.csv'
    csv_file_path = save_path + csv_file_name
    main_dataset_path = settings['data']['main_dataset']

    df_bf = pd.read_csv(csv_file_path)
    df_main = pd.read_csv(main_dataset_path)

    df_bf['canon_smiles'] = [canonicalize_smiles(Chem.MolFromSmiles(x)) for x in df_bf['smiles']]
    df_bf.drop_duplicates(subset=['canon_smiles'])

    BF_SMILES = df_bf['smiles'].tolist()
    PC_SMILES = df_main['smiles'].tolist()

    BF_FP = fp_generation(BF_SMILES)
    PC_FP = fp_generation(PC_SMILES)

    df_bf['FP'] = BF_FP
    df_main['FP'] = PC_FP

    df_bf = df_bf.drop_duplicates(subset='FP')
    df_bool = df_bf['FP'].isin(df_main['FP'])
    df_filtered = df_bf[df_bool*1==0]
    del df_filtered['FP']

    df_filtered.to_csv(save_path + 'sanitised_bf_results.csv', index=False)
  


def df_prep(df_base, start_idx, stop_idx):

    '''This is where the pandas dataframes used in network B are prepared. We generate a df containing 3 fingerprints:
        - Daylight fingerprints
        - Morgan fingerprints
        - Mol2Vec objects'''

    '''Arguments:
                        df_base: This is the original pandas dataframe. It should be the dataframe representation of 'sanitised_bf_results.csv (Pandas DataFrame)'
                        start_idx: This is the starting id of our pandas dataframe. E.g., if we wanted our output df to be made of rows 1-10 of df_base, this would be 1
                        stop_idx: This is the finish id of our pandas dataframe. E.g., if we wanted our output df to be made of rows 1-10 of df_base, this would be 10'''


    '''Outputs:
                        sub_df: This is a pandas dataframe containing a subset of rows from df_base but with the fingerprints representations also (Pandas DataFrame)'''
    

    sub_df = df_base.iloc[start_idx:stop_idx].reset_index(drop=True)
    

    toMol2Vec = lambda z: np.array(featurizer(z).flatten())

    fpgen_morgan = AllChem.GetMorganGenerator(radius=3, fpSize=2048)
    fpgen_daylight = AllChem.GetRDKitFPGenerator()
    featurizer = dc.feat.Mol2VecFingerprint()


    mols = sub_df['smiles'].map(Chem.MolFromSmiles)

    sub_df['morgan_fingerprints'] = fpgen_morgan.GetFingerprints(mols,numThreads=32)
    sub_df['morgan_fingerprints'] = sub_df['morgan_fingerprints'].map(lambda x: x.ToBitString())

    sub_df['daylight_fingerprints'] = fpgen_daylight.GetFingerprints(mols,numThreads=32)
    sub_df['daylight_fingerprints'] = sub_df['daylight_fingerprints'].map(lambda x: x.ToBitString())

    sub_df['mol2vec'] = sub_df['smiles'].map(toMol2Vec)

    return sub_df





def NETWORK_B(settings):

    '''BF-NETWORK B: This is where our output from network A is fed through the superior MLP and our final predictions for the transition energies and
                        oscillator strengths are made.'''

    '''Arguments:
                        settings: dict of settings used (dict)'''


    file_path = settings['data']['save_path']
    B_OS_THRESHOLD = settings['network_B']['OS']
    B_dE_THRESHOLD = settings['network_B']['dE']
    bin_str_to_array = lambda x: np.frombuffer(x.encode('ascii'),'u1')- ord('0')
    batch_size = 13100

    df_base = pd.read_csv(file_path + '/sanitised_bf_results.csv')

    num_batches = int(len(df_base) / batch_size)
    if num_batches*batch_size < len(df_base):
        num_batches = num_batches +1

    size_feature = 4396
    size_target = 4
    activation = torch.nn.GELU()
    loss_fn = dc.models.losses.L1Loss()
    dropout = 0.20
    width = 4096*2
    model = make_model(size_feature, width, activation, dropout, 13, 0.0001, batch_size, loss_fn,size_target)
    m_dir = settings['network_B']['save_dir']
    model.restore(model_dir=m_dir)

    run_list = get_run_log_list_fp(settings)
    fp_index = max(run_list)


    for batch_iteration in range(fp_index, num_batches):
        print('NETWORK B Batch iteration:', batch_iteration, '/', num_batches)
        start_idx = batch_iteration * batch_size
        stop_idx = (batch_iteration + 1) * batch_size

        df_sanitised = df_prep(df_base, start_idx, stop_idx)
        df_sanitised['daylight_fingerprints'] = df_sanitised['daylight_fingerprints'].apply(bin_str_to_array)
        df_sanitised['morgan_fingerprints'] = df_sanitised['morgan_fingerprints'].apply(bin_str_to_array)
        df_sanitised['mol2vec'] = df_sanitised['mol2vec'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' ', dtype=np.float32) if isinstance(x, str) else x)



        df_sanitised = df_sanitised.rename(columns={'smiles': 'SMILES'})
        del df_sanitised['SAS']
        del df_sanitised['canon_smiles']
        df_sanitised['SELFIES'] = 0
        col = df_sanitised.pop('property')
        df_sanitised['property'] = col
        df_sanitised['TransitionEnergies1'] = 0
        df_sanitised['TransitionEnergies2'] = 0
        df_sanitised['TransitionEnergies3'] = 0
        df_sanitised['TransitionEnergies4'] = 0
        df_sanitised['TransitionEnergies5'] = 0
        df_sanitised['TransitionEnergies6'] = 0
        df_sanitised['TransitionEnergies7'] = 0
        df_sanitised['TransitionEnergies8'] = 0
        df_sanitised['TransitionEnergies9'] = 0
        df_sanitised['TransitionEnergies10'] = 0
        df_sanitised['OscillatorStrength1'] =0
        df_sanitised['OscillatorStrength2'] =0
        df_sanitised['OscillatorStrength3'] =0
        df_sanitised['OscillatorStrength4'] =0
        df_sanitised['OscillatorStrength5'] =0
        df_sanitised['OscillatorStrength6'] =0
        df_sanitised['OscillatorStrength7'] =0
        df_sanitised['OscillatorStrength8'] =0
        df_sanitised['OscillatorStrength9'] =0
        df_sanitised['OscillatorStrength10'] =0

        col = df_sanitised.pop('morgan_fingerprints')
        df_sanitised['morgan_fingerprints'] = col
        col = df_sanitised.pop('daylight_fingerprints')
        df_sanitised['daylight_fingerprints'] = col
        col = df_sanitised.pop('mol2vec')
        df_sanitised['mol2vec'] = col

        osc_cols = df_sanitised.columns[13:-3]
        for col in osc_cols:
            df_sanitised[col] = df_sanitised[col].apply(np.log10)
            df_sanitised[col] = df_sanitised[col].apply(lambda x: x)


        y_cols = df_sanitised.columns[3:5].append(df_sanitised.columns[13:15])
        n=1
        X = np.hstack((np.stack(df_sanitised['morgan_fingerprints'][::n].to_numpy()).astype('float32'),np.stack(df_sanitised['daylight_fingerprints'][::n].to_numpy()).astype('float32'),np.stack(df_sanitised['mol2vec'][::n].to_numpy())))
        y = df_sanitised[y_cols][::n].to_numpy().astype('float32')
        w = ([1]*np.shape(y)[0])
        ids = np.hstack(df_sanitised['SMILES'][::n])
        del df_sanitised
        dataset = dc.data.NumpyDataset(X, y, w, ids,n_tasks=np.shape(y)[0])

        predictions = model.predict(dataset)
        trans1 = [predictions[x][0] for x in range(predictions.shape[0])]
        osc1 = [10**(predictions[x][2])/predictions[x][0] for x in range(predictions.shape[0])]
        trans2 = [predictions[x][1] for x in range(predictions.shape[0])]
        osc2 = [10**(predictions[x][3])/predictions[x][1] for x in range(predictions.shape[0])]


        pubchem_no_fp = pd.read_csv(file_path + '/sanitised_bf_results.csv')
        pubchem_no_fp_2 = pd.read_csv(file_path + '/sanitised_bf_results.csv')
        pubchem_no_fp = pubchem_no_fp[start_idx:stop_idx]
        pubchem_no_fp_2 = pubchem_no_fp_2[start_idx:stop_idx]


        del pubchem_no_fp['smiles']
        del pubchem_no_fp_2['smiles']


        pubchem_no_fp['OscillatorStrength1'] = osc1
        pubchem_no_fp_2['OscillatorStrength1'] = osc1
        pubchem_no_fp['TransitionEnergies1'] = trans1
        pubchem_no_fp_2['TransitionEnergies1'] = trans1
        pubchem_no_fp_2['OscillatorStrength2'] = osc2
        pubchem_no_fp_2['TransitionEnergies2'] = trans2

        pubchem_no_fp.to_csv(file_path + '/sanitised_final_bf_results.csv', index=False)
        pubchem_no_fp_2.to_csv(file_path + '/sanitised_final_bf_results_2.csv', index=False)

        pubchem_check = pubchem_no_fp[pubchem_no_fp['OscillatorStrength1'] > B_OS_THRESHOLD]
        pubchem_final = pubchem_check[pubchem_check['TransitionEnergies1'] < B_dE_THRESHOLD]
        pubchem_final.drop_duplicates(subset=['canon_smiles'])
        del pubchem_final['property']
        col_ind = pubchem_final.pop('smiles_index')
        pubchem_final.insert(len(pubchem_final.columns), col_ind.name, col_ind)
        col_ind = pubchem_final.pop('SAS')
        pubchem_final.insert(len(pubchem_final.columns), col_ind.name, col_ind)
        dump_csvs(pubchem_final, settings, 1)


        pubchem_check = pubchem_no_fp_2[pubchem_no_fp_2['OscillatorStrength1'] > B_OS_THRESHOLD]
        pubchem_check2 = pubchem_no_fp_2[pubchem_no_fp_2['OscillatorStrength2'] > B_OS_THRESHOLD]
        pubchem_check = pd.concat([pubchem_check,pubchem_check2])
        pubchem_final = pubchem_check[pubchem_check['TransitionEnergies1'] < B_dE_THRESHOLD]
        pubchem_final2 = pubchem_check[pubchem_check['TransitionEnergies2'] < B_dE_THRESHOLD]
        pubchem_final = pd.concat([pubchem_final, pubchem_final2])
        pubchem_final.drop_duplicates(subset=['canon_smiles'])
        del pubchem_final['property']
        col_ind = pubchem_final.pop('smiles_index')
        pubchem_final.insert(len(pubchem_final.columns), col_ind.name, col_ind)
        col_ind = pubchem_final.pop('SAS')
        pubchem_final.insert(len(pubchem_final.columns), col_ind.name, col_ind)
        dump_csvs(pubchem_final, settings, 2)
        save_index_fp(batch_iteration, settings)  

    print('Complete')




def main():
    settings = yaml.safe_load(open("bf_settings.yml", "r"))
    smiles_path = settings['data']['smiles_path']

    vae_encoder, vae_decoder, mlp_model, z_model, selfies_alphabet, largest_molecule_len, lpoint_size = get_models(settings)
    encoding_list, _, _ = get_selfie_and_smiles_encodings_for_dataset(smiles_path)

    bf(vae_encoder, vae_decoder, mlp_model, z_model, selfies_alphabet, largest_molecule_len, lpoint_size, encoding_list, settings)
    final_sanitiser(settings)

    del vae_decoder
    del vae_encoder
    del encoding_list
    torch.cuda.empty_cache()

    NETWORK_B(settings)
