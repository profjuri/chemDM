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
from functions_sub import selfies_to_all, lpoints_to_onehots, onehots_to_lpoints, canonicalize_smiles, fp_generation, make_model, threshold_refiner, try_encode, get_run_log_list, save_index, unique_values_and_first_indices, non_none_values_and_indices, generate_normal_tensor, get_free_memory, get_free_ram

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

def save_smiles(generated_smiles, predictions, sascores, smiles_index, settings):

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

    if file_type ==1:
        loc_file = '/ideal_mols.csv'
    else:
        loc_file = '/ideal_mols_2.csv'


    save_path = settings['data']['save_path']
    csv_file_path = save_path + loc_file

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print('Len of DF:', len(df))

    if not os.path.exists(csv_file_path):
        df.to_csv(csv_file_path, mode='a', header=True, index=False)
    else:
        df.to_csv(csv_file_path, mode='a', header=False, index=False)


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


            _, mu, log_var = selfies_to_all([encoding_list[smiles_index]], selfies_alphabet, largest_molecule_len, vae_encoder, lpoint_size)
            log_var = log_var
            std = torch.exp(0.5 * log_var)
            std_repeat = std.squeeze().repeat(z_num, 1)
            del std
            mus_repeat = mu.squeeze().repeat(z_num, 1)
            del mu
            zs = eps.mul(std_repeat).add_(mus_repeat).to(device)
            
            del std_repeat
            del mus_repeat
            del log_var
            z_model.eval()
            
            
            predictions= z_model(zs).detach()
            mask = condition3(predictions)
            del predictions


            threshold_indices = mask.squeeze().nonzero().squeeze().tolist()
            del mask
            if type(threshold_indices) == int:
                threshold_indices = [threshold_indices]

            if len(threshold_indices) > 0:  

                  
                        
                zs = zs[threshold_indices]
                del threshold_indices
                
                if zs.shape[0] > 0:
                    scattered_one_hots = lpoints_to_onehots(zs, selfies_alphabet, vae_decoder)
                    del zs
                    lpoints_decoded = onehots_to_lpoints(scattered_one_hots, vae_encoder).to(device)
                    del scattered_one_hots

                    mlp_model.eval()
                    predictions = mlp_model(lpoints_decoded)


                    index_tensor = condition2(predictions).squeeze().nonzero().squeeze().tolist()
                    if type(index_tensor) == int:
                        index_tensor = [index_tensor]


                    if len(index_tensor) > 0:

                        lpoints_decoded = lpoints_decoded[index_tensor]
                        predictions = predictions[index_tensor]
                        del index_tensor
                        scattered_smiles = decode_lpoints(lpoints_decoded, selfies_alphabet, vae_decoder)
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


                        if len(top_smiles) > 0:


                            sascores = [sascorer.calculateScore(Chem.MolFromSmiles(x)) for x in top_smiles]
                            sascores = torch.tensor(sascores)
                            _, top_smiles, sascores, top_predictions = threshold_refiner(top_selfies, sascores, condition4, top_predictions) ### returns SMILES meeting the SA score threshold


                            if len(top_smiles) > 0:




                                save_smiles(top_smiles, top_predictions.squeeze().tolist(), sascores.squeeze().tolist(), smiles_index, settings)
                                del top_smiles
                                del top_predictions
                                del sascores
                    


        save_index(smiles_index, settings)
        smiles_index +=1
        counter +=1


def final_sanitiser(settings):

    fp_dict = {}
    unique_indices = []
    unique_indices2 = []

    save_path = settings['data']['save_path']
    csv_file_name = 'bf_results.csv'
    csv_file_path = save_path + csv_file_name
    main_dataset_path = settings['data']['main_dataset']

    df_bf = pd.read_csv(csv_file_path)
    df_main = pd.read_csv(main_dataset_path)

    df_bf['canon_smiles'] = [canonicalize_smiles(Chem.MolFromSmiles(x)) for x in df_bf['smiles']]
    df_bf.drop_duplicates(subset=['canon_smiles'])

    BF_SMILES = df_bf['smiles'].tolist()
    BF_CANON_SMILES = df_bf['canon_smiles']
    BF_PROP = df_bf['property'].tolist()
    BF_SAS = df_bf['SAS'].tolist()
    BF_INDEX = df_bf['smiles_index'].tolist()

    PC_SMILES = df_main['smiles'].tolist()

    print('Beginning fp generation')


    BF_FP = fp_generation(BF_SMILES)
    PC_FP = fp_generation(PC_SMILES)

    df_bf['FP'] = BF_FP
    df_main['FP'] = PC_FP

    print('Length before duplicate drop:', len(df_bf))
    df_bf = df_bf.drop_duplicates(subset='FP')
    print('Length after duplicate drop:', len(df_bf))

    df_bool = df_bf['FP'].isin(df_main['FP'])

    df_filtered = df_bf[df_bool*1==0]
    print('Length after main dataset duplicate drop:', len(df_filtered))
    del df_filtered['FP']

    df_filtered.to_csv(save_path + 'sanitised_bf_results.csv', index=False)



def data_prep(settings):

    print('Data prep start')

    file_path = settings['data']['save_path']

    sanitised_csv_path = file_path + '/sanitised_bf_results.csv'
    df_san = pd.read_csv(sanitised_csv_path)

    toMol2Vec = lambda z: np.array(featurizer(z).flatten())


    fpgen_morgan = AllChem.GetMorganGenerator(radius=3, fpSize=2048)
    fpgen_daylight = AllChem.GetRDKitFPGenerator()
    featurizer = dc.feat.Mol2VecFingerprint()


    mols = df_san['smiles'].map(Chem.MolFromSmiles)

    print('Mol finished')


    df_san['morgan_fingerprints'] = fpgen_morgan.GetFingerprints(mols,numThreads=32)
    df_san['morgan_fingerprints'] = df_san['morgan_fingerprints'].map(lambda x: x.ToBitString())

    print('Morgan fingerprints gen')

    df_san['daylight_fingerprints'] = fpgen_daylight.GetFingerprints(mols,numThreads=32)
    df_san['daylight_fingerprints'] = df_san['daylight_fingerprints'].map(lambda x: x.ToBitString())

    print('Daylight fingerprints gen')

    df_san['mol2vec'] = df_san['smiles'].map(toMol2Vec)

    print('Mol2vec generated')

    df_san.to_csv(file_path + 'sanitised_bf_results_fingerprints.csv', index =False)


def NETWORK_B(settings):

    file_path = settings['data']['save_path']
    B_OS_THRESHOLD = settings['network_B']['OS']
    B_dE_THRESHOLD = settings['network_B']['dE']
    bin_str_to_array = lambda x: np.frombuffer(x.encode('ascii'),'u1')- ord('0')
    batch_size = 131072

    sanitised_fp_csv_path = file_path + 'sanitised_bf_results_fingerprints.csv'
    big_df_sanitised = pd.read_csv(sanitised_fp_csv_path, converters = {'daylight_fingerprints':bin_str_to_array,'morgan_fingerprints':bin_str_to_array,'mol2vec': lambda x:np.fromstring(x.strip('[]'), sep=' ', dtype=np.float32)})

    num_batches = int(len(big_df_sanitised) / batch_size)
    if num_batches*batch_size < len(big_df_sanitised):
        num_batches = num_batches +1

    size_feature = 4396
    size_target = 4
    activation = torch.nn.GELU()
    #batch_size = 1000
    loss_fn = dc.models.losses.L1Loss()
    dropout = 0.20
    width = 4096*2
    model = make_model(size_feature, width, activation, dropout, 13, 0.0001, batch_size, loss_fn,size_target)
    m_dir = settings['network_B']['save_dir']
    model.restore(model_dir=m_dir)


    for batch_iteration in range(num_batches):
        print('Batch iteration:', batch_iteration, '/', num_batches)
        start_idx = batch_iteration * batch_size
        stop_idx = (batch_iteration + 1) * batch_size


        df_sanitised = big_df_sanitised[start_idx:stop_idx]
        print('LEN df_sanitised', len(df_sanitised))
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

    print('Complete')




def main():
    settings = yaml.safe_load(open("bf_settings.yml", "r"))
    smiles_path = settings['data']['smiles_path']

    vae_encoder, vae_decoder, mlp_model, z_model, selfies_alphabet, largest_molecule_len, lpoint_size = get_models(settings)
    encoding_list, _, _ = get_selfie_and_smiles_encodings_for_dataset(smiles_path)

    bf(vae_encoder, vae_decoder, mlp_model, z_model, selfies_alphabet, largest_molecule_len, lpoint_size, encoding_list, settings)
    
    final_sanitiser(settings)

    data_prep(settings)
    del vae_decoder
    del vae_encoder
    del encoding_list


    torch.cuda.empty_cache()

    NETWORK_B(settings)


