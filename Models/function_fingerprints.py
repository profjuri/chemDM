'''
Functions to generate molecular fingerprints. Contains an altered version of the RDKit mol2vec function which can run on a GPU.
'''

import torch
import numpy as np
from os import path

from rdkit import Chem
from rdkit.Chem import AllChem
from deepchem.utils.data_utils import download_url, get_data_dir, untargz_file
from rdkit import RDLogger 

import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from gensim.models import word2vec

from functions import selfies_to_lpoints

RDLogger.DisableLog('rdApp.*') 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_PRETRAINED_MODEL_URL = 'https://deepchemdata.s3-us-west-1.amazonaws.com/trained_models/mol2vec_model_300dim.tar.gz'



def _mol2alt_sentence(mol, radius):

    '''Taken from the RDKit mol2vec function. Takes RDKit molecules and converts them to sentences'''

    '''Arguments:
                    mol: RDKit mol object describing a molecule (RDKit mol object)
                    radius: the desired molecular fingerprint radius (int)'''
    
    '''Outputs:
                    list(alternating_sentence): contains a list of the sentences used to describe the RDKit mol object (list)'''
    

    radii = list(range(int(radius) + 1))
    info = {}
    _ = AllChem.GetMorganFingerprint(
        mol, radius,
        bitInfo=info) 

    mol_atoms = [a.GetIdx() for a in mol.GetAtoms()]
    dict_atoms = {x: {r: None for r in radii} for x in mol_atoms}

    for element in info:
        for atom_idx, radius_at in info[element]:
            dict_atoms[atom_idx][
                radius_at] = element  

    identifiers_alt = []
    for atom in dict_atoms: 
        for r in radii:  
            identifiers_alt.append(dict_atoms[atom][r])

    alternating_sentence = map(str, [x for x in identifiers_alt if x])

    return list(alternating_sentence)

def get_model():

    '''Taken from the RDKit mol2vec function. Generates an instance of the word2vec model used'''
    
    '''Outputs:
                    model: word2vec model used to convert sentences to mol2vec vectors (Gensim word2vec model object)'''


    data_dir = get_data_dir()
    pretrain_model_path = path.join(data_dir,
                                    'mol2vec_model_300dim.pkl')
    if not path.exists(pretrain_model_path):
        targz_file = path.join(data_dir, 'mol2vec_model_300dim.tar.gz')
        if not path.exists(targz_file):
            download_url(DEFAULT_PRETRAINED_MODEL_URL, data_dir)
        untargz_file(path.join(data_dir, 'mol2vec_model_300dim.tar.gz'),
                        data_dir)
    model = word2vec.Word2Vec.load(pretrain_model_path)

    return model


def m2v_1ds(sentences, alphabet_dict, unknown_index, pad_idx, tensor_vectors):

    '''Converts the sentences from _mol2alt_sentence into 1D pytorch tensors where each element corresponds to the number of times a key of that index appears'''

    '''Arguments:
                    sentences: contains a list of sentences used to describe RDKit mol objects (list)
                    alphabet_dict: a dict of the keys within the word2vec model and their corresponding index (dict)
                    unknown_index: the index of the unknown key defined by the word2vec model (int)
                    pad_idx: the index which will be used for padding (int)
                    tensor_vectors: pytorch tensor of the vectors contained within the word2vec model (Pytorch float.32 tensor)'''
    
    '''Outputs:
                    padded_tensor: N-dimensional tensor of vectors corresponding to the indices of the keys described within the sentences (Pytorch float.32 tensor)'''


    integer_encoded = [[alphabet_dict[symbol] if symbol in alphabet_dict else unknown_index for symbol in x] for x in sentences]
    max_length = max(len(inner_list) for inner_list in integer_encoded)
    padded_list = [torch.tensor(inner_list + [pad_idx] * (max_length - len(inner_list))) for inner_list in integer_encoded]
    padded_tensor = pad_sequence(padded_list, batch_first=True, padding_value=pad_idx)

    if padded_tensor.shape[1] < tensor_vectors.shape[0]:
        extra_padding = torch.zeros(padded_tensor.shape[0],(tensor_vectors.shape[0]), dtype = torch.int64) + pad_idx
        extra_padding[:, :padded_tensor.shape[1]] = padded_tensor
        padded_tensor = extra_padding

    return padded_tensor


def get_m2v(mols):

    '''Function to generate the mol2vecs'''

    '''Arguments:
                    mols: list of RDKit mol object describing molecules (list)'''
    
    '''Outputs:
                    m2v_tensor: pytorch tensor containing mol2vec representations of all inputted molecules. Should be size [N, X] where N is the number of mols and X is the 
                                size of the mol2vec vectors (Pytorch float.32 tensor)'''


    m2v_list = []
    sentences = [_mol2alt_sentence(x,1) for x in mols]
    model = get_model()

    batch_size = 32768
    num_batches = int(len(sentences) / batch_size)
    if num_batches * batch_size < len(sentences):
        num_batches +=1


    keys = set(model.wv.key_to_index.keys())
    reordered_vectors = [model.wv.get_vector(y) for y in keys]

    tensor_vectors = torch.tensor(reordered_vectors).to(device)
    unknown_index = torch.nonzero(tensor_vectors.squeeze() == torch.tensor(model.wv.get_vector('UNK')).squeeze().to(device))[0][0].item()
    nil_index = torch.zeros(tensor_vectors.shape[1]).to(device)
    tensor_vectors = torch.cat((tensor_vectors, nil_index.unsqueeze(0)))
    pad_idx = tensor_vectors.shape[0]-1
    alphabet_dict = {letter: index for index, letter in enumerate(keys)}


    for batch in range(num_batches):
        start_idx = batch * batch_size
        stop_idx = (batch+1) * batch_size

        batch_sentences = sentences[start_idx:stop_idx]
        padded_tensor = m2v_1ds(batch_sentences, alphabet_dict, unknown_index, pad_idx, tensor_vectors).to(device)

        num_rows, _ = padded_tensor.shape
        num_bins = pad_idx + 1

        count_tensors = torch.zeros((num_rows, num_bins), dtype=torch.float32).to(device)
        values = torch.ones_like(padded_tensor, dtype=torch.float32).to(device)
        count_tensors.scatter_add_(1, padded_tensor, values)
        
        m2v_tensor = torch.matmul(count_tensors, tensor_vectors).to('cpu')
        m2v_list.append(m2v_tensor)
        
        del count_tensors
        del values
        del m2v_tensor

    m2v_tensor = torch.cat(m2v_list)

    return m2v_tensor


def gen_fingerprints(smiles_list, encoding_list, vae_encoder, encoding_alphabet, largest_molecule_len, fingerprint_size):

    '''Function used to generate the fingerprints used within the MLP'''

    '''Arguments:
                    smiles_list: the list of SMILES encodings (list)
                    encoding_list: a list containing the SELFIES (list)
                    vae_encoder: the encoder object (VAEEncoder object)
                    encoding_alphabet: the alphabet generated by the function get_selfie_and_smiles_encodings_for_dataset (list)
                    largest_molecule_len: the maximum length of the molecule encodings (int)
                    fingerprint_size: size of the daylight and morgan fingerprints (int)'''
    
    '''Outputs:
                    lpoints: the mean latent vectors (Pytorch float.32 tensor)
                    morgan_fp_tensor: pytorch tensor containing morgan fingerprints (Pytorch float.32 tensor)
                    daylight_fingerprints_tensor: pytorch tensor containing daylight fingerprints (Pytorch float.32 tensor)
                    mol2vec_tensor: pytorch tensor containing mol2vec fingerprints (Pytorch float.32 tensor)'''


    fpgen_morgan = AllChem.GetMorganGenerator(radius=3, fpSize=fingerprint_size)
    fpgen_daylight = AllChem.GetRDKitFPGenerator(fpSize=fingerprint_size)
    

    if len(smiles_list) < 10:
        num_batches = len(smiles_list)
    else:  
        num_batches = 15
        gen_batch_size = int(len(smiles_list) /num_batches)

    while gen_batch_size * num_batches < len(smiles_list):
        num_batches +=1

    lpoint_list = []
    morgan_list = []
    daylight_list = []
    mol2vec_list = []


    for x in range(num_batches):
        start_idx = x * gen_batch_size
        stop_idx = (x + 1) * gen_batch_size
        
        sub_smiles = smiles_list[start_idx:stop_idx]
        sub_encoding = encoding_list[start_idx:stop_idx]

        mols = [Chem.MolFromSmiles(x) for x in sub_smiles]
        vae_encoder.eval()
        with torch.no_grad():
            lpoints = selfies_to_lpoints(sub_encoding, encoding_alphabet, largest_molecule_len, vae_encoder).flatten(1).to('cpu').to(torch.float32)
        lpoint_list.append(lpoints)

        morgan_fps = fpgen_morgan.GetFingerprints(mols,numThreads=32)
        morgan_fp_tensor = torch.tensor(np.array(morgan_fps).astype(float)).to('cpu').to(torch.float32)
        morgan_list.append(morgan_fp_tensor)

        daylight_fps = fpgen_daylight.GetFingerprints(mols,numThreads=32)
        daylight_fingerprints_tensor = torch.tensor(np.array(daylight_fps).astype(float)).to('cpu').to(torch.float32)
        daylight_list.append(daylight_fingerprints_tensor)

        mol2vec_tensor = get_m2v(mols)
        mol2vec_list.append(mol2vec_tensor)

    lpoints = torch.cat(lpoint_list)
    del lpoint_list
    morgan_fp_tensor = torch.cat(morgan_list)
    del morgan_list
    daylight_fingerprints_tensor = torch.cat(daylight_list)
    del daylight_list
    mol2vec_tensor = torch.cat(mol2vec_list)
    del mol2vec_list


    return lpoints, morgan_fp_tensor, daylight_fingerprints_tensor, mol2vec_tensor

def gen_fingerprints_BF(mols, fingerprint_size):


    '''Function used to generate the fingerprints used within the brute force algorithm'''

    '''Arguments:
                    mols: list of RDKit mol object describing molecules (list)
                    fingerprint_size: size of the daylight and morgan fingerprints (int)'''
    
    '''Outputs:
                    morgan_fp_tensor: pytorch tensor containing morgan fingerprints (Pytorch float.32 tensor)
                    daylight_fingerprints_tensor: pytorch tensor containing daylight fingerprints (Pytorch float.32 tensor)
                    mol2vec_tensor: pytorch tensor containing mol2vec fingerprints (Pytorch float.32 tensor)'''


    fpgen_morgan = AllChem.GetMorganGenerator(radius=3, fpSize=fingerprint_size)
    fpgen_daylight = AllChem.GetRDKitFPGenerator(fpSize=fingerprint_size)
    gen_batch_size = 65536


    if len(mols) < 10:
        num_batches = len(mols)
    else:  
        num_batches = int(len(mols) // gen_batch_size) + 1

    while gen_batch_size * num_batches < len(mols):
        num_batches +=1

    morgan_list = []
    daylight_list = []
    mol2vec_list = []
    
    for x in range(num_batches):
        start_idx = x * gen_batch_size
        stop_idx = (x + 1) * gen_batch_size
        sub_mols = mols[start_idx:stop_idx]

        mol2vec_tensor = get_m2v(sub_mols)
        mol2vec_list.append(mol2vec_tensor)

        morgan_fps = fpgen_morgan.GetFingerprints(sub_mols,numThreads=32)
        morgan_fp_tensor = torch.tensor(np.array(morgan_fps).astype(float)).to('cpu').to(torch.float32)
        morgan_list.append(morgan_fp_tensor)

        daylight_fps = fpgen_daylight.GetFingerprints(sub_mols,numThreads=32)
        daylight_fingerprints_tensor = torch.tensor(np.array(daylight_fps).astype(float)).to('cpu').to(torch.float32)
        daylight_list.append(daylight_fingerprints_tensor)

    del morgan_fp_tensor
    morgan_fp_tensor = torch.cat(morgan_list)
    del morgan_list

    del daylight_fingerprints_tensor
    daylight_fingerprints_tensor = torch.cat(daylight_list)
    del daylight_list
    
    del mol2vec_tensor
    mol2vec_tensor = torch.cat(mol2vec_list)
    del mol2vec_list


    return morgan_fp_tensor, daylight_fingerprints_tensor, mol2vec_tensor