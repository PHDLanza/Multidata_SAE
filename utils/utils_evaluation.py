import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["HF_HUB_CACHE"]="/data/lanza/hub"

os.environ["TOKENIZERS_PARALLELISM"]="false"
os.environ["PATH"] += os.pathsep + "/sbin/"

import torch
import json
from typing import  List, Dict
from tqdm import tqdm
from pathlib import Path
import numpy as np
from utils.utils_image import create_image_patches, reconstruct_image
import pandas as pd
from PIL import Image
import torch.nn as nn
import glob
import clip
import glob
from datasets import load_dataset 
from utils.api import unite_dictionaries
from utils.utils_image import reconstruct_image_blurring
def create_dictionary_neurons(folder_save_embedding:Path, list_neurons:List[int])->None:
        """_summary_

        Args:
            folder_save_embedding (Path): _description_
            list_neurons (List[int]): _description_

        Returns:
            _type_: _description_
        """        
        json_files = glob.glob(os.path.join(folder_save_embedding, 'vqa_res_block_*.json'))
        # Create a combined dictionary for all files
        combined_data = {}

        # Read and combine all json files
        for file_path in json_files:
            with open(file_path, 'r') as f:
                data = json.load(f)
                combined_data.update(data)

        dictionary_neurons={}
        average_activation_dictionary = json.load(open(folder_save_embedding+'average_activation_dictionary.json'))
        for neuron in tqdm(list_neurons,desc='Sorting the activations'):
            sorted_list = sorted(average_activation_dictionary[str(neuron)].items(), key=lambda x: x[1][1], reverse=True)

            new_sorted_list=[el[0] for el in sorted_list[0:5]]

            dictionary_neurons[neuron]={el:combined_data[el] for el in new_sorted_list}
        with open(os.path.join(folder_save_embedding, 'dictionary_neurons.json'), 'a') as f:
            json.dump(dictionary_neurons, f)
        return folder_save_embedding+'dictionary_neurons.json'
 
def eval_text_vision(CLIP_model,preprocess, image, concept_list,device):
    """_summary_

    Args:
        CLIP_model (_type_): _description_
        preprocess (_type_): _description_
        image (_type_): _description_
        list_text (_type_): _description_

    Returns:
        _type_: _description_
    """    

    

    image = preprocess(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # image_features = CLIP_model.encode_image(image)
        # text_features = CLIP_model.encode_text(concept_list)
        
        logits_per_image, _ = CLIP_model(image, concept_list)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        
     
    
    return probs

def eval_text_text(CLIP_model:clip,concept_designed:str, concept_list:List,cosine_function)->List[float]:
    """_summary_

    Args:
        model (clip): _description_
        tokenizer (_type_): _description_
        text1 (str): _description_
        text2 (str): _description_

    Returns:
        float: _description_
    """    
    # Tokenize and encode texts
    similarity_vector=[]
    with torch.no_grad():
        
        text_features_concept = CLIP_model.encode_text(concept_designed)
        
        cos=cosine_function(text_features_concept, concept_list)
        
        similarity_vector.append(cos)
        
    return similarity_vector

def cleaning_hypotheses(dictionary_hypotheses:Dict)->Dict:
    cleaned_dictionary_hypotheses={}  
    list_concepts=[]
    for key,value in tqdm(dictionary_hypotheses.items(),desc='Cleaning the dictionary' ,total=len(dictionary_hypotheses),leave=True):
        
        if value:       
            if isinstance(value, tuple):
                tmp_values = value[0]
            else:
                tmp_values = value
            if len(tmp_values.split('Concept: '))>1:
                
                concept=tmp_values.split('Concept:')[1].replace('"','').strip()
                
                if "No textual concept" not in concept and "No visual concept" not in concept:
                    
                    if concept not in list_concepts:
                        list_concepts.append(concept)
                        cleaned_dictionary_hypotheses[key]=concept
                        
    return cleaned_dictionary_hypotheses,list_concepts

def eval_visual_hypothteses(folder_embeddings:Path,folder_dataset:Path,dictionary_neurons_path:Path,device:torch.cuda.device='cuda:0'):
    """
    Evaluate visual hypotheses for neurons using CLIP and dataset images.

    Args:
        folder_embeddings (Path): Path to the folder containing embedding files.
        folder_dataset (Path): Path to the dataset cache directory.
        dictionary_neurons_path (Path): Path to the JSON file with neuron-to-image mappings.
        device (torch.cuda.device, optional): Device to run computations on. Defaults to 'cuda:0'.

    Returns:
        None
    """
    data = load_dataset("lmms-lab/LLaVA-NeXT-Data", split="train[:15%]", cache_dir=folder_dataset, num_proc=10)

    dictionary_sae_neurons=json.load(open(dictionary_neurons_path,'r'))
    needed_ids = set()
    for _, batch in dictionary_sae_neurons.items():  # limit to 1000 for progress bar
        needed_ids.update(map(int, batch.keys()))
    lookup = {}

    for example in tqdm(data, desc="Building lookup", leave=False):
        img_id = int(example["id"])
        if img_id in needed_ids:
            lookup[img_id] = {
                
                "conversations": example["conversations"],
                "image": example["image"].convert('RGB')
            
            }
        if len(lookup) >= len(needed_ids):
            break
    hypotheses_path = folder_embeddings + 'dictionary_hypotheses_complete_visual.json'
    if not os.path.exists(hypotheses_path):
        unite_dictionaries(folder_embeddings,modality='visual')   
    dictionary_hypotheses = json.load(open(hypotheses_path, 'r'))
    dictionary_concepts,list_concepts=cleaning_hypotheses(dictionary_hypotheses)


    list_concepts = [concept for concept in dictionary_concepts.values() if "pixelated" not in concept.lower()]
    
    clip_model,preprocess=clip.load("ViT-B/16",device=device)
    clip_model.eval()
    prob_dictionary={}
    prob_dictionary_categories={}
    tmp=clip.tokenize([concept for concept in list_concepts]).to(device)
    categories = ["scene", "object", "part", "material", "texture", "color"]
    categories_clip=clip.tokenize([label for label in categories]).to(device)
    
    for neuron_number,_ in tqdm(dictionary_concepts.items(),desc='Evaluate the visual hypothesis' ,total=len(dictionary_concepts),leave=True):
        batch=dictionary_sae_neurons[neuron_number]
        texts,tmp_list, tmp_categories_list = [],[],[]
        for img_id_str, feats in batch.items():
            img_id = int(img_id_str)
            entry = lookup.get(img_id)
            
            if entry is None:
                continue

            # build prompt text
            convo = entry["conversations"][0]["value"]
            text_feat = feats["textual_features"]["final_output"]
            texts.append(convo.replace("<image>", " ").replace("\n", " ")
                            + f" [ {text_feat} ]")

            # mask out patches
            image = entry["image"]
            # Process image patches
            patches = create_image_patches(image)
                    
            
            # Create masked image based on neuron activation
            
            zeros = np.zeros(len(patches), dtype=np.uint8)
            for patch_idx, inds in enumerate(feats["visual_features"]["latent_indices"]):
                if int(neuron_number) in inds:
                    zeros[patch_idx] = 1
            
            reconstructed_array = reconstruct_image(patches, zeros)
            # reconstructed_array = reconstruct_image_blurring(patches, zeros)
                
            prob=eval_text_vision(CLIP_model=clip_model,preprocess=preprocess,concept_list=tmp,
                                image=Image.fromarray(reconstructed_array),device=device)
            
            prob_categories = eval_text_vision(CLIP_model=clip_model, preprocess=preprocess, concept_list=categories_clip,
                                               image=Image.fromarray(reconstructed_array), device=device)
            tmp_list.append(prob[0])   
            tmp_categories_list.append(prob_categories[0])
            
   
            
        average_miner=np.mean(tmp_list,axis=0)
        variance_miner=np.var(tmp_list,axis=0)
        
        average_categories=np.mean(tmp_list,axis=0)
        variance_categories=np.var(tmp_list,axis=0)
        
                
        prob_dictionary[neuron_number]=[average_miner.tolist(),variance_miner.tolist()]
        prob_dictionary_categories[neuron_number]=[average_categories.tolist(),variance_categories.tolist()]
                
   
                    
    
    json.dump(prob_dictionary, open(folder_embeddings + 'prob_concept_image_dictionary.json', 'a'), indent=4)
    json.dump(prob_dictionary_categories, open(folder_embeddings + 'prob_concept_image_categories_dictionary.json', 'a'), indent=4)



def eval_textual_hypothtesis(folder_embeddings:Path,folder_dataset:Path,dictionary_neurons_path:Path,device:torch.cuda.device='cuda:0'):
    """_summary_

    Args:
        folder_embeddings (Path): _description_
        dictionary_neurons_path (Path): _description_
        dicty (Dict): _description_
        device (_type_, optional): _description_. Defaults to 'cuda:0'.
        
    """     
    clip_model,_=clip.load("ViT-B/16",device=device)
    clip_model.eval()
    
  
    
    # df_label=pd.DataFrame.from_dict(average_activation_dictionary)
    cosine = nn.CosineSimilarity(dim=1, eps=1e-6)


    
    clip_model.eval()
    prob_dictionary={}
                            

    data = load_dataset("lmms-lab/LLaVA-NeXT-Data", split="train[:15%]", cache_dir=folder_dataset, num_proc=10)

    dictionary_sae_neurons=json.load(open(dictionary_neurons_path,'r'))
    needed_ids = set()
    for _, batch in dictionary_sae_neurons.items():  # limit to 1000 for progress bar
        needed_ids.update(map(int, batch.keys()))
    lookup = {}

    for example in tqdm(data, desc="Building lookup", leave=False):
        img_id = int(example["id"])
        if img_id in needed_ids:
            lookup[img_id] = {
                
                "conversations": example["conversations"],
                "image": example["image"].convert('RGB')
            
            }
        if len(lookup) >= len(needed_ids):
            break
    hypotheses_path = folder_embeddings + 'dictionary_hypotheses_complete_textual.json'
    if not os.path.exists(hypotheses_path):
        unite_dictionaries(folder_embeddings,modality='textual')   
    dictionary_hypotheses = json.load(open(hypotheses_path, 'r'))
    dictionary_concepts,list_concepts=cleaning_hypotheses(dictionary_hypotheses)


    
    prob_dictionary={}
    prob_dictionary_categories={}
    tmp=clip.tokenize([concept for concept in list_concepts]).to(device)
    categories = ["scene", "object", "part", "material", "texture", "color"]
    categories_clip=clip.tokenize([label for label in categories]).to(device)
    
    for neuron_number,_ in tqdm(dictionary_concepts.items(),desc='Evaluate the textual hypothesis' ,total=len(dictionary_concepts),leave=True):
        batch=dictionary_sae_neurons[neuron_number]
        texts,tmp_list, tmp_categories_list = [],[],[]
        for img_id_str, feats in batch.items():
            img_id = int(img_id_str)
            entry = lookup.get(img_id)
            convo = entry["conversations"][0]["value"]
            text_feat = feats["textual_features"]["final_output"]
            tmp_text=text_feat
            tmp_text=clip.tokenize(tmp_text).to(device)
            
            
            prob=eval_text_text(CLIP_model=clip_model,concept_designed=tmp_text,concept_list=tmp,
                                cosine_function=cosine)

            prob_categories = eval_text_text(CLIP_model=clip_model,concept_designed=tmp_text,concept_list=categories_clip,
                                cosine_function=cosine)
            tmp_list.append(prob[0])   
            tmp_categories_list.append(prob_categories[0])
            
   
            
        average_miner=np.mean(tmp_list,axis=0)
        variance_miner=np.var(tmp_list,axis=0)
        
        average_categories=np.mean(tmp_list,axis=0)
        variance_categories=np.var(tmp_list,axis=0)
        
                
        prob_dictionary[neuron_number]=[average_miner.tolist(),variance_miner.tolist()]
        prob_dictionary_categories[neuron_number]=[average_categories.tolist(),variance_categories.tolist()]
                
   
                    
    
    json.dump(prob_dictionary, open(folder_embeddings + 'prob_concept_textual_dictionary.json', 'a'), indent=4)
    json.dump(prob_dictionary_categories, open(folder_embeddings + 'prob_concept_image_categories_dictionary.json', 'a'), indent=4)

