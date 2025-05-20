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

    
    # model, preprocess = clip.load("ViT-B/32", device=device)

    image = preprocess(image).unsqueeze(0).to(device)
    # ["a diagram", "a dog", "a cat"]
    

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
        
        text_features_list=concept_list
      
        # for el in text_features_list:
            
        #     cos=cosine_function(text_features_concept, el)
        #     similarity_vector_tmp.append(cos.item())
        cos=cosine_function(text_features_concept, text_features_list)
        
        similarity_vector.append(cos)
        
    return similarity_vector

def cleaning_hypotheses(dictionary_hypotheses:Dict)->Dict:
    cleaned_dictionary_hypotheses={}  
    for key,value in tqdm(dictionary_hypotheses.items(),desc='Cleaning the dictionary' ,total=len(dictionary_hypotheses),leave=True):
    
        if value:       
                    
            if len(value.split('Concept: '))>1:
                
                concept=value.split('Concept:')[1].replace('"','').strip()
                
                if concept!="No textual concept" and concept!="No visual concept.":
                    
                    cleaned_dictionary_hypotheses[key]=concept
                    enum+=1
                    
    return cleaned_dictionary_hypotheses

def eval_hypothtesis_image(folder_embeddings:Path,folder_dataset:Path,folder_labels:Path,dictionary_neurons_path:Path,device:torch.cuda.device='cuda:0'):
    """_summary_

    Args:
        folder_embeddings (Path): _description_
        folder_dataset (Path): _description_
        folder_labels (Path): _description_
        dictionary_neurons_path (Path): _description_
        dicty (Dict): _description_
        device (_type_, optional): _description_. Defaults to 'cuda:0'.
        
    """     
    average_activation_dictionary=json.load(open(folder_labels, 'r'))
    df_label=pd.DataFrame.from_dict(average_activation_dictionary)
    dictionary_sae_neurons=json.load(open(dictionary_neurons_path,'r'))
    dictionary_hypotheses=json.load(open(folder_embeddings+'dictionary_hypotheses_complete_image.json','r'))
    dictionary_hypotheses_cleaned=cleaning_hypotheses(dictionary_hypotheses)

    
    clip_model,preprocess=clip.load("ViT-B/16",device=device)
    clip_model.eval()
    prob_dictionary={}
    concept_list = clip.tokenize( [el for el in dictionary_hypotheses_cleaned.values()]).to(device)

    for key,value in tqdm(dictionary_hypotheses_cleaned.items(),desc='Evaluate the visual hypothesis' ,total=len(dictionary_hypotheses_cleaned),leave=True):
        
        if value:
            neuron_number=key
            batch=dictionary_sae_neurons[neuron_number]
            ids_list=batch.keys()
            neuron_number=int(neuron_number)
            tmp_list=[]
            for i,id in enumerate(ids_list):
                
                img_name=df_label[id]['image_name']
                folder_tmp = folder_dataset+'train2014/' if 'train' in img_name else folder_dataset+'val2014/'
                patches=create_image_patches(folder_tmp+img_name)
                zeros_vector = np.zeros(576)
                
                for i,indices_array in enumerate(batch[id]["visual_features"]['latent_indices']):
                    # common_elements=set(indices_array.tolist()).intersection(indices[0].tolist())
                    if neuron_number in indices_array:
                        zeros_vector[i]=1
                        
                    reconstructed_array = reconstruct_image(patches,zeros_vector)
                
                prob=eval_text_vision(CLIP_model=clip_model,preprocess=preprocess,concept_list=concept_list,image=Image.fromarray(reconstructed_array),device=device)
                tmp_list.append(prob[0]) 
            average=np.mean(tmp_list,axis=0)
                
            prob_dictionary[neuron_number]=average.tolist()
                
   
                    
    
    json.dump(prob_dictionary, open(folder_embeddings + 'prob_concept_image_dictionary.json', 'a'), indent=4)

def eval_hypothtesis_text(folder_embeddings:Path,dictionary_neurons_path:Path,device:torch.cuda.device='cuda:0'):
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
    dictionary_sae_neurons=json.load(open(dictionary_neurons_path,'r'))
    dictionary_hypotheses=json.load(open(folder_embeddings+'dictionary_hypotheses_complete_text.json','r'))
    dictionary_hypotheses_cleaned=cleaning_hypotheses(dictionary_hypotheses)
    cosine = nn.CosineSimilarity(dim=1, eps=1e-6)
    concept_list = clip.tokenize( [el for el in dictionary_hypotheses_cleaned.values()]).to(device)
    concept_list = clip_model.encode_text(concept_list)
    
    clip_model.eval()
    prob_dictionary={}
    
    for key,value in tqdm(dictionary_hypotheses_cleaned.items(),desc='Evaluate the textual hypothesis' ,total=len(dictionary_hypotheses_cleaned),leave=True):
        
        if value:

            neuron_number=key
            batch=dictionary_sae_neurons[neuron_number]
            ids_list=batch.keys()
    
            neuron_number=int(neuron_number)
            tmp_tensor = torch.tensor([]).to(device)
    
            for id in ids_list:
                tmp_text=batch[id]["text_features"]['final_output'].replace('assistant','')
                tmp_text=tmp_text.replace('\n','')
                tmp_text=clip.tokenize(tmp_text).to(device)
                
                
                prob=eval_text_text(CLIP_model=clip_model,concept_designed=tmp_text,concept_list=concept_list,cosine_function=cosine)
                if len(tmp_tensor)==0:       
                    tmp_tensor=prob[0].unsqueeze(0)
                else:
                    tmp_tensor = torch.cat((tmp_tensor, prob[0].unsqueeze(0)), dim=0)
                
            average=torch.mean(torch.tensor(tmp_tensor),axis=0)
            
            prob_dictionary[neuron_number]=average.tolist()
                
           
                    
    
    json.dump(prob_dictionary, open(folder_embeddings + 'prob_concept_text_dictionary.json', 'a'), indent=4)
    
    
