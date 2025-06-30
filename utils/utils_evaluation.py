import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
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
import gin
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from sparsify.sparsify.sparse_coder import SparseCoder as SAE
from datasets import load_dataset
def create_dictionary_neurons(path_embeddings:Path, list_neurons:List[int])->str:
        """
        Creates a dictionary mapping each neuron to its top-5 most activated image IDs.

        Args:
            path_embeddings (Path): Path to the folder containing embedding files and average activations.
            list_neurons (List[int]): List of neuron indices to process.

        Returns:
            str: Path to the saved JSON file containing the neuron-to-image dictionary.
        """
        json_files = glob.glob(os.path.join(path_embeddings, 'vqa_res_block_*.json'))
        # Create a combined dictionary for all files
        combined_data = {}

        # Read and combine all json files
        for file_path in json_files:
            with open(file_path, 'r') as f:
                data = json.load(f)
                combined_data.update(data)

        dictionary_neurons={}
        average_activation_dictionary = json.load(open(path_embeddings+'average_activation_dictionary.json'))
        for neuron in tqdm(list_neurons,desc='Sorting the activations'):
            sorted_list = sorted(average_activation_dictionary[str(neuron)].items(), key=lambda x: x[1][1], reverse=True)

            new_sorted_list=[el[0] for el in sorted_list[0:5]]

            dictionary_neurons[neuron]={el:combined_data[el] for el in new_sorted_list}
        with open(os.path.join(path_embeddings, 'dictionary_neurons.json'), 'a') as f:
            json.dump(dictionary_neurons, f)
        return path_embeddings+'dictionary_neurons.json'
 
def eval_text_vision(CLIP_model,preprocess, image, concept_list,device)->np.ndarray:
    """
    Evaluates the similarity between an image and a list of textual concepts using a CLIP model.

    Args:
        CLIP_model: The CLIP model used for encoding and similarity computation.
        preprocess: Preprocessing function for the input image.
        image: The input image to be evaluated.
        concept_list: A list of tokenized textual concepts to compare against the image.

    Returns:
        np.ndarray: Probabilities representing the similarity between the image and each concept.
    """

    

    image = preprocess(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # image_features = CLIP_model.encode_image(image)
        # text_features = CLIP_model.encode_text(concept_list)
        
        logits_per_image, _ = CLIP_model(image, concept_list)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        
     
    
    return probs

def eval_text_textual(CLIP_model:clip,concept_designed:str, concept_list:List,cosine_function)->List[float]:
    """
    Computes the similarity between a designed concept and a list of concepts using a CLIP model.

    Args:
        CLIP_model (clip): The CLIP model used for encoding and similarity computation.
        concept_designed (str): The input concept to compare.
        concept_list (List): List of tokenized concepts to compare against.
        cosine_function: Function to compute cosine similarity.

    Returns:
        List[float]: Similarity scores between the designed concept and each concept in the list.
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

@gin.configurable
def eval_visual_hypotheses(path_embeddings:Path,path_dataset:Path,path_dictionary_neurons:Path,
                            device:torch.cuda.device='cuda:0')->None:
    """Evaluate the visual hypotheses using CLIP 

    Args:
        path_embeddings (Path):  Path to the folder containing embedding files
        path_dataset (Path):  Path to the dataset cache directory
        path_dictionary_neurons (Path): Path to the JSON file with neuron-to-image mappings
        device (torch.cuda.device, optional): Device to run computations on. Defaults to 'cuda:0'
        
    Retrurns:
        None: Saves the results in JSON files, one for visual concepts and one for categories.
    """    
    
    data = load_dataset("lmms-lab/LLaVA-NeXT-Data", split="train[:15%]", cache_dir=path_dataset, num_proc=10)

    dictionary_sae_neurons=json.load(open(path_dictionary_neurons,'r'))
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
    hypotheses_path = path_embeddings + 'dictionary_hypotheses_complete_visual.json'
    if not os.path.exists(hypotheses_path):
        unite_dictionaries(path_embeddings,modality='visual')   
    dictionary_hypotheses = json.load(open(hypotheses_path, 'r'))
    dictionary_concepts,list_concepts=cleaning_hypotheses(dictionary_hypotheses)


    list_concepts = [concept for concept in dictionary_concepts.values() if "pixelated" not in concept.lower()]
    
    clip_model,preprocess=clip.load("ViT-B/16",device=device)
    clip_model.eval()
    prob_dictionary={}
    prob_dictionary_categories={}
    concepts_clip=clip.tokenize([concept for concept in list_concepts]).to(device)
    categories = ["scene", "object", "part", "material", "texture", "color"]
    categories_clip=clip.tokenize([label for label in categories]).to(device)
    
    for neuron_number,_ in tqdm(dictionary_concepts.items(),desc='Evaluate the visual hypothesis' ,total=len(dictionary_concepts),leave=True):
        batch=dictionary_sae_neurons[neuron_number]
        visual_concept_probabilities, category_probabilities =[],[]
        for img_id_str, feats in batch.items():
            img_id = int(img_id_str)
            entry = lookup.get(img_id)
            
            if entry is None:
                continue

            # build prompt text
       

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
                
            prob=eval_text_vision(CLIP_model=clip_model,preprocess=preprocess,concept_list=concepts_clip,
                                image=Image.fromarray(reconstructed_array),device=device)
            
            prob_categories = eval_text_vision(CLIP_model=clip_model, preprocess=preprocess, concept_list=categories_clip,
                                               image=Image.fromarray(reconstructed_array), device=device)
            visual_concept_probabilities.append(prob[0])   
            category_probabilities.append(prob_categories[0])
            
   
            
        average_concepts=np.mean(visual_concept_probabilities,axis=0)
        variance_concepts=np.var(visual_concept_probabilities,axis=0)
        
        average_categories=np.mean(category_probabilities,axis=0)
        variance_categories=np.var(category_probabilities,axis=0)
        
                
        prob_dictionary[neuron_number]=[average_concepts.tolist(),variance_concepts.tolist()]
        prob_dictionary_categories[neuron_number]=[average_categories.tolist(),variance_categories.tolist()]
                
   
                    
    
    json.dump(prob_dictionary, open(path_embeddings + 'prob_concept_visual_dictionary.json', 'a'), indent=4)
    json.dump(prob_dictionary_categories, open(path_embeddings + 'prob_concept_visual_categories_dictionary.json', 'a'), indent=4)

@gin.configurable
def eval_textual_hypotheses(path_embeddings:Path,path_dataset:Path,path_dictionary_neurons:Path,
                             device:torch.cuda.device='cuda:0')->None:
    """Evaluate textual hypotheses using CLIP 

    Args:
        path_embeddings (Path):  Path to the folder containing embedding files
        path_dataset (Path):  Path to the dataset cache directory
        path_dictionary_neurons (Path): Path to the JSON file with neuron-to-image mappings
        device (torch.cuda.device, optional): Device to run computations on. Defaults to 'cuda:0'.
    Returns:
        None: Saves the results in JSON files, one for textual concepts and one for categories.
    """    
        
    clip_model,_=clip.load("ViT-B/16",device=device)
    clip_model.eval()
    
  
    
    # df_label=pd.DataFrame.from_dict(average_activation_dictionary)
    cosine = nn.CosineSimilarity(dim=1, eps=1e-6)


    
    clip_model.eval()
    prob_dictionary={}
                            

    data = load_dataset("lmms-lab/LLaVA-NeXT-Data", split="train[:15%]", cache_dir=path_dataset, num_proc=10)

    dictionary_sae_neurons=json.load(open(path_dictionary_neurons,'r'))
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
    hypotheses_path = path_embeddings + 'dictionary_hypotheses_complete_textual.json'
    if not os.path.exists(hypotheses_path):
        unite_dictionaries(path_embeddings,modality='textual')   
    dictionary_hypotheses = json.load(open(hypotheses_path, 'r'))
    dictionary_concepts,list_concepts=cleaning_hypotheses(dictionary_hypotheses)


    
    prob_dictionary={}
    prob_dictionary_categories={}
    tmp=clip.tokenize([concept for concept in list_concepts]).to(device)
    categories = ["scene", "object", "part", "material", "texture", "color"]
    categories_clip=clip.tokenize([label for label in categories]).to(device)
    
    for neuron_number,_ in tqdm(dictionary_concepts.items(),desc='Evaluate the textual hypothesis' ,total=len(dictionary_concepts),leave=True):
        batch=dictionary_sae_neurons[neuron_number]
        textual_concept_probabilities, category_probabilities = [],[]
        
        for img_id_str, feats in batch.items():
            img_id = int(img_id_str)
            entry = lookup.get(img_id)
            convo = entry["conversations"][0]["value"]
            text_feat = feats["textual_features"]["final_output"]
            tmp_text=text_feat
            tmp_text=clip.tokenize(tmp_text).to(device)
            
            
            prob=eval_text_textual(CLIP_model=clip_model,concept_designed=tmp_text,concept_list=tmp,
                                cosine_function=cosine)

            prob_categories = eval_text_textual(CLIP_model=clip_model,concept_designed=tmp_text,concept_list=categories_clip,
                                cosine_function=cosine)
            textual_concept_probabilities.append(prob[0])   
            category_probabilities.append(prob_categories[0])
            
   
            
        average_miner=np.mean(textual_concept_probabilities,axis=0)
        variance_miner=np.var(textual_concept_probabilities,axis=0)
        
        average_categories=np.mean(category_probabilities,axis=0)
        variance_categories=np.var(category_probabilities,axis=0)
        
                
        prob_dictionary[neuron_number]=[average_miner.tolist(),variance_miner.tolist()]
        prob_dictionary_categories[neuron_number]=[average_categories.tolist(),variance_categories.tolist()]
                
   
                    
    
    json.dump(prob_dictionary, open(path_embeddings + 'prob_concept_textual_dictionary.json', 'a'), indent=4)
    json.dump(prob_dictionary_categories, open(path_embeddings + 'prob_concept_textual_categories_dictionary.json', 'a'), indent=4)


def top_alpha_binarize(activations: torch.Tensor, alpha: float) -> torch.Tensor:
    """Return a binary tensor where top alpha fraction of activations are 1."""
    threshold = torch.quantile(activations, 1 - alpha)
    return (activations >= threshold).int()

def recall_score(a_k: torch.Tensor, c_t: torch.Tensor, alpha: float = 0.05) -> float:
    """Implements Recall-based evaluation."""
    B_ak = top_alpha_binarize(a_k, alpha)
    B_ct = (c_t >= 0.5).int()
    return (B_ak * B_ct).sum().item() / (B_ak.sum().item() + 1e-8)

def iou_score(a_k: torch.Tensor, c_t: torch.Tensor, alpha: float = 0.05) -> float:
    """Implements IoU-based evaluation."""
    B_ak = top_alpha_binarize(a_k, alpha)
    B_ct = (c_t >= 0.5).int()
    intersection = (B_ak * B_ct).sum().item()
    union = B_ak.sum().item() + B_ct.sum().item() - intersection
    return intersection / (union + 1e-8)

def pearson_corr(a_k: torch.Tensor, c_t: torch.Tensor) -> float:
    """Pearson correlation coefficient between neuron activations and concept activations."""
    return torch.corrcoef(torch.stack([a_k, c_t]))[0, 1].item()

def neuron_eval(a_k: torch.Tensor, c_t: torch.Tensor, metric: str = 'recall') -> float:
    if metric == 'recall':
        return recall_score(a_k, c_t)
    elif metric == 'iou':
        return iou_score(a_k, c_t)
    elif metric == 'correlation':
        return pearson_corr(a_k, c_t)
    else:
        raise NotImplementedError(f"Metric '{metric}' not implemented.")

