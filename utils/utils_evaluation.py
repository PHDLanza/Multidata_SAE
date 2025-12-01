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
from llava.model.builder import load_pretrained_model
from utils.api import compute_fvu
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.api import unite_dictionaries
import argparse
# Because we are running the explainer and scorer separately, we need to add the explanation and extra examples back to the record
from pathlib import Path
from utils.utils_prompt import DSCORER_SYSTEM_PROMPT, DSCORER_EXAMPLE_ONE, DSCORER_EXAMPLE_TWO,DSCORER_EXAMPLE_THREE
from utils.utils_prompt import  DSCORER_RESPONSE_ONE, DSCORER_RESPONSE_TWO,DSCORER_RESPONSE_THREE
import random
from utils.utils_prompt import FSCORER_SYSTEM_PROMPT, FSCORER_EXAMPLE_ONE, FSCORER_EXAMPLE_TWO,FSCORER_EXAMPLE_THREE
from utils.utils_prompt import  FSCORER_RESPONSE_ONE, FSCORER_RESPONSE_TWO,FSCORER_RESPONSE_THREE
def create_dictionary_neurons(embedding_path:Path, neuron_list:List[int])->str:
        """
        Creates a dictionary mapping each neuron to its top-5 most activated image IDs.

        Args:
            embedding_path (Path): Path to the folder containing embedding files and average activations.
            neuron_list (List[int]): List of neuron indices to process.

        Returns:
            str: Path to the saved JSON file containing the neuron-to-image dictionary.
        """
        json_files = glob.glob(os.path.join(embedding_path, 'vqa_res_block_*.json'))
        # Create a combined dictionary for all files
        combined_data = {}

        # Read and combine all json files
        for file_path in json_files:
            with open(file_path, 'r') as f:
                data = json.load(f)
                combined_data.update(data)

        dictionary_neurons={}
        average_activation_dictionary = json.load(open(embedding_path+'average_activation_dictionary.json'))
        for neuron in tqdm(neuron_list,desc='Sorting the activations'):
            sorted_list = sorted(average_activation_dictionary[str(neuron)].items(), key=lambda x: x[1][1], reverse=True)

            new_sorted_list=[el[0] for el in sorted_list[0:5]]

            dictionary_neurons[neuron]={el:combined_data[el] for el in new_sorted_list}
        with open(os.path.join(embedding_path, 'dictionary_neurons.json'), 'a') as f:
            json.dump(dictionary_neurons, f)
        return embedding_path+'dictionary_neurons.json'
 
def eval_text_vision(CLIP_model:clip,preprocess, image:Image, concept_list:list)->np.ndarray:
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

    

    image = preprocess(image).unsqueeze(0).to(CLIP_model.device)
    
    with torch.inference_mode():
        # image_features = CLIP_model.encode_image(image)
        # text_features = CLIP_model.encode_text(concept_list)
        
        logits_per_image, _ = CLIP_model(image, concept_list)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        
     
    
    return probs

def eval_text_textual(CLIP_model:clip,concept_designed:str, concept_list:List,cosine_function:nn.CosineSimilarity)->List[float]:
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
    with torch.inference_mode():
        
        text_features_concept = CLIP_model.encode_text(concept_designed)
        
        cos=cosine_function(text_features_concept, concept_list)
        
        similarity_vector.append(cos)
        
    return similarity_vector


def cleaning_hypotheses(hypothesis_dictionary: Dict) -> tuple[Dict, List]:
    """Cleans a dictionary of hypotheses by removing repeated and hallucinated long concepts.

    Args:
        hypothesis_dictionary: (Dict):Dictionary with hypotheses as string

    Tuple[Dict, List]: 
        - A dictionary with same keys as input, but only unique, valid concepts as values.
        - A list of unique, valid concepts extracted from the input dictionary.
    """    
    cleaned_dictionary_hypotheses={}  
    list_concepts=[]
    # list_concepts_embedding=[]
    
    for key,value in tqdm(hypothesis_dictionary.items(),desc='Cleaning the dictionary' ,total=len(hypothesis_dictionary),leave=True):
        
        if value:       
            if isinstance(value, tuple):
                tmp_values = value[0]
            else:
                tmp_values = value
            if len(tmp_values.split('Concept: '))>1:
                
                concept=tmp_values.split('Concept:')[1].replace('"','').strip()
                # Skip hallucinated concepts that do not contain the word 'Concept:'
                if len(concept)<40 and"No textual concept" not in concept and "No visual concept" not in concept:
                    
                    # similarity_score_s = [cosine(concept_embedding, el)[0] for el in list_concepts_embedding]
                    # if any(score > 0.95 for score in similarity_score_s):
                    #     continue
                    if concept not in list_concepts:
                            
                        list_concepts.append(concept)
                        cleaned_dictionary_hypotheses[key]=concept
                        
    return [cleaned_dictionary_hypotheses,list_concepts]

@gin.configurable
def eval_visual_hypotheses(embedding_path:Path,dataset_path:Path,dictionary_neurons_path:Path,
                            device:torch.cuda.device='cuda:0')->None:
    """Evaluate the visual hypotheses using CLIP 

    Args:
        embedding_path (Path):  Path to the folder containing embedding files
        dataset_path(Path):  Path to the dataset cache directory
        dictionary_neurons_path (Path): Path to the JSON file with neuron-to-image mappings
        device (torch.cuda.device, optional): Device to run computations on. Defaults to 'cuda:0'
        
    Retrurns:
        None: Saves the results in JSON files, one for visual concepts and one for categories.
    """    
    
    data = load_dataset("lmms-lab/LLaVA-NeXT-Data", split="train[:15%]", cache_dir=dataset_path, num_proc=10)

    dictionary_sae_neurons=json.load(open(dictionary_neurons_path,'r'))
    needed_ids = set()
    for _, batch in dictionary_sae_neurons.items():  
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
    hypotheses_path = embedding_path + 'dictionary_hypotheses_complete_visual.json'
    if not os.path.exists(hypotheses_path):
        unite_dictionaries(embedding_path,modality='visual')   
    dictionary_hypotheses = json.load(open(hypotheses_path, 'r'))
    dictionary_concepts,list_concepts=cleaning_hypotheses(dictionary_hypotheses)

    # Filter out concepts containing "pixelated" from both dictionary_concepts and list_concepts
    filtered_dictionary_concepts = {k: v for k, v in dictionary_concepts.items() if "pixelated" not in v.lower()}
    list_concepts = [concept for concept in filtered_dictionary_concepts.values()]
    dictionary_concepts = filtered_dictionary_concepts
    
    clip_model,preprocess=clip.load("ViT-B/32",device=device)
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
            
            # reconstructed_array = reconstruct_image(patches, zeros)
            reconstructed_array = reconstruct_image_blurring(patches, zeros)
                
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
                
   
                    
    
    json.dump(prob_dictionary, open(embedding_path + 'prob_concept_visual_dictionary_Vit32.json', 'a'), indent=4)
    json.dump(prob_dictionary_categories, open(embedding_path + 'prob_concept_visual_categories_dictionary_Vit32.json', 'a'), indent=4)


@gin.configurable
def eval_textual_hypotheses(embedding_path:Path,path_dictionary_neurons:Path,
                             device:torch.cuda.device='cuda:0')->None:
    """Evaluate textual hypotheses using CLIP 

    Args:
        embedding_path (Path):  Path to the folder containing embedding files
        dataset_path (Path):  Path to the dataset cache directory
        dictionary_neurons_path (Path): Path to the JSON file with neuron-to-image mappings
        device (torch.cuda.device, optional): Device to run computations on. Defaults to 'cuda:0'.
    Returns:
        None: Saves the results in JSON files, one for textual concepts and one for categories.
    """    
        
    clip_model,_=clip.load("ViT-B/32",device=device)
    clip_model.eval()
    
  
    
    # df_label=pd.DataFrame.from_dict(average_activation_dictionary)
    cosine = nn.CosineSimilarity(dim=1, eps=1e-6)
    clip_model.eval()
    prob_dictionary={}                      
    dictionary_sae_neurons=json.load(open(path_dictionary_neurons,'r'))
    hypotheses_path = embedding_path + 'dictionary_hypotheses_complete_textual.json'
    if not os.path.exists(hypotheses_path):
        unite_dictionaries(embedding_path,modality='textual')   
    dictionary_hypotheses = json.load(open(hypotheses_path, 'r'))
    dictionary_concepts,list_concepts=cleaning_hypotheses(dictionary_hypotheses)
    processor = LlavaNextProcessor.from_pretrained("llava-hf/llama3-llava-next-8b-hf")
    
    tmp =[clip_model.encode_text(clip.tokenize(el).to(device)) for el in list_concepts]
    clip_concepts_matrix=torch.cat(tmp,dim=0)

    for neuron_number,_ in tqdm(dictionary_concepts.items(),desc='Evaluate the textual hypothesis' ,total=len(dictionary_concepts),leave=True):
        batch=dictionary_sae_neurons[neuron_number]
        textual_concept_probabilities = []
        
        for _, feats in batch.items():
            # convo = entry["conversations"][0]["value"]
            final_string=[]
          
            input_ids = processor(feats["textual_features"]["final_output"])
            token_ids = input_ids["input_ids"][0]  # Assumes batch size = 1
            
            decoded_tokens = processor.batch_decode([[tok_id] for tok_id in token_ids], skip_special_tokens=True)

            final_string = []
            enter=False
            ids_super_token=-1
            max_activation=0
            half_num_characters=5
            num_tokens=2*half_num_characters

            # Start from index 1 (if skipping BOS or CLS tokens)
            for i, (token_str, neuron_ids,activation) in enumerate(zip(decoded_tokens, feats["textual_features"]["latent_indices"],feats["textual_features"]["latent_acts"])):
                if i == 0:
                    continue  # skip first token, <begin>
                # Extract id with most active textual token
                if int(neuron_number) in neuron_ids:
                    index_neuron_number=neuron_ids.index(int(neuron_number))
                    
                    if max_activation<activation[index_neuron_number]:
                        ids_super_token=i
                        max_activation=activation[index_neuron_number]
                    enter=True

                final_string.append(token_str)
            
            if enter:
                if ids_super_token> half_num_characters and (len(final_string)-ids_super_token)>half_num_characters:
                    final_string=final_string[ids_super_token-half_num_characters:ids_super_token+half_num_characters]
                    
                    
                elif ids_super_token<=half_num_characters:
                    final_string=final_string[:num_tokens]
                    
                elif (len(final_string)-ids_super_token)<half_num_characters:
                    
                    final_string=final_string[-num_tokens:]
                    
                print(ids_super_token,len(final_string))
                final_string = "".join(final_string)
 
                final_string=clip.tokenize(final_string).to(device)
                prob=eval_text_textual(CLIP_model=clip_model,concept_designed=final_string,concept_list=clip_concepts_matrix,
                                    cosine_function=cosine)

                
                textual_concept_probabilities.append(prob[0].cpu())   
            else:
                # textual_concept_probabilities.append(None)   
                textual_concept_probabilities.append(np.zeros(len(clip_concepts_matrix), dtype=np.uint8))   
                
            

            
        average_miner=np.mean(textual_concept_probabilities,axis=0)
        variance_miner=np.var(textual_concept_probabilities,axis=0)
        prob_dictionary[neuron_number]=[average_miner.tolist(),variance_miner.tolist()]
                

                    

    json.dump(prob_dictionary, open(embedding_path + 'prob_concept_textual_dictionary_ViT32_segmentation.json', 'a'), indent=4)

@gin.configurable
def SAE_fvu(dataset_path,embedding_path, device='cpu', log=False):
    """
    Computes Fraction of Variance Unexplained (FVU) metrics for visual, textual, and combined hidden states
    Args:
        path_dataset (str): Path to the dataset cache directory.
        path_embeddings (str): Directory where FVU results will be saved.
        device (str, optional): Device to run the SAE model on ('cpu' or 'cuda'). Defaults to 'cpu'.
        log (bool, optional): If True, computes and logs FVU statistics after processing. Defaults to False.

  
    """
    
     
    # Load the full dataset and split into subsets
    full_dataset = load_dataset(
        "lmms-lab/LLaVA-NeXT-Data", split="train", cache_dir=dataset_path
    )
    num_subsets = 15
    subset_size = len(full_dataset) // num_subsets
    data_subsets = [
        full_dataset.select(range(i * subset_size, (i + 1) * subset_size))
        for i in range(num_subsets)
    ]
    data = data_subsets[0]  # Use the first subset

    # Model and processor setup
    processor = LlavaNextProcessor.from_pretrained(
        "llava-hf/llama3-llava-next-8b-hf"
    )
    model = LlavaNextForConditionalGeneration.from_pretrained(
        "llava-hf/llama3-llava-next-8b-hf",
        attn_implementation="sdpa",
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_4bit=True,
    )
    model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
    model.eval()

    # Hook to capture hidden states
    hooked_res = {"hidden_states": None}

    def forward_hook(model, input, output):
        if hooked_res["hidden_states"] is not None:
            hooked_res["hidden_states"].append(output)
        else:
            hooked_res["hidden_states"] = [output]
        return output

    hook = model.language_model.model.layers[24].register_forward_hook(forward_hook)

    # Get the image tag token id
    image_tag = processor(text="<image>")["input_ids"][0][1]

    # Load the Sparse Autoencoder (SAE)
    sae = SAE.load_from_hub(
        "lmms-lab/llama3-llava-next-8b-hf-sae-131k", hookpoint="model.layers.24"
    ).to(device)
    sae.eval()

    # Prepare lists to store FVU valuess
    fvu_image, fvu_text, fvu_combined = [], [], []
    count = 0
    results_dict = {}

    for batch in tqdm(data, desc="Extract from LlaVA"):
        hooked_res["hidden_states"] = None
        with torch.inference_mode():
            id_dictionary = batch["id"]
            text = batch["conversations"][1]["value"]

            if id_dictionary in results_dict:
                continue

            if batch["image"]:
                img = batch["image"].resize((336, 336))
                img_conversation = [
                    {
                        "role": "system",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": text},
                        ],
                    },
                ]
                prompt = processor.apply_chat_template(
                    img_conversation, add_generation_prompt=True
                )
                inputs = processor(
                    images=img, text=prompt, return_tensors="pt"
                ).to(model.device)

                output = model(
                    input_ids=inputs["input_ids"].to(model.device),
                    pixel_values=inputs["pixel_values"].to(model.device),
                    image_sizes=inputs["image_sizes"].to(model.device),
                    attention_mask=inputs["attention_mask"].to(model.device),
                )

                hidden_state = hooked_res["hidden_states"][0]
                input_ids = inputs["input_ids"][0]

                # Find indices for visual and textual tokens
                indices_visual = torch.where(input_ids == image_tag)[0]
                indices_textual = torch.where(input_ids != image_tag)[0]

                # Limit visual indices to match textual length
                indices_visual = indices_visual[: len(indices_textual)]

                hidden_state_ = hidden_state[0][0].to(sae.device)
                result_visual = sae(hidden_state_[indices_visual])
                result_textual = sae(hidden_state_[indices_textual])
                result_combined = sae(hidden_state_)

                fvu_image.append(result_visual.fvu.item())
                fvu_text.append(result_textual.fvu.item())
                fvu_combined.append(result_combined.fvu.item())
                count += 1

                torch.cuda.empty_cache()
                #We execute this test for 10000 samples
                if count == 10000:
                    break

    # Save FVU values and compute statistics (optional)
    with open(embedding_path+"fvu_textual_values.json", "a") as f:
        json.dump(fvu_text, f)
    

    with open(embedding_path+"fvu_visual_values.json", "a") as f:
        json.dump(fvu_image, f)
    if log:   
        compute_fvu(embedding_path+"fvu_textual_values.json")
        compute_fvu(embedding_path+"fvu_visual_values.json")

@gin.configurable
def eval_visual_hypotheses_coco(path_embeddings:Path,path_dataset:Path,path_labels:Path,path_dictionary_neurons:Path,
                            device:torch.cuda.device='cuda:0')->None:
    """Evaluate the visual hypotheses using CLIP 

    Args:
        path_embeddings (Path):  Path to the folder containing embedding files
        path_dataset (Path):  Path to the dataset directory (COCO)
        path_labels (Path): Path to the dataset labels (COCO)
        path_dictionary_neurons (Path): Path to the JSON file with neuron-to-image mappings
        device (torch.cuda.device, optional): Device to run computations on. Defaults to 'cuda:0'
        
    Retrurns:
        None: Saves the results in JSON files, one for visual concepts and one for categories.
    """    
    
   
    
    dictionary_sae_neurons=json.load(open(path_dictionary_neurons,'r'))
    needed_ids = set()
    for _, batch in dictionary_sae_neurons.items():  
        needed_ids.update(map(int, batch.keys()))
    lookup = {}
    # VQA labels
    data=json.load(open(path_labels))
    images=[]
    for example in tqdm(data, desc="Building lookup", leave=False):
        img_id = int(example["id"])
        if img_id in needed_ids:
            
       
            lookup[img_id] = {
                
                "conversations": data["conversations"],
                "image": example["image"].convert('RGB')
            
            }
        if len(lookup) >= len(needed_ids):
            break
    hypotheses_path = path_embeddings + 'dictionary_hypotheses_complete_visual.json'
    if not os.path.exists(hypotheses_path):
        unite_dictionaries(path_embeddings,modality='visual')   
    dictionary_hypotheses = json.load(open(hypotheses_path, 'r'))
    dictionary_concepts,list_concepts=cleaning_hypotheses(dictionary_hypotheses)

    # Filter out concepts containing "pixelated" from both dictionary_concepts and list_concepts
    filtered_dictionary_concepts = {k: v for k, v in dictionary_concepts.items() if "pixelated" not in v.lower()}
    list_concepts = [concept for concept in filtered_dictionary_concepts.values()]
    dictionary_concepts = filtered_dictionary_concepts
    
    clip_model,preprocess=clip.load("ViT-B/32",device=device)
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
            
            # reconstructed_array = reconstruct_image(patches, zeros)
            reconstructed_array = reconstruct_image_blurring(patches, zeros)
                
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
                
   
                    
    
    json.dump(prob_dictionary, open(path_embeddings + 'prob_concept_visual_dictionary_Vit32.json', 'a'), indent=4)
    json.dump(prob_dictionary_categories, open(path_embeddings + 'prob_concept_visual_categories_dictionary_Vit32.json', 'a'), indent=4)



def encode_examples(examples,explanation):
    processed_inputs=f"""Latent explanation: {explanation}"""
    processed_inputs+="""\n Test examples: \n\n"""
    
    for i,example in enumerate(examples):
        processed_inputs+=f"""Example {i}: {example} \n\n"""
        
    
    return processed_inputs

def create_prompt_llama_detection (examples,explanation):
    processed_inputs=encode_examples(examples,explanation)
    default = [
        {"role": "system", "content": DSCORER_SYSTEM_PROMPT},
        {"role": "user", "content": DSCORER_EXAMPLE_ONE},
        {"role": "assistant", "content": DSCORER_RESPONSE_ONE},
        {"role": "user", "content": DSCORER_EXAMPLE_TWO},
        {"role": "assistant", "content": DSCORER_RESPONSE_TWO},
        {"role": "user", "content": DSCORER_EXAMPLE_THREE},
        {"role": "assistant", "content": DSCORER_RESPONSE_THREE},
        {"role": "user", "content": processed_inputs},
        
        
    ]
    return default

def create_prompt_llama_fuzzing (examples,explanation):
    processed_inputs=encode_examples(examples,explanation)
    default = [
        {"role": "system", "content": FSCORER_SYSTEM_PROMPT},
        {"role": "user", "content": FSCORER_EXAMPLE_ONE},
        {"role": "assistant", "content": FSCORER_RESPONSE_ONE},
        {"role": "user", "content": FSCORER_EXAMPLE_TWO},
        {"role": "assistant", "content": FSCORER_RESPONSE_TWO},
        {"role": "user", "content": FSCORER_EXAMPLE_THREE},
        {"role": "assistant", "content": FSCORER_RESPONSE_THREE},
        {"role": "user", "content": processed_inputs},
        
        
    ]
    return default
def neuron_ids_excluding_target(neuron_list: list, neuron_target:str,max_samples: int = 5) -> list:
    
    num_samples = random.randint(0, max_samples)
    neuron_list.remove(neuron_target)
    
    possible_random_neuron=random.sample(neuron_list, num_samples)

    return possible_random_neuron
## LLaVA ##
def detection_score_llava(dataset_path:Path,embedding_path:Path,dictionary_textual_concepts_path:Path,dictionary_neurons_path:Path,device:torch.cuda.device='cuda:0'):
    """Compute the detection score on the llava dataset

    Args:
        dataset_path (Path):Path to the dataset folder
        embedding_path (Path):Path to the embedding folder
        dictionary_neurons_path (Path): Path to the dictionary_neurons dictionary (with activations and texts)
        dictionary_textual_concepts_path (Path): Path to the dictionary_textual_concepts (with concepts hypotheses)
        device (_type_, optional): Device used. Defaults to 'cuda:0'.
    
    Return
        none: Save the scores on a json file in embedding_path folder
    """      
    # Load of the model
    tokenizer = AutoTokenizer.from_pretrained("hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4")
    model = AutoModelForCausalLM.from_pretrained("hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4")
    model.to(device)
    model.eval()
    
    
    # Prepare the data
    dictionary_complete_textual=json.load(open(dictionary_textual_concepts_path,'r'))
    output_path=embedding_path+'dictionary_autointerpretability_detection_llava.json'
    data = load_dataset("lmms-lab/LLaVA-NeXT-Data", split="train[:15%]", cache_dir=dataset_path, num_proc=10)
    
    neuron_ids=dictionary_complete_textual.keys()
    dictionary_neurons=json.load(open(dictionary_neurons_path,'r'))
    
    portion_dictionary_neurons={k: dictionary_neurons[k] for k in neuron_ids if k in dictionary_neurons}
    
    #Create a lookup
    needed_ids = set()
    for _, batch in portion_dictionary_neurons.items():  # limit to 1000 for progress bar
        needed_ids.update(map(int, batch.keys()))
    lookup = {}
    for example in tqdm(data, desc="Building lookup", leave=False):
        id_sample = int(example["id"])
        if id_sample in needed_ids:
            lookup[id_sample] = {
                "conversations": example["conversations"],
                
            }
        if len(lookup) >= len(needed_ids):
            break
    
    dictionary_labels={}
    for neuron_number,batch in tqdm(portion_dictionary_neurons.items(),desc='Evaluate Textual Hypotheses' ,total=len(portion_dictionary_neurons),
                                leave=False):
        examples=[]
        # Number of wrong samples, random between 0 and 5, each wrong sample is from a different neuron
        other_neurons = neuron_ids_excluding_target(list(dictionary_complete_textual.keys()),neuron_number)
        example_labels=[]
        for neuron in other_neurons:
            #Samples between the neuron examples
            new_batch = portion_dictionary_neurons[neuron]
            
            # Keep sampling until we find a key not in batch, i.e. a sample not used for the generation 
            finisher=0
            while True:
                
                wrong_key = random.choice(list(new_batch.keys()))  # pick 1 random key
                if wrong_key not in batch :  # check against batch keys
                    wrong_example_feats = new_batch[wrong_key]
                    
                    examples.append(wrong_example_feats["textual_features"]["final_output"])
                    example_labels.append('0')
                    break 
                finisher+=1
                if finisher==5:
                    break
            
        for _,feats in batch.items():
            if len(examples)==5:
                break
            examples.append(feats["textual_features"]["final_output"])
            example_labels.append('1')
            
        
            
            
        
        
        result=[]
        explanation=dictionary_complete_textual[neuron_number]
        prompt_text=create_prompt_llama_detection(examples=examples,explanation=explanation)
        
        
        inputs = tokenizer.apply_chat_template(
            prompt_text,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        
        
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=40)
        output_labels=tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])
        dictionary_labels[neuron_number]={"Real_labels":str(example_labels),"Output_labels":output_labels}

   
    json.dump(dictionary_labels, open(output_path,'a'), indent=4)

  
    # embedding_path='/data/lanza/coco_new_exp/'
    # label_path='/informatik3/wtm/datasets/External Datasets/coco_captions/labels_VQA/vqaX_train.json'
    # dictionary_neurons_path='/data/lanza/coco_new_exp/dictionary_neurons_textual.json'
    #dictionary_textual_concepts_path='/data/lanza/coco_new_exp/dictionary_hypotheses_complete_textual.json'
## COCO ##
def detection_score_coco(embedding_path:Path,dictionary_textual_concepts_path:Path,dictionary_neurons_path:Path,device:torch.cuda.device='cuda:0'):
    """Compute the detection score on the coco dataset

    Args:
        embedding_path (Path):Path to the embedding folder
        dictionary_neurons_path (Path): Path to the dictionary_neurons dictionary (with activations and texts)
        dictionary_textual_concepts_path (Path): Path to the dictionary_textual_concepts (with concepts hypotheses)
        device (_type_, optional): Device used. Defaults to 'cuda:0'.
    
    Return
        none: Save the scores on a json file in embedding_path folder
    """      
     

   
   

    
    if not os.path.exists(dictionary_textual_concepts_path):
        unite_dictionaries(embedding_path,modality='textual')    
    dictionary_complete_textual=json.load(open(dictionary_textual_concepts_path,'r'))
    neuron_ids=dictionary_complete_textual.keys()
    dictionary_neurons=json.load(open(dictionary_neurons_path,'r'))
    portion_dictionary_neurons={k: dictionary_neurons[k] for k in neuron_ids if "No textual " not in dictionary_complete_textual[k] and k in dictionary_neurons}
    

    path_output=embedding_path+'dictionary_autointerpretability_detection_coco.json'

    
    
 
    #Load the model
    tokenizer = AutoTokenizer.from_pretrained("hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4")
    model = AutoModelForCausalLM.from_pretrained("hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4")
    model.to(device)
    model.eval()
    
    dictionary_labels={}
    for neuron_number,batch in tqdm(portion_dictionary_neurons.items(),desc='Evaluate Textual Hypotheses' ,total=len(portion_dictionary_neurons),
                                leave=False):
        examples=[]
        # Number of wrong samples, random between 0 and 5, each wrong sample is from a different neuron
        other_neurons = neuron_ids_excluding_target(list(portion_dictionary_neurons.keys()),neuron_number)
        example_labels=[]
        for neuron in other_neurons:
            #Samples between the neuron examples
            new_batch = portion_dictionary_neurons[neuron]
            # Keep sampling until we find a key not in batch, i.e. a sample not used for the generation 
            finisher=0
            while True:
                
                wrong_key = random.choice(list(new_batch.keys()))  # pick 1 random key
                if wrong_key not in batch :  # check against batch keys
                    wrong_example_feats = new_batch[wrong_key]
                    
                    examples.append(wrong_example_feats["textual_features"]["final_output"])
                    example_labels.append('0')
                    break 
                finisher+=1
                if finisher==5:
                    break
            
        for _,feats in batch.items():
            if len(examples)==5:
                break
            examples.append(feats["textual_features"]["final_output"])
            example_labels.append('1')
            
        
            
        explanation=dictionary_complete_textual[neuron_number]
        prompt_text=create_prompt_llama_detection(examples=examples,explanation=explanation)
        
        
        inputs = tokenizer.apply_chat_template(
            prompt_text,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        
        
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=40)
        output_labels=tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])
        dictionary_labels[neuron_number]={"Real_labels":str(example_labels),"Output_labels":output_labels}

   
    json.dump(dictionary_labels, open(path_output,'a'), indent=4)


def highlight_random_word(text: str) -> str:
    """
    Pick a random word in the string and surround it with << >>.

    Args:
        text (str): The input sentence.

    Returns:
        str: The modified sentence with one word surrounded by << >>.
    """
    words = text.split()
    if not words:
        return text  # handle empty string
    
    # Pick a random word index
    idx = random.randint(0, len(words) - 1)
    
    # Surround the chosen word
    words[idx] = f"<<{words[idx]}>>"
    
    # Reconstruct the string
    return " ".join(words)
## LLaVA###
@gin.configurable
def fuz_score_llava( embedding_path:Path,dictionary_neurons_path:Path,dictionary_textual_concepts_path:Path,device:torch.cuda.device='cuda:0'):
        """Compute the fuzzing score on the llava-next dataset

        Args:
            embedding_path (Path):Path to the embedding folder.
            dictionary_neurons_path (Path): Path to the dictionary_neurons dictionary (with activations and texts).
            dictionary_textual_concepts_path (Path): Path to the dictionary_textual_concepts (with concepts hypotheses).
            device (_type_, optional): Device used. Defaults to 'cuda:0'.
        Return
            none: Save the scores on a json file in embedding_path folder.
        """   

        
        
        
        # Load the model
        tokenizer = AutoTokenizer.from_pretrained("hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4")
        model = AutoModelForCausalLM.from_pretrained("hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4")
        model.to(device)
        model.eval()
        
        #Prepare the data
        output_path=embedding_path+'dictionary_autointerpretability_fuzzing_llava.json'
        dictionary_complete_textual=json.load(open(dictionary_textual_concepts_path,'r'))
        neuron_ids=dictionary_complete_textual.keys()
        dictionary_neurons=json.load(open(dictionary_neurons_path,'r'))
        
        portion_dictionary_neurons={k: dictionary_neurons[k] for k in neuron_ids if k in dictionary_neurons}
        needed_ids = set()
        for _, batch in portion_dictionary_neurons.items():  # limit to 1000 for progress bar
            needed_ids.update(map(int, batch.keys()))
    
        
        dictionary_labels={}
        for neuron_number,batch in tqdm(portion_dictionary_neurons.items(),desc='Evaluate Textual Hypotheses' ,total=len(portion_dictionary_neurons),
                                    leave=False):
            examples=[]
            # Number of wrong samples, random between 0 and 5, each wrong sample is from a different neuron
            other_neurons = neuron_ids_excluding_target(list(dictionary_complete_textual.keys()),neuron_number)
            example_labels=[]
            for neuron in other_neurons:
                #Samples between the neuron examples
                new_batch = portion_dictionary_neurons[neuron]
                
                # Keep sampling until we find a key not in batch, i.e. a sample not used for the generation 
                finisher=0# Ensure the end with this flag
                while True:
                    
                    wrong_key = random.choice(list(new_batch.keys()))  # pick 1 random key
                    if wrong_key not in batch :  # check against batch keys
                        wrong_example_feats = new_batch[wrong_key]
                        final_string=highlight_random_word (wrong_example_feats["textual_features"]["final_output"])
                        examples.append(final_string)
                        example_labels.append('0')
                        break 
                    finisher+=1
                    if finisher==5:
                        break
                
            for _,feats in batch.items():
            
                if len(examples)==5:
                    break
                final_string=feats["textual_features"]["final_output"]
                examples.append(feats["textual_features"]["final_output"])
                example_labels.append('1')
                
                        
            explanation=dictionary_complete_textual[neuron_number]
            prompt_text=create_prompt_llama_fuzzing(examples=examples,explanation=explanation)
            
            
            inputs = tokenizer.apply_chat_template(
                prompt_text,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)
            
            
            with torch.inference_mode():
                outputs = model.generate(**inputs, max_new_tokens=40)
            output_labels=tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])
            dictionary_labels[neuron_number]={"Real_labels":str(example_labels),"Output_labels":output_labels}

    
        json.dump(dictionary_labels, open(output_path,'a'), indent=4)   
## COCO###
@gin.configurable
def fuz_score_coco( embedding_path:Path,dictionary_neurons_path:Path,dictionary_textual_concepts_path:Path,device:torch.cuda.device='cuda:0'):
    """Compute the fuzzing score on the coco dataset

    Args:
        embedding_path (Path):Path to the embedding folder.
        dictionary_neurons_path (Path): Path to the dictionary_neurons dictionary (with activations and texts).
        dictionary_textual_concepts_path (Path): Path to the dictionary_textual_concepts (with concepts hypotheses).
        device (_type_, optional): Device used. Defaults to 'cuda:0'.
    Return
        none: Save the scores on a json file in embedding_path folder.
    """    

    #Load the model
    
    tokenizer = AutoTokenizer.from_pretrained("hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4")
    model = AutoModelForCausalLM.from_pretrained("hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4")
    model.to(device)
    model.eval()
    #Prepare the data
    if not os.path.exists(dictionary_textual_concepts_path):
        unite_dictionaries(embedding_path,modality='textual')    
    # Dictionary with hypotheses
    dictionary_complete_textual=json.load(open(dictionary_textual_concepts_path,'r'))
    #Dictionary with texts used to extract the concepts
    dictionary_neurons=json.load(open(dictionary_neurons_path,'r'))
    neuron_ids=dictionary_complete_textual.keys()
    
    # Only extract the needed neuron 
    portion_dictionary_neurons={k: dictionary_neurons[k] for k in neuron_ids if "No textual " not in dictionary_complete_textual[k] and k in dictionary_neurons}
    output_path=embedding_path+'dictionary_autointerpretability_fuzzing_coco.json'


  

    dictionary_labels={}
    for neuron_number,batch in tqdm(portion_dictionary_neurons.items(),desc='Evaluate Textual Hypotheses Fuzzing' ,total=len(portion_dictionary_neurons),
                                leave=False):
        examples=[]
        # Number of wrong samples, random between 0 and 5, each wrong sample is from a different neuron
        other_neurons = neuron_ids_excluding_target(list(portion_dictionary_neurons.keys()),neuron_number)
        
        example_labels=[]
        for neuron in other_neurons:
            #Samples between the neuron examples
            new_batch = portion_dictionary_neurons[neuron]
            
            # Keep sampling until we find a key not in batch, i.e. a sample not used for the generation 
            finisher=0# Ensure the end with this flag
            while True:
                
                wrong_key = random.choice(list(new_batch.keys()))  # pick 1 random key
                if wrong_key not in batch :  # check against batch keys
                    wrong_example_feats = new_batch[wrong_key]
                    final_string=highlight_random_word (wrong_example_feats["textual_features"]["final_output"])
                    examples.append(final_string)
                    example_labels.append('0')
                    break 
                finisher+=1
                if finisher==5:
                    break
            
        for _,feats in batch.items():
        
            if len(examples)==5:
                break
            final_string=feats["textual_features"]["final_output"]
            examples.append(feats["textual_features"]["final_output"])
            example_labels.append('1')
            
        
            
        explanation=dictionary_complete_textual[neuron_number]
        prompt_text=create_prompt_llama_fuzzing(examples=examples,explanation=explanation)
        
        
        inputs = tokenizer.apply_chat_template(
            prompt_text,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        
        
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=40)
        output_labels=tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])
        dictionary_labels[neuron_number]={"Real_labels":str(example_labels),"Output_labels":output_labels}

   
    json.dump(dictionary_labels, open(output_path,'a'), indent=4)
