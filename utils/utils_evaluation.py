import os
os.environ["TOKENIZERS_PARALLELISM"]="false"
import torch
import json
from typing import  List, Dict
from tqdm import tqdm
from pathlib import Path
import numpy as np
from PIL import Image
import torch.nn as nn
import glob
import clip
import glob
import gin
from utils.utils_image import create_image_patches
from datasets import load_dataset 
from utils.api import unite_dictionaries
from utils.utils_image import reconstruct_image_blurring
from utils.api import compute_fvu
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from sparsify.sparsify.sparse_coder import SparseCoder as SAE
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AlignProcessor, AlignModel
from utils.utils_prompt import DSCORER_SYSTEM_PROMPT, DSCORER_EXAMPLE_ONE, DSCORER_EXAMPLE_TWO,DSCORER_EXAMPLE_THREE
from utils.utils_prompt import  DSCORER_RESPONSE_ONE, DSCORER_RESPONSE_TWO,DSCORER_RESPONSE_THREE
from utils.utils_prompt import FSCORER_SYSTEM_PROMPT, FSCORER_EXAMPLE_ONE, FSCORER_EXAMPLE_TWO,FSCORER_EXAMPLE_THREE
from utils.utils_prompt import  FSCORER_RESPONSE_ONE, FSCORER_RESPONSE_TWO,FSCORER_RESPONSE_THREE
from utils.utils_prompt import GUIDELINES_LABELING
from api import model_generation, initialize_llava
import random
def similarity_text_image_ALIGN(ALIGN_model:AlignModel, processor:AlignProcessor, concept_list:list, image:Image, device:torch.cuda.device ):
    """
    Compute the similarity between an image and a list of textual concepts using ALIGN.

    Args:
        ALIGN_model (AlignModel): The ALIGN model used for encoding and similarity computation.
        processor (AlignProcessor): The processor used for preprocessing inputs.
        texts (list): A list of textual concepts to compare against the image.
        image (Image): The input image to be evaluated.
       

    Returns:
        np.ndarray: Probabilities representing the similarity between the image and each concept.
    """
    # Process image and text inputs    
    
    inputs = processor(images=image, text=concept_list, return_tensors="pt",
                        padding="max_length", max_length=15).to(device)
    
    # Get the embedding
    with torch.no_grad():
        outputs = ALIGN_model(**inputs)


    image_embeds = outputs.image_embeds
    text_embeds = outputs.text_embeds

  

    # Normalize
    image_embeds = image_embeds / image_embeds.norm(dim=1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(dim=1, keepdim=True)

    # Cosine similarities
    # For each text, compute dot(text_norm, image_norm)
    similarity_scores = torch.matmul(text_embeds, image_embeds[0])  # shape (N_texts,)
    probs = torch.nn.functional.softmax(similarity_scores, dim=0)


    
    return  probs
def similarity_text_text_ALIGN(ALIGN_model:AlignModel,concept_designed:str, concept_list:List,cosine_function:nn.CosineSimilarity ):
    """
    Compute the similarity between an image and a list of textual concepts using ALIGN.

    Args:
        ALIGN_model (AlignModel): The ALIGN model used for encoding and similarity computation.
        processor (AlignProcessor): The processor used for preprocessing inputs.
        texts (list): A list of textual concepts to compare against the image.
        image (Image): The input image to be evaluated.
       

    Returns:
        np.ndarray: Probabilities representing the similarity between the image and each concept.
    """
    # Process image and text inputs    
    
    similarity_list=[]
    with torch.inference_mode():
        
        text_features_concept = ALIGN_model.encode_text(concept_designed)
        
        cos=cosine_function(text_features_concept, concept_list)
        
        similarity_list.append(cos)
        
    return similarity_list
 
def similarity_text_image_CLIP(CLIP_model:clip,preprocess, image:Image, concept_list:list)->np.ndarray:
    """
    Compute the similarity between an image and a list of textual concepts using a CLIP model.

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

        logits_per_image, _ = CLIP_model(image, concept_list)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        
     
    
    return probs

def similarity_text_text_CLIP(CLIP_model:clip,concept_designed:str, concept_list:List,cosine_function:nn.CosineSimilarity)->List[float]:
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
    similarity_list=[]
    with torch.inference_mode():
        
        text_features_concept = CLIP_model.encode_text(concept_designed)
        
        cos=cosine_function(text_features_concept, concept_list)
        
        similarity_list.append(cos)
        
    return similarity_list


def cleaning_hypotheses(hypothesis_dictionary: Dict) -> tuple[Dict, List]:
    """Cleans a dictionary of hypotheses by removing repeated and hallucinated long concepts.

    Args:
        hypothesis_dictionary: (Dict):Dictionary with hypotheses as string

    Tuple[Dict, List]: 
        - A dictionary with same keys as input, but only unique, valid concepts as values.
        - A list of unique, valid concepts extracted from the input dictionary.
    """    
    cleaned_hypothesis_dictionary={}  
    list_concepts=[]
   
    
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
        
                    if concept not in list_concepts:
                            
                        list_concepts.append(concept)
                        cleaned_hypothesis_dictionary[key]=concept
                        
    return [cleaned_hypothesis_dictionary,list_concepts]
def generate_labelling(embedding_path:Path,concept_dictionary)->None:
    """
    Generate candidate labels for neurons using provided concept mappings.

    Args:
        embedding_path (Path): directory containing the saved SAE latent activations.
        concept_dictionary (dict): mapping from neuron identifiers to associated concepts.

    Returns:
        None. Writes the generated label dictionary to a JSON.
    """

    save_result_path= embedding_path + 'concept_label_dictionary.json'
    
    # Mdels
    processor = LlavaNextProcessor.from_pretrained("llava-hf/llama3-llava-next-8b-hf")
    model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llama3-llava-next-8b-hf",attn_implementation="sdpa", torch_dtype=torch.float16, device_map="auto")
    model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
    model.eval()
    
   
    
    # Initialize the conversation with the system prompt
    system="""You are a meticulous AI researcher conducting an important investigation into a certain neuron in a vision language model."""
    actual_conversation=[
                        {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": system},
                            
                            ],
                        },
                    ]
    actual_conversation=initialize_llava(model=model,processor=processor,actual_conversation=actual_conversation)
    
     # Result dictionary
    label_dictionary={str(i): [] for i in range(5000)}
    for neuron_number, definition in tqdm(concept_dictionary, desc="Labeling the concepts", leave=True):
    


        content=GUIDELINES_LABELING.format(question=definition)
        
        actual_conversation,output_text,_=model_generation(model,actual_conversation,
                   content,processor,images=None,max_new_tokens=20)

        actual_conversation.pop()
            
        actual_conversation.pop()
        
        label_dictionary[neuron_number] = output_text
        torch.cuda.empty_cache()

        
    with open(save_result_path, 'w') as json_file:
        json.dump(label_dictionary, json_file, indent=4)
@gin.configurable
def eval_visual_hypotheses_CLIP(embedding_path:Path,dataset_path:Path,neuron_top5_visual_dictionary_path:Path,
                            device:torch.cuda.device='cuda:0')->None:
    """Evaluate the visual hypotheses using CLIP 

    Args:
        embedding_path (Path):  Path to the folder containing embedding files
        dataset_path(Path):  Path to the dataset cache directory
        neurons_top5_visual_dictionary_path (Path): Path to the JSON file with neuron-to-image mappings
        device (torch.cuda.device, optional): Device to run computations on. Defaults to 'cuda:0'
        
    Retrurns:
        None: Saves the results in JSON files, one for visual concepts and one for categories.
    """    
    
    data = load_dataset("lmms-lab/LLaVA-NeXT-Data", split="train[:15%]", cache_dir=dataset_path, num_proc=10)
    image_neuron_dictionary=json.load(open(neuron_top5_visual_dictionary_path+'neuron_top5_visual_dictionary.json','r'))
    clip_model,preprocess=clip.load("ViT-B/32",device=device)
    clip_model.eval()
    
    
    
    
    needed_ids = set()
    # Create the lookup dictionary
    for _, batch in image_neuron_dictionary.items():  
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
    hypotheses_path = embedding_path + 'visual_hypothesis_dictionary.json'
    if not os.path.exists(hypotheses_path):
        unite_dictionaries(embedding_path,modality='visual')   
        
    # Filter out concepts
    hypothesis_dictionary = json.load(open(hypotheses_path, 'r'))
    concept_dictionary,concept_list=cleaning_hypotheses(hypothesis_dictionary)

   
    filtered_concept_dictionary = {k: v for k, v in concept_dictionary.items() if "pixelated" not in v.lower()}
    concept_list = [concept for concept in filtered_concept_dictionary.values()]
    concept_dictionary = filtered_concept_dictionary
    
    
    
    #Results dictionaries
    prob_dictionary={}
    prob_dictionary_categories={}
    #Concepts encoded by clip
    concepts_clip_list=clip.tokenize([concept for concept in concept_list]).to(device)
    
    #Categories taken by Bau et al. 2017 paper
    categories = ["scene", "object", "part", "material", "texture", "color"]
    categories_clip_list=clip.tokenize([label for label in categories]).to(device)
    
    for neuron_number,_ in tqdm(concept_dictionary.items(),desc='Evaluate the visual hypothesis' ,total=len(concept_dictionary),leave=True):
        batch=image_neuron_dictionary[neuron_number]
        visual_concept_probabilities, category_probabilities =[],[]
        for img_id_str, feats in batch.items():
            img_id = int(img_id_str)
            entry = lookup.get(img_id)
            
            if entry is None:
                continue


            # mask out patches
            image = entry["image"]
            # Process image patches
            patches = create_image_patches(image)
                    
            
            # Create masked image based on neuron activation
            
            zeros = np.zeros(len(patches), dtype=np.uint8)
            for patch_idx, inds in enumerate(feats["visual_features"]["latent_indices"]):
                if int(neuron_number) in inds:
                    zeros[patch_idx] = 1
            
            reconstructed_array = reconstruct_image_blurring(patches, zeros)
                
            prob=similarity_text_image_CLIP(CLIP_model=clip_model,preprocess=preprocess,concept_list=concepts_clip_list,
                                image=Image.fromarray(reconstructed_array),device=device)
            
            prob_categories = similarity_text_image_CLIP(CLIP_model=clip_model, preprocess=preprocess, concept_list=categories_clip_list,
                                               image=Image.fromarray(reconstructed_array), device=device)
            visual_concept_probabilities.append(prob[0])   
            category_probabilities.append(prob_categories[0])
            
        # Collect the average and varience of probabilities
  
        average_concepts_vector=np.mean(visual_concept_probabilities,axis=0)
        variance_concepts_vector=np.var(visual_concept_probabilities,axis=0)
        
        average_categories_vector=np.mean(category_probabilities,axis=0)
        variance_categories_vector=np.var(category_probabilities,axis=0)
        
                
        prob_dictionary[neuron_number]=[average_concepts_vector.tolist(),variance_concepts_vector.tolist()]
        prob_dictionary_categories[neuron_number]=[average_categories_vector.tolist(),variance_categories_vector.tolist()]
                
    json.dump(prob_dictionary, open(embedding_path + 'prob_concept_visual_dictionary_Vit32.json', 'a'), indent=4)
    json.dump(prob_dictionary_categories, open(embedding_path + 'prob_concept_visual_categories_dictionary_Vit32.json', 'a'), indent=4)
    # For the categories test, we need to label each concept in the predifined categories 
    if os.path.exists(embedding_path + 'concept_label_dictionary.json'):
        generate_labelling(embedding_path,concept_dictionary)
@gin.configurable
def eval_visual_hypotheses_ALIGN(dataset_path:Path, embedding_path:Path,neuron_top5_visual_dictionary_path:Path,device:torch.cuda.device='cuda:0')->None:
    """Evaluate the visual hypotheses using Align 

    Args:
        embedding_path (Path):  Path to the folder containing embedding files
        dataset_path(Path):  Path to the dataset cache directory
        neurons_top5_visual_dictionary_path (Path): Path to the JSON file with neuron-to-image mappings
        device (torch.cuda.device, optional): Device to run computations on. Defaults to 'cuda:0'
        
    Retrurns:
        None: Saves the results in JSON files, one for visual concepts and one for categories.
    """    
   

    data = load_dataset("lmms-lab/LLaVA-NeXT-Data", split="train[:15%]", cache_dir=dataset_path, num_proc=10)
    hypotheses_path = embedding_path + 'visual_hypothesis_dictionary.json'


    hypothesis_dictionary = json.load(open(hypotheses_path, 'r'))
    processor_align = AlignProcessor.from_pretrained("kakaobrain/align-base")
    model_align = AlignModel.from_pretrained("kakaobrain/align-base")
    model_align.eval()
    model_align.to(device)


    sae_neurons_dictionary=json.load(open(neuron_top5_visual_dictionary_path+'neuron_top5_visual_dictionary.json','r'))
    needed_ids = set()
    for _, batch in sae_neurons_dictionary.items():  
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
    #Filtering of useless concepts (hallucinated and repeated long concepts)
    concepts_visual_dictionary,list_concepts_visual=cleaning_hypotheses(hypothesis_dictionary)
    filtered_dictionary_concepts = {k: v for k, v in concepts_visual_dictionary.items() if "pixelated" not in v.lower()}
    list_concepts_visual = [concept for concept in filtered_dictionary_concepts.values()]
    concepts_visual_dictionary = filtered_dictionary_concepts
    
    # Result dictionaries
    prob_dictionary_visual={}
    prob_dictionary_categories_visual={}
    # Categories taken by Bau et al. 2017 paper
    categories = ["scene", "object", "part", "material", "texture", "color"]

    for neuron_number,_ in tqdm(concepts_visual_dictionary.items(),desc='Evaluate the visual hypothesis' ,total=len(concepts_visual_dictionary),leave=True):
        batch=sae_neurons_dictionary[neuron_number]
        visual_concept_probabilities, category_probabilities =[],[]
        for img_id_str, feats in batch.items():
            img_id = int(img_id_str)
            entry = lookup.get(img_id)
            
            if entry is None:
                continue

          

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
            
            prob=similarity_text_image_ALIGN(model_align,processor_align,list_concepts_visual,image=Image.fromarray(reconstructed_array),device=device)
            prob_categories=similarity_text_image_ALIGN(model_align,processor_align,categories,image=Image.fromarray(reconstructed_array),device=device)
            visual_concept_probabilities.append(prob.cpu())   
            category_probabilities.append(prob_categories.cpu())
            
        # Collect the average and varience of probabilities
        average_concepts=np.mean(visual_concept_probabilities,axis=0)
        variance_concepts=np.var(visual_concept_probabilities,axis=0)

        average_categories=np.mean(category_probabilities,axis=0)
        variance_categories=np.var(category_probabilities,axis=0)
                
        prob_dictionary_visual[neuron_number]=[average_concepts.tolist(),variance_concepts.tolist()]
        prob_dictionary_categories_visual[neuron_number]=[average_categories.tolist(),variance_categories.tolist()]
        
        
                    
    
    json.dump(prob_dictionary_visual, open(embedding_path + 'prob_concept_visual_dictionary_Align.json', 'a'), indent=4)
    json.dump(prob_dictionary_categories_visual, open(embedding_path + 'prob_concept_visual_categories_dictionary_Align.json', 'a'), indent=4)
@gin.configurable
def eval_textual_hypotheses_CLIP(embedding_path:Path,neuron_top5_textual_dictionary_path:Path,device:torch.cuda.device='cuda:0')->None:
    """Evaluate textual hypotheses using CLIP 

    Args:
        embedding_path (Path):  Path to the folder containing embedding files
        dataset_path (Path):  Path to the dataset cache directory
        neurons_top5_textual_dictionary_path (Path): Path to the JSON file with neuron-to-image mappings
        device (torch.cuda.device, optional): Device to run computations on. Defaults to 'cuda:0'.
    Returns:
        None: Saves the results in JSON files, one for textual concepts and one for categories.
    """    
        
    clip_model,_=clip.load("ViT-B/32",device=device)
    clip_model.eval()

    clip_model.eval()
    cosine = nn.CosineSimilarity(dim=1, eps=1e-6)
    # Result dictionary
    prob_dictionary={} 
    
                         
    sae_neurons_dictionary=json.load(open(neuron_top5_textual_dictionary_path+'neuron_top5_textual_dictionary.json','r'))
    hypotheses_path = embedding_path + 'textual_hypotheses_dictionary.json'
    
    if not os.path.exists(hypotheses_path):
        unite_dictionaries(embedding_path,modality='textual')  
         
    hypothesis_dictionary = json.load(open(hypotheses_path, 'r'))
    
    # Filiterin out concepts
    concept_dictionary,concept_list=cleaning_hypotheses(hypothesis_dictionary)
    processor = LlavaNextProcessor.from_pretrained("llava-hf/llama3-llava-next-8b-hf")
    
    #Encode the concepts with CLIP
    tmp =[clip_model.encode_text(clip.tokenize(el).to(device)) for el in concept_list]
    clip_concepts_matrix=torch.cat(tmp,dim=0)

    for neuron_number,_ in tqdm(concept_dictionary.items(),desc='Evaluate the textual hypothesis' ,total=len(concept_dictionary),leave=True):
        batch=sae_neurons_dictionary[neuron_number]
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
            # Select window of tokens around the most activated one (usally is 10 tokens in total, 5 before and 5 after) to create the prompt for CLIP text encoder
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
                    
                final_string = "".join(final_string)
 
                final_string=clip.tokenize(final_string).to(device)
                prob=similarity_text_text_CLIP(CLIP_model=clip_model,concept_designed=final_string,concept_list=clip_concepts_matrix,
                                    cosine_function=cosine)

                
                textual_concept_probabilities.append(prob[0].cpu())   
            else:  
                textual_concept_probabilities.append(np.zeros(len(clip_concepts_matrix), dtype=np.uint8))   
                
            

            
        average_probability_vector=np.mean(textual_concept_probabilities,axis=0)
        variance_probability_vector=np.var(textual_concept_probabilities,axis=0)
        prob_dictionary[neuron_number]=[average_probability_vector.tolist(),variance_probability_vector.tolist()]
                

                    

    json.dump(prob_dictionary, open(embedding_path + 'prob_concept_textual_dictionary_ViT32_segmentation.json', 'a'), indent=4)
@gin.configurable
def eval_textual_hypotheses_ALIGN(dataset_path:Path, embedding_path:Path,neuron_top5_textual_dictionary_path:Path,device:torch.cuda.device='cuda:0')->None:
    """Evaluate the visual hypotheses using Align 

    Args:
        embedding_path (Path):  Path to the folder containing embedding files
        dataset_path(Path):  Path to the dataset cache directory
        neurons_top5_textual_dictionary_path (Path): Path to the JSON file with neuron-to-image mappings
        device (torch.cuda.device, optional): Device to run computations on. Defaults to 'cuda:0'
        
    Retrurns:
        None: Saves the results in JSON files, one for visual concepts and one for categories.
    """    
   

    data = load_dataset("lmms-lab/LLaVA-NeXT-Data", split="train[:15%]", cache_dir=dataset_path, num_proc=10)
    hypotheses_path = embedding_path + 'textual_hypothesis_dictionary.json'


    hypothesis_dictionary = json.load(open(hypotheses_path, 'r'))
    processor_align = AlignProcessor.from_pretrained("kakaobrain/align-base")
    model_align = AlignModel.from_pretrained("kakaobrain/align-base")
    model_align.eval()
    model_align.to(device)
    cosine = nn.CosineSimilarity(dim=1, eps=1e-6)


    sae_neurons_dictionary=json.load(open(neuron_top5_textual_dictionary_path+'neuron_top5_textual_dictionary.json','r'))
    needed_ids = set()
    for _, batch in sae_neurons_dictionary.items():  
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
        
    concept_dictionary,concept_list=cleaning_hypotheses(hypothesis_dictionary)
    processor = LlavaNextProcessor.from_pretrained("llava-hf/llama3-llava-next-8b-hf")
    #Filtering of useless concepts (hallucinated and repeated long concepts)
    tmp =[model_align.encode_text(model_align.tokenize(el).to(device)) for el in concept_list]
    clip_concepts_matrix=torch.cat(tmp,dim=0)
    prob_dictionary={}
    for neuron_number,_ in tqdm(concept_dictionary.items(),desc='Evaluate the textual hypothesis' ,total=len(concept_dictionary),leave=True):
        batch=sae_neurons_dictionary[neuron_number]
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
            # Select window of tokens around the most activated one (usally is 10 tokens in total, 5 before and 5 after) to create the prompt for CLIP text encoder
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
                    
                final_string = "".join(final_string)
 
                final_string=model_align.tokenize(final_string).to(device)
                prob=similarity_text_text_ALIGN(model_align,processor_align,final_string,clip_concepts_matrix,cosine)

                
                textual_concept_probabilities.append(prob[0].cpu())   
            else:  
                textual_concept_probabilities.append(np.zeros(len(clip_concepts_matrix), dtype=np.uint8))   
                
            

            
        average_probability_vector=np.mean(textual_concept_probabilities,axis=0)
        variance_probability_vector=np.var(textual_concept_probabilities,axis=0)
        prob_dictionary[neuron_number]=[average_probability_vector.tolist(),variance_probability_vector.tolist()]
        
                    
    
    json.dump(prob_dictionary, open(embedding_path + 'prob_concept_textual_dictionary_Align.json', 'a'), indent=4)
@gin.configurable
def SAE_fvu(dataset_path:Path,embedding_path:Path, device:torch.cuda.device='cuda:0', log=False):
    """
    Computes Fraction of Variance Unexplained (FVU) metrics for visual, textual, and combined hidden states
    Args:
        dataset_path (Path): Path to the dataset cache directory.
        embedding_path (Path): Directory where FVU results will be saved.
        device (str, optional): Device to run the SAE model on ('cpu' or 'cuda'). Defaults to 'cpu'.
  
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

def encode_examples(examples_list,explanation):
    processed_inputs=f"""Latent explanation: {explanation}"""
    processed_inputs+="""\n Test examples_list: \n\n"""
    
    for i,example in enumerate(examples_list):
        processed_inputs+=f"""Example {i}: {example} \n\n"""
        
    
    return processed_inputs

def create_prompt_llama_detection (examples_list,explanation):
    processed_inputs=encode_examples(examples_list,explanation)
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

def create_prompt_llama_fuzzing (examples_list,explanation):
    processed_inputs=encode_examples(examples_list,explanation)
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

def detection_score_llava(embedding_path:Path,device:torch.cuda.device='cuda:0'):
    """Compute the detection score on the llava dataset

    Args:
        embedding_path (Path):Path to the embedding folder
        device (torch.cuda.device, optional): Device used. Defaults to 'cuda:0'.
    
    Return
        none: Save the scores on a json file in embedding_path folder
    """      
     #Load the model and data
    tokenizer = AutoTokenizer.from_pretrained("hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4")
    model = AutoModelForCausalLM.from_pretrained("hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4")
    model.to(device)
    model.eval()
    
    # Dictionary with hypotheses
    concept_textual_dictionary=json.load(open(embedding_path + 'textual_hypothesis_dictionary.json','r'))
    #Dictionary with texts used to extract the concepts
    neuron_dictionary_path=embedding_path + 'average_activation_textual_dictionary.json'
    neuron_activation_dictionary=json.load(open(neuron_dictionary_path,'r'))
    neuron_ids=concept_textual_dictionary.keys()
    
    # Only extract the needed neuron 
    used_neurons_dictionary={k: neuron_activation_dictionary[k] for k in neuron_ids if "No textual " not in concept_textual_dictionary[k] and k in neuron_activation_dictionary}
    #Output path
    output_path=embedding_path+'autointerpretability_scores_llava_dictionary.json'
    
   
    
    label_dictionary={}
    for neuron_number,batch in tqdm(used_neurons_dictionary.items(),desc='Evaluate Textual Hypotheses' ,total=len(used_neurons_dictionary),
                                leave=False):
        examples_list=[]
        # Number of wrong samples, random between 0 and 5, each wrong sample is from a different neuron
        other_neurons = neuron_ids_excluding_target(list(concept_textual_dictionary.keys()),neuron_number)
        example_labels=[]
        for neuron in other_neurons:
            #Samples between the neuron examples_list
            new_batch = used_neurons_dictionary[neuron]
            
            # Keep sampling until we find a key not in batch, i.e. a sample not used for the generation 
            finisher=0
            while True:
                
                wrong_key = random.choice(list(new_batch.keys()))  # pick 1 random key
                if wrong_key not in batch :  # check against batch keys
                    wrong_example_feats = new_batch[wrong_key]
                    
                    examples_list.append(wrong_example_feats["textual_features"]["final_output"])
                    example_labels.append('0')
                    break 
                finisher+=1
                if finisher==5:
                    break
            
        for _,feats in batch.items():
            if len(examples_list)==5:
                break
            examples_list.append(feats["textual_features"]["final_output"])
            example_labels.append('1')
                 
        explanation=concept_textual_dictionary[neuron_number]
        prompt_text=create_prompt_llama_detection(examples_list=examples_list,explanation=explanation)
        
        
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
        label_dictionary[neuron_number]={"Real_labels":str(example_labels),"Output_labels":output_labels}

   
    json.dump(label_dictionary, open(output_path,'a'), indent=4)

def detection_score_coco(embedding_path:Path,device:torch.cuda.device='cuda:0'):
    """Compute the detection score on the coco dataset

    Args:
        embedding_path (Path):Path to the embedding folder
        device (torch.cuda.device, optional): Device used. Defaults to 'cuda:0'.
    
    Return
        none: Save the scores on a json file in embedding_path folder
    """      
    
    #Load the model and data
    tokenizer = AutoTokenizer.from_pretrained("hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4")
    model = AutoModelForCausalLM.from_pretrained("hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4")
    model.to(device)
    model.eval()
    
    # Dictionary with hypotheses
    concept_textual_dictionary=json.load(open(embedding_path + 'textual_hypothesis_dictionary.json','r'))
    
    #Dictionary with texts used to extract the concepts
    neuron_dictionary_path=embedding_path + 'average_activation_textual_dictionary.json'
    neuron_activation_dictionary=json.load(open(neuron_dictionary_path,'r'))
    neuron_ids=concept_textual_dictionary.keys()
    
    # Only extract the needed neuron 
    used_neurons_dictionary={k: neuron_activation_dictionary[k] for k in neuron_ids if "No textual " not in concept_textual_dictionary[k] and k in neuron_activation_dictionary}
    #Output path
    output_path=embedding_path+'autointerpretability_scores_coco_dictionary.json'
    
    
 
    #Load the model
    tokenizer = AutoTokenizer.from_pretrained("hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4")
    model = AutoModelForCausalLM.from_pretrained("hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4")
    model.to(device)
    model.eval()
    
    label_dictionary={}
    for neuron_number,batch in tqdm(used_neurons_dictionary.items(),desc='Evaluate Textual Hypotheses' ,total=len(used_neurons_dictionary),
                                leave=False):
        examples_list=[]
        # Number of wrong samples, random between 0 and 5, each wrong sample is from a different neuron
        other_neurons = neuron_ids_excluding_target(list(used_neurons_dictionary.keys()),neuron_number)
        example_labels=[]
        for neuron in other_neurons:
            #Samples between the neuron examples_list
            new_batch = used_neurons_dictionary[neuron]
            # Keep sampling until we find a key not in batch, i.e. a sample not used for the generation 
            finisher=0
            while True:
                
                wrong_key = random.choice(list(new_batch.keys()))  # pick 1 random key
                if wrong_key not in batch :  # check against batch keys
                    wrong_example_feats = new_batch[wrong_key]
                    
                    examples_list.append(wrong_example_feats["textual_features"]["final_output"])
                    example_labels.append('0')
                    break 
                finisher+=1
                if finisher==5:
                    break
            
        for _,feats in batch.items():
            if len(examples_list)==5:
                break
            examples_list.append(feats["textual_features"]["final_output"])
            example_labels.append('1')
            
        
            
        explanation=concept_textual_dictionary[neuron_number]
        prompt_text=create_prompt_llama_detection(examples_list=examples_list,explanation=explanation)
        
        
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
        label_dictionary[neuron_number]={"Real_labels":str(example_labels),"Output_labels":output_labels}

   
    json.dump(label_dictionary, open(output_path,'a'), indent=4)

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

@gin.configurable
def fuz_score_llava( embedding_path:Path,device:torch.cuda.device='cuda:0'):
        """Compute the fuzzing score on the llava-next dataset

        Args:
            embedding_path (Path):Path to the embedding folder.
            device (torch.cuda.device, optional): Device used. Defaults to 'cuda:0'.
        Return
            none: Save the scores on a json file in embedding_path folder.
        """   

        
        
        
        #Load the model
        tokenizer = AutoTokenizer.from_pretrained("hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4")
        model = AutoModelForCausalLM.from_pretrained("hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4")
        model.to(device)
        model.eval()
        
        # Dictionary with hypotheses
        concept_textual_dictionary=json.load(open(embedding_path + 'textual_hypothesis_dictionary.json','r'))
        
        #Dictionary with texts used to extract the concepts
        neuron_dictionary_path=embedding_path + 'average_activation_textual_dictionary.json'
        neuron_activation_dictionary=json.load(open(neuron_dictionary_path,'r'))
        neuron_ids=concept_textual_dictionary.keys()
        
        # Only extract the needed neuron 
        used_neurons_dictionary={k: neuron_activation_dictionary[k] for k in neuron_ids if "No textual " not in concept_textual_dictionary[k] and k in neuron_activation_dictionary}
        #Output path
        
        output_path=embedding_path+'autointerpretability_fuzzing_scores_llava_dictionary.json'
    
        needed_ids = set()
        for _, batch in used_neurons_dictionary.items():  # limit to 1000 for progress bar
            needed_ids.update(map(int, batch.keys()))
    
        
        dictionary_labels={}
        for neuron_number,batch in tqdm(used_neurons_dictionary.items(),desc='Evaluate Textual Hypotheses' ,total=len(used_neurons_dictionary),
                                    leave=False):
            examples_list=[]
            # Number of wrong samples, random between 0 and 5, each wrong sample is from a different neuron
            other_neurons = neuron_ids_excluding_target(list(concept_textual_dictionary.keys()),neuron_number)
            example_labels=[]
            for neuron in other_neurons:
                #Samples between the neuron examples_list
                new_batch = used_neurons_dictionary[neuron]
                
                # Keep sampling until we find a key not in batch, i.e. a sample not used for the generation 
                finisher=0# Ensure the end with this flag
                while True:
                    
                    wrong_key = random.choice(list(new_batch.keys()))  # pick 1 random key
                    if wrong_key not in batch :  # check against batch keys
                        wrong_example_feats = new_batch[wrong_key]
                        final_string=highlight_random_word (wrong_example_feats["textual_features"]["final_output"])
                        examples_list.append(final_string)
                        example_labels.append('0')
                        break 
                    finisher+=1
                    if finisher==5:
                        break
                
            for _,feats in batch.items():
            
                if len(examples_list)==5:
                    break
                final_string=feats["textual_features"]["final_output"]
                examples_list.append(feats["textual_features"]["final_output"])
                example_labels.append('1')
                
                        
            explanation=concept_textual_dictionary[neuron_number]
            prompt_text=create_prompt_llama_fuzzing(examples_list=examples_list,explanation=explanation)
            
            
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

@gin.configurable
def fuz_score_coco( embedding_path:Path,device:torch.cuda.device='cuda:0'):
    """Compute the fuzzing score on the coco dataset

    Args:
        embedding_path (Path):Path to the embedding folder.
        device (torch.cuda.device, optional): Device used. Defaults to 'cuda:0'.
    Return
        none: Save the scores on a json file in embedding_path folder.
    """    

    #Load the model
    tokenizer = AutoTokenizer.from_pretrained("hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4")
    model = AutoModelForCausalLM.from_pretrained("hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4")
    model.to(device)
    model.eval()
     
    # Dictionary with hypotheses
    concept_textual_dictionary=json.load(open(embedding_path + 'textual_hypothesis_dictionary.json','r'))
    
    #Dictionary with texts used to extract the concepts
    neuron_dictionary_path=embedding_path + 'average_activation_textual_dictionary.json'
    neuron_activation_dictionary=json.load(open(neuron_dictionary_path,'r'))
    neuron_ids=concept_textual_dictionary.keys()
    
    # Only extract the needed neuron 
    used_neurons_dictionary={k: neuron_activation_dictionary[k] for k in neuron_ids if "No textual " not in concept_textual_dictionary[k] and k in neuron_activation_dictionary}
    #Output path
    output_path=embedding_path+'autointerpretability_fuzzing_scores_coco_dictionary.json'


  

    dictionary_labels={}
    for neuron_number,batch in tqdm(used_neurons_dictionary.items(),desc='Evaluate Textual Hypotheses Fuzzing' ,total=len(used_neurons_dictionary),
                                leave=False):
        examples_list=[]
        # Number of wrong samples, random between 0 and 5, each wrong sample is from a different neuron
        other_neurons = neuron_ids_excluding_target(list(used_neurons_dictionary.keys()),neuron_number)
        
        example_labels=[]
        for neuron in other_neurons:
            #Samples between the neuron examples_list
            new_batch = used_neurons_dictionary[neuron]
            
            # Keep sampling until we find a key not in batch, i.e. a sample not used for the generation 
            finisher=0# Ensure the end with this flag
            while True:
                
                wrong_key = random.choice(list(new_batch.keys()))  # pick 1 random key
                if wrong_key not in batch :  # check against batch keys
                    wrong_example_feats = new_batch[wrong_key]
                    final_string=highlight_random_word (wrong_example_feats["textual_features"]["final_output"])
                    examples_list.append(final_string)
                    example_labels.append('0')
                    break 
                finisher+=1
                if finisher==5:
                    break
            
        for _,feats in batch.items():
        
            if len(examples_list)==5:
                break
            final_string=feats["textual_features"]["final_output"]
            examples_list.append(feats["textual_features"]["final_output"])
            example_labels.append('1')
            
        
            
        explanation=concept_textual_dictionary[neuron_number]
        prompt_text=create_prompt_llama_fuzzing(examples_list=examples_list,explanation=explanation)
        
        
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
