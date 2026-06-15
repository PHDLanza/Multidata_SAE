import os
os.environ["SAE_DISABLE_TRITON"] = "0"
os.environ["TOKENIZERS_PARALLELISM"]="false"
import torch
import json
from sparsify.sparsify.sparse_coder import SparseCoder as SAE
from typing import  List
import pandas as pd 
from tqdm import tqdm
import numpy as np
import copy
from pathlib import Path
from llava.mm_utils import  process_images, tokenizer_image_token
from utils.utils_image import reconstruct_image, create_image_patches
from utils.api  import create_neuron_top5_dictionary
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from PIL import Image
from utils.utils_image import reconstruct_image, create_image_patches
import gin
from datasets import load_dataset
from llava.model.builder import load_pretrained_model
from utils.utils_prompt import GUIDELINES_VISUAL_GENERATION, GUIDELINES_TEXTUAL_GENERATION

def prompt_routine(list_texts:List,modality:str='visual'):
    """
    Prepares a prompt for the Llava-Next routine to extract shared concepts from a list of texts and images.

    Args:
        list_texts (List): List of text entries to analyze for concept extraction.

    Returns:
        List: A list containing the formatted prompt string for the model.
    """ 
    if modality=='visual':
    
        GUIDELINES = GUIDELINES_VISUAL_GENERATION
    elif modality=='textual':
        GUIDELINES= GUIDELINES_TEXTUAL_GENERATION


    GUIDELINES += """\n\n"""
    if modality=='visual':

        for text  in list_texts:
            GUIDELINES += " image {}: question and answer {}.\n\n".format(DEFAULT_IMAGE_TOKEN,text)
    else:
         for text  in list_texts:
            GUIDELINES += " Text {}.\n\n".format(text)

    return [GUIDELINES]
     
@gin.configurable
def generate_visual_hypotheses_coco(dataset_path:Path,embedding_path:Path,labels_path:Path)->None:
    """
    Generate possibles hypotheses of concept for each neuron saved in the df_neuron variables

    Args:
        dataset_path (Path): path to folder where is saved the dataset  
        embedding_path (Path): path to folder where the latent activations of the SAE are saved 
        labels_path (Path): path to folder were the labels are collected
    Result:
        For each neuron in df_neurons, the function engender a txt file with the LlaVa prompt and besides and a summary.json 
        with a short version of each of all of them.
       


    """    
    modality='visual'
    save_result_path =Path(embedding_path+'/'+modality+'hypothesis_dictionary.json')
    neuron_top5_visual_dictionary_path=embedding_path+'/neuron_top5_visual_dictionary.json'
    if not os.path.exists(neuron_top5_visual_dictionary_path):
        neuron_list=range(5000)
        neuron_top5_visual_dictionary_path=create_neuron_top5_dictionary(embedding_path,neuron_list,modality=modality)
        # neuron_output_dictionary_path=create_neuron_top5_dictionary_llava_next(embedding_path,list_neurons)
        
       
    


    # Load model
    pretrained = "lmms-lab/llava-onevision-qwen2-72b-ov"
    model_name = "llava_qwen"
   
    llava_model_args = {
            "multimodal": True,
        }
    overwrite_config = {}
    overwrite_config["image_aspect_ratio"] = "pad"
    llava_model_args["overwrite_config"] = overwrite_config
    
    tokenizer, model, image_processor, _ = load_pretrained_model(pretrained, None, model_name, device_map='auto', **llava_model_args)
    
    # If only one A100 or H100 is available, load the model in 4-bit
    # tokenizer, model, image_processor, _ = load_pretrained_model(pretrained, None, model_name, device_map='auto', **llava_model_args,load_4bit=True)
    


    # Used to retrive the question and the image_id

    model.generation_config.temperature=None
    model.generation_config.top_p=None
    model.eval()
    
    

    average_activation_dictionary=json.load(open(labels_path, 'r'))

    df_label=pd.DataFrame.from_dict(average_activation_dictionary)
    
    neuron_output_dictionary=json.load(open(neuron_top5_visual_dictionary_path,'r'))
    hypothesis_dictionary={str(i): [] for i in range(5000)}

   

    
    conv_template = "qwen_1_5"
    system="You are a meticulous AI researcher conducting an important investigation into a certain neuron in a vision language model. Your task is to analyze the neuron and provide an explanation that thoroughly encapsulates its behavior."
    conv = copy.deepcopy(conv_templates[conv_template])

    conv.system=system
    
    actual_conversation=conv
    
    
    for neuron_number,batch in tqdm(neuron_output_dictionary,desc=' Generate the Visual Hypotheses' ,total=len(neuron_output_dictionary),leave=False):
    
        ids_list=batch.keys()   
        #Used to memorize which images are used to derive the hypothesis
        masked_image=[]
    
        texts=[]
        
        for id in ids_list:
            # Extract and clean text
            textual_features = batch[id]["textual_features"]
            tmp_text = textual_features['final_output'].replace('assistant', '').replace('\n', '')
            
            texts.append(df_label[id]['question'] + tmp_text)

            # Get image path
            img_name = df_label[id]['image_name']
            is_train = 'train' in img_name
            folder_tmp = dataset_path + ('train2014/' if is_train else 'val2014/')
            img_path = folder_tmp + img_name
            
            # Process image patches
            patches = create_image_patches(img_path)
            zeros_vector = np.zeros(576)
            # images.append(img_path)
            
            # Create masked image based on neuron activation
            latent_indices = batch[id]["visual_features"]['latent_indices']
            for patch_idx, indices_array in enumerate(latent_indices):
                if int(neuron_number) in indices_array:
                    zeros_vector[patch_idx] = 1
            
            reconstructed_array = reconstruct_image(patches, zeros_vector)
            masked_image.append(Image.fromarray(reconstructed_array))
        
        # TypeError: color must be int or single-element tuple
       
        if masked_image:

            image_tensors = process_images(masked_image, image_processor, model.config)
            image_tensors = [_image.to(dtype=torch.float16, device=model.device) for _image in image_tensors]
            image_sizes = [image.size for image in masked_image]

            # Prepare the template

            questions= prompt_routine(texts,modality)
            # questions= questions_routine_vqa(texts,modality)

            
            
            with torch.inference_mode():
                
                actual_conversation.append_message(actual_conversation.roles[0], questions[0])
                actual_conversation.append_message(actual_conversation.roles[1], None)
                prompt_question = actual_conversation.get_prompt()
                input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
                # Generate response
                
                cont = model.generate(

                    input_ids,
                    images=image_tensors,
                    image_sizes=image_sizes,
                    do_sample=False,
                    
                    max_new_tokens=4096
                    
                )

                text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
                del input_ids,cont, prompt_question
                # print(text_outputs[0])
                torch.cuda.empty_cache()
                actual_conversation.messages.pop()
                actual_conversation.messages.pop()

              
                hypothesis_dictionary[neuron_number]=text_outputs[0]
        
    with open(save_result_path, 'w') as json_file:
        json.dump(hypothesis_dictionary, json_file, indent=4)

@gin.configurable
def generate_textual_hypotheses_coco(dataset_path:Path,embedding_path:Path,labels_path:Path)->None:
    """
    Generate possibles hypotheses of concept for each neuron saved in the df_neuron variables

    Args:
        dataset_path (Path): path to folder where is saved the dataset  
        embedding_path (Path): path to folder where the latent activations of the SAE are saved 
        labels_path (Path): path to folder were the labels are collected
        
    Result:
        For each neuron in df_neurons, the function engender a txt file with the LlaVa prompt and besides and a summary.json 
        with a short version of each of all of them.
       


    """    
    modality='textual'
    save_result_path =Path(embedding_path+'dictionary_hypo_'+modality+'_.json')
    neuron_top5_textual_dictionary_path=embedding_path+'/neuron_top5_textual_dictionary.json'
    if not os.path.exists(neuron_top5_textual_dictionary_path):
        neuron_list=range(5000)
        neuron_top5_textual_dictionary_path=create_neuron_top5_dictionary(embedding_path,neuron_list,modality=modality)
      
        
       
    


    # Load model
    pretrained = "lmms-lab/llava-onevision-qwen2-72b-ov"
    model_name = "llava_qwen"
   
    llava_model_args = {
            "multimodal": True,
        }
    overwrite_config = {}
    overwrite_config["image_aspect_ratio"] = "pad"
  
    llava_model_args["overwrite_config"] = overwrite_config
    
    # If only one A100 or H100 is available, load the model in 4-bit
    # tokenizer, model, image_processor, _ = load_pretrained_model(pretrained, None, model_name, device_map='auto', **llava_model_args,load_4bit=True)
    
    tokenizer, model, image_processor, _ = load_pretrained_model(pretrained, None, model_name, device_map='auto', **llava_model_args)

    # Used to retrive the question and the image_id

    model.generation_config.temperature=None
    model.generation_config.top_p=None
    model.eval()
    
    

    average_activation_dictionary=json.load(open(labels_path, 'r'))

    df_label=pd.DataFrame.from_dict(average_activation_dictionary)
    
    neuron_output_dictionary=json.load(open(neuron_top5_textual_dictionary_path,'r'))
    hypothesis_dictionary={str(i): [] for i in range(5000)}

    conv_template = "qwen_1_5"
    system="You are a meticulous AI researcher conducting an important investigation into a certain neuron in a vision language model. Your task is to analyze the neuron and provide an explanation that thoroughly encapsulates its behavior."
    conv = copy.deepcopy(conv_templates[conv_template])

    conv.system=system
    

    
    actual_conversation=conv
    
  
    for neuron_number,batch in tqdm(neuron_output_dictionary,desc=' Generate the Textual Hypotheses' ,total=len(neuron_output_dictionary),leave=False):
        ids_list=batch.keys()   
        #Used to memorize which images are used to derive the hypothesis
        
        texts=[]
        images=[]
       
        for id in ids_list:
            tmp_text=batch[id]["textual_features"]['final_output'].replace('assistant','')
            tmp_text=tmp_text.replace('\n','')
            texts.append(df_label[id]['question'] + tmp_text)

            img_name=df_label[id]['image_name']
            folder_tmp = dataset_path+'train2014/' if 'train' in img_name else dataset_path+'val2014/'
            images.append(Image.open(folder_tmp+img_name).convert("RGB"))
            
        
    # TypeError: color must be int or single-element tuple

        if images:
        
            image_tensors = process_images(images, image_processor, model.config)
            
            image_tensors = [_image.to(dtype=torch.float16, device=model.device) for _image in image_tensors]
            image_sizes = [image.size for image in images]

                    # Prepare the template

            questions= prompt_routine(texts,modality='textual')
            
            with torch.inference_mode():
                
                actual_conversation.append_message(actual_conversation.roles[0], questions[0])
                actual_conversation.append_message(actual_conversation.roles[1], None)
                prompt_question = actual_conversation.get_prompt()
                input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
                # Generate response
                
                cont = model.generate(

                    input_ids,
                    images=image_tensors,
                    image_sizes=image_sizes,
                    do_sample=False,
                    
                    max_new_tokens=4096
                    
                )

                text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
                del input_ids,cont, prompt_question
                # print(text_outputs[0])
                torch.cuda.empty_cache()
                actual_conversation.messages.pop()
                actual_conversation.messages.pop()
    
        
                hypothesis_dictionary[neuron_number]=text_outputs[0]
                
        else:
            hypothesis_dictionary[neuron_number]='No textual concept'
            
 
    with open(save_result_path, 'w') as json_file:
        json.dump(hypothesis_dictionary, json_file, indent=4)

@gin.configurable
def generate_visual_hypotheses_llava(dataset_path:Path,embedding_path:Path)->None:
    """
    Generate possibles hypotheses of concept for each neuron saved in the df_neuron variables

    Args:
        dataset_path (Path): path to folder where is saved the dataset  
        embedding_path (Path): path to folder where the latent activations of the SAE are saved 
        
    Result:
        For each neuron in df_neurons, the function engender a txt file with the LlaVa prompt and besides and a summary.json 
        with a short version of each of all of them.
       


    """    
    modality='visual'
    save_result_path =Path(embedding_path+'/'+modality+'hypothesis_dictionary.json')
    neuron_top5_visual_dictionary_path=embedding_path+'/neuron_top5_visual_dictionary.json'
    if not os.path.exists(neuron_top5_visual_dictionary_path):
        neuron_list=range(5000)
        print('create visual dictionary')
        
        neuron_top5_visual_dictionary_path=create_neuron_top5_dictionary(embedding_path,neuron_list,modality=modality)
        

    
    data = load_dataset("lmms-lab/LLaVA-NeXT-Data", split="train[:15%]", cache_dir=dataset_path, num_proc=10)
   


    # Load model
    pretrained = "lmms-lab/llava-onevision-qwen2-72b-ov"
    model_name = "llava_qwen"
   
    llava_model_args = {
            "multimodal": True,
            "attn_implementation": "sdpa"
            
        }
    overwrite_config = {}
    overwrite_config["image_aspect_ratio"] = "pad"
    llava_model_args["overwrite_config"] = overwrite_config
    # If only one A100 or H100 is available, load the model in 4-bit
    # tokenizer, model, image_processor, _ = load_pretrained_model(pretrained, None, model_name, device_map='auto', **llava_model_args,load_4bit=True)
    
    tokenizer, model, image_processor, _ = load_pretrained_model(pretrained, None, model_name, device_map='auto', **llava_model_args)

    # Used to retrive the question and the image_id

    model.generation_config.temperature=None
    model.generation_config.top_p=None
    model.eval()
    
    neuron_output_dictionary=json.load(open(neuron_top5_visual_dictionary_path,'r'))
    hypothesis_dictionary={str(i): [] for i in range(5000)}

    conv_template = "qwen_1_5"
    system="""You are a meticulous AI researcher conducting an important investigation into a certain neuron in a vision language model. Your task is to analyze the neuron and provide an explanation that thoroughly encapsulates its behavior."""
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.system=system
    

  
    needed_ids = set()
    for _, batch in neuron_output_dictionary:  
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
    
    for neuron_number, batch in tqdm(neuron_output_dictionary, desc="Generate Visual Hypotheses", total=len(neuron_output_dictionary), leave=False):
        texts, masked_images = [], []

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
            
            
            # images.append(img_path)
            
            # Create masked image based on neuron activation
            
            zeros = np.zeros(len(patches), dtype=np.uint8)
            for patch_idx, inds in enumerate(feats["visual_features"]["latent_indices"]):
                if int(neuron_number) in inds:
                    zeros[patch_idx] = 1
      
            
            reconstructed_array = reconstruct_image(patches, zeros)
            masked_images.append(Image.fromarray(reconstructed_array))
        
       
        
        # TypeError: color must be int or single-element tuple
       
        if masked_images:

            # prepare tensors
            image_tensors = process_images(masked_images, image_processor, model.config)
            image_tensors = [_image.to(dtype=torch.float16, device=model.device) for _image in image_tensors]
            image_sizes = [image.size for image in masked_images]

            # Prepare the template
          
            questions= prompt_routine(texts,modality)

            with torch.inference_mode():
                conv.append_message(conv.roles[0], questions[0])
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
                
                # Generate response
                
                output = model.generate(

                    input_ids,
                    images=image_tensors,
                    image_sizes=image_sizes,
                    do_sample=False,
                    
                    max_new_tokens=4096
                )
                output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
                # print(text_outputs[0])
                conv.messages.pop()
                conv.messages.pop()
                hypothesis_dictionary[neuron_number] = output
                torch.cuda.empty_cache()


        
    with open(save_result_path, 'a') as json_file:
        json.dump(hypothesis_dictionary, json_file, indent=4)

@gin.configurable
def generate_textual_hypotheses_llava(dataset_path:Path,embedding_path:Path)->None:
    """
    Generate possibles hypotheses of concept for each neuron saved in the df_neuron variables

    Args:
        dataset_path (Path): path to folder where is saved the dataset  
        embedding_path (Path): path to folder where the latent activations of the SAE are saved 

        
    Result:
        For each neuron in df_neurons, the function engender a txt file with the LlaVa prompt and besides and a summary.json 
        with a short version of each of all of them.
       


    """    
    modality='textual'
    save_result_path =Path(embedding_path+'/'+modality+'hypothesis_dictionary.json')
    neuron_top5_textual_dictionary_path=embedding_path+'/neuron_top5_textual_dictionary.json'
    if not os.path.exists(neuron_top5_textual_dictionary_path):
        neuron_list=range(5000)
        print('create textual dictionary')
        neuron_top5_textual_dictionary_path=create_neuron_top5_dictionary(embedding_path,neuron_list,modality=modality)
      
        
     # train[:15%]
    data = load_dataset("lmms-lab/LLaVA-NeXT-Data", split="train[:15%]", cache_dir=dataset_path, num_proc=10)
    
    # Load model
    pretrained = "lmms-lab/llava-onevision-qwen2-72b-ov"
    model_name = "llava_qwen"
   
    llava_model_args = {
            "multimodal": True,
            "attn_implementation": "sdpa",
        }
    overwrite_config = {}
    overwrite_config["image_aspect_ratio"] = "pad"
  
    llava_model_args["overwrite_config"] = overwrite_config
    
    
    # If only one A100 or H100 is available, load the model in 4-bit
    # tokenizer, model, image_processor, _ = load_pretrained_model(pretrained, None, model_name, device_map='auto', **llava_model_args,load_4bit=True)
    tokenizer, model, image_processor, _ = load_pretrained_model(pretrained, None, model_name, device_map='auto', **llava_model_args)
    # Used to retrive the question and the image_id

    model.generation_config.temperature=None
    model.generation_config.top_p=None
    model.eval()
    

    
    neuron_output_dictionary=json.load(open(neuron_top5_textual_dictionary_path,'r'))
    hypothesis_dictionary={str(i): [] for i in range(5000)}

    
    
    conv_template = "qwen_1_5"
    system="You are a meticulous AI researcher conducting an important investigation into a certain neuron in a vision language model. Your task is to analyze the neuron and provide an explanation that thoroughly encapsulates its behavior."
    
    ##where to cut
  
    needed_ids = set()
    for _, batch in  neuron_output_dictionary:  
        needed_ids.update(map(int, batch.keys()))
    lookup = {}
    for example in tqdm(data, desc="Building lookup", leave=False):
        id_sample = int(example["id"])
        if id_sample in needed_ids:
            lookup[id_sample] = {
                "conversations": example["conversations"],
                # "image": example["image"].convert('RGB')
            }
        if len(lookup) >= len(needed_ids):
            break
    conv = copy.deepcopy(conv_templates[conv_template])

    conv.system=system


    
    for neuron_number,batch in tqdm( neuron_output_dictionary,desc='Generate the Textual Hypotheses' ,total=len(neuron_output_dictionary),leave=False):
        images=[]
        texts=[]
        for img_id_str, feats in batch.items():
            img_id = int(img_id_str)
            entry = lookup.get(img_id)
            
            if entry is None:
                continue

            # build prompt text
            convo = entry["conversations"][0]["value"]
            text_feat = feats["text_features"]["final_output"]
            texts.append(convo.replace("<image>", " ").replace("\n", " ")+ f" [ {text_feat} ]")

            # mask out patches
            image = entry["image"]
           
            
            images.append(image)
            
        
            # TypeError: color must be int or single-element tuple
            if texts:
                if images:
                
                    image_tensors = process_images(images, image_processor, model.config)
                    
                    image_tensors = [_image.to(dtype=torch.float16, device=model.device) for _image in image_tensors]
                    image_sizes = [image.size for image in images]

                    # Prepare the template
            

            # Prepare the template
          

            questions= prompt_routine(texts,modality=modality)
                
            with torch.inference_mode():
                
                image_tensors = process_images(images, image_processor, model.config)
                image_tensors = [_image.to(dtype=torch.float16, device=model.device) for _image in image_tensors]
               
                conv.append_message(conv.roles[0], questions[0])
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
                # Generate response
                
                output = model.generate(

                    input_ids,
                    images=image_tensors,
                    image_sizes=image_sizes,
                    do_sample=False,
                    
                    max_new_tokens=4096
                )
                
                output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
                conv.messages.pop()
                conv.messages.pop()
                hypothesis_dictionary[neuron_number] = output
                torch.cuda.empty_cache()
                

            
 
    with open(save_result_path, 'w') as json_file:
        json.dump(hypothesis_dictionary, json_file, indent=4)
        