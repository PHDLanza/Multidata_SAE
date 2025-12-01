import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["HF_HUB_CACHE"]="/data/lanza/hub"
os.environ["SAE_DISABLE_TRITON"] = "0"
os.environ["TOKENIZERS_PARALLELISM"]="false"
os.environ["PATH"] += os.pathsep + "/sbin/"
from io import BytesIO
import base64
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
from utils.api  import create_dictionary_neurons
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from PIL import Image
from utils.utils_image import reconstruct_image, create_image_patches

import gin
from transformers import LlavaNextProcessor 
from datasets import load_dataset
from llava.model.builder import load_pretrained_model
def take(id_loader, iterable):
    "Return one fifth of items from the iterable based on id_loader (0-4)"
    # Convert iterable to list to get length
    items = list(iterable)
    total_len = len(items)
    set_size = total_len // 5
    
    # Calculate start and end indices for the requested set
    start_idx = id_loader * set_size
    end_idx = start_idx + set_size if id_loader < 4 else total_len
    
    return items[start_idx:end_idx]

    
def prompt_routine(list_texts:List,modality:str='visual'):
    """
    Prepares a prompt for the Llava-Next routine to extract shared concepts from a list of texts and images.

    Args:
        list_texts (List): List of text entries to analyze for concept extraction.

    Returns:
        List: A list containing the formatted prompt string for the model.
    """ 
    if modality=='visual':
    
        GUIDELINES = """ 
            [REQUIREMENTS]
                Focus only on the highlighted region in each image. If no region is highlighted or if the highlighted region is minimal (e.g., a few bright spots), ignore the image.
            
                Identify common visual patterns, objects, or concepts in the activated regions. For example, note if highlighted areas show consistent structures, such as mesh patterns or similar objects.\
            
            [GUIDELINES]

            1.Consider Text Context: While maintaining primary focus on the highlighted regions in images, you may marginally consider the associated text (questions and answers) to support or refine your visual observations. 
            However, the final concept should be predominantly based on visual patterns.
            
            2.Concise Description Only: Provide a short, direct description of the common features within the highlighted regions. Avoid any interpretive language—simply state what you see, such as “mesh-like structures” or “actions related to joy or happiness”
            
            3. Describe Only the Highlighted Regions: Generate captions solely based on the highlighted regions. If no meaningful pattern is visible, or if only a few scattered spots are highlighted,
                output: \"Concept:  `No visual concept`\"
                
          
        """
    elif modality=='textual':
        GUIDELINES=""" 
            [REQUIREMENTS]
            Focus only on the text content provided with each example. If the text is missing, irrelevant, or extremely minimal (e.g., a few unrelated words), ignore the text.

            Identify textual pattern common themes, objects, or concepts mentioned across the text snippets. Pay special attention to any word in each text between parentheses , this word should be treated as the most important cue for concept identification.

        [GUIDELINES]

         


            1.Concise Description Only: Provide a short, direct description of the common concept emerging from the texts. Avoid speculation or abstract interpretation—simply state what is explicitly or implicitly repeated, especially in relation to the words in parentheses (e.g., "Description of vehicle," "Cooking actions," "Chinese characters").
            
            2.If no clear concept emerges from the texts (e.g., if they are too diverse or vague),  write:  \"Concept:  `No textual concept`\"
            
            
  
            """


    GUIDELINES += """\n\n"""
    if modality=='visual':

        for text  in list_texts:
            GUIDELINES += " image {}: question and answer {}.\n\n".format(DEFAULT_IMAGE_TOKEN,text)
    else:
         for text  in list_texts:
            GUIDELINES += " Text {}.\n\n".format(text)

    return [GUIDELINES]
     
@gin.configurable
def generate_visual_hypotheses_coco(dataset_path:Path,embedding_path:Path, dictionary_neurons_path:Path,labels_path:Path,id_loader:int=0)->None:
    """
    Generate possibles hypotheses of concept for each neuron saved in the df_neuron variables

    Args:
        dataset_path (Path): path to folder where is saved the dataset  
        dictionary_neurons_path (Path): path to folder where the dictionary with the outputs of the backbone is saved 
        embedding_path (Path): path to folder where the latent activations of the SAE are saved 
        labels_path (Path): path to folder were the labels are collected
        device (torch.device, optional): name of the device. Defaults to 'cpu'.
    Result:
        For each neuron in df_neurons, the function engender a txt file with the LlaVa prompt and besides and a summary.json 
        with a short version of each of all of them.
       


    """    
    modality='visual'
    if not os.path.exists(dictionary_neurons_path):
        neuron_list=range(5000)
        dictionary_neurons_path=create_dictionary_neurons(embedding_path,neuron_list,modality=modality)
        # dictionary_neurons_path=create_dictionary_neurons_llava_next(embedding_path,list_neurons)
        
       
    


    # Load model
    pretrained = "lmms-lab/llava-onevision-qwen2-72b-ov"
    model_name = "llava_qwen"
   
    llava_model_args = {
            "multimodal": True,
        }
    overwrite_config = {}
    overwrite_config["image_aspect_ratio"] = "pad"
    llava_model_args["overwrite_config"] = overwrite_config
        
    # tokenizer, model, image_processor, _ = load_pretrained_model(pretrained, None, model_name, device_map='auto', **llava_model_args)
    tokenizer, model, image_processor, _ = load_pretrained_model(pretrained, None, model_name, device_map='auto', **llava_model_args,load_4bit=True)
    
    # tokenizer, model, image_processor, _ = load_pretrained_model(pretrained, None, model_name, device_map='auto', **llava_model_args)

    # Used to retrive the question and the image_id

    model.generation_config.temperature=None
    model.generation_config.top_p=None
    model.eval()
    
    

    average_activation_dictionary=json.load(open(labels_path, 'r'))

    df_label=pd.DataFrame.from_dict(average_activation_dictionary)
    
    dictionary_neurons=json.load(open(dictionary_neurons_path,'r'))
    dictionary_hypo={str(i): [] for i in range(5000)}

    path_hypo =Path(embedding_path+'dictionary_hypo_'+str(id_loader)+'_'+modality+'_.json')

    
    conv_template = "qwen_1_5"
    system="You are a meticulous AI researcher conducting an important investigation into a certain neuron in a vision language model. Your task is to analyze the neuron and provide an explanation that thoroughly encapsulates its behavior."
    conv = copy.deepcopy(conv_templates[conv_template])

    conv.system=system
    
    actual_conversation=conv
    portion_dictionary_neurons=take(id_loader, dictionary_neurons.items())
    
    for neuron_number,batch in tqdm(portion_dictionary_neurons,desc=' Generate the Image Hypotheses  ' ,total=len(portion_dictionary_neurons),leave=False):
    
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

              
                dictionary_hypo[neuron_number]=text_outputs[0]
        
    with open(path_hypo, 'w') as json_file:
        json.dump(dictionary_hypo, json_file, indent=4)

@gin.configurable
def generate_textual_hypotheses_coco(dataset_path:Path,embedding_path:Path, dictionary_neurons_path:Path,labels_path:Path,id_loader:int=0)->None:
    """
    Generate possibles hypotheses of concept for each neuron saved in the df_neuron variables

    Args:
        dataset_path (Path): path to folder where is saved the dataset  
        dictionary_neurons_path (Path): path to folder where the dictionary with the outputs of the backbone is saved 
        embedding_path (Path): path to folder where the latent activations of the SAE are saved 
        labels_path (Path): path to folder were the labels are collected
        device (torch.device, optional): name of the device. Defaults to 'cpu'.
        
    Result:
        For each neuron in df_neurons, the function engender a txt file with the LlaVa prompt and besides and a summary.json 
        with a short version of each of all of them.
       


    """    
    modality='textual'
    if not os.path.exists(dictionary_neurons_path):
        neuron_list=range(5000)
        dictionary_neurons_path=create_dictionary_neurons(embedding_path,neuron_list,modality=modality)
      
        
       
    


    # Load model
    pretrained = "lmms-lab/llava-onevision-qwen2-72b-ov"
    model_name = "llava_qwen"
   
    llava_model_args = {
            "multimodal": True,
        }
    overwrite_config = {}
    overwrite_config["image_aspect_ratio"] = "pad"
  
    llava_model_args["overwrite_config"] = overwrite_config
    tokenizer, model, image_processor, _ = load_pretrained_model(pretrained, None, model_name, device_map='auto', **llava_model_args,load_4bit=True)
    
    # tokenizer, model, image_processor, _ = load_pretrained_model(pretrained, None, model_name, device_map='auto', **llava_model_args)

    # Used to retrive the question and the image_id

    model.generation_config.temperature=None
    model.generation_config.top_p=None
    model.eval()
    
    

    average_activation_dictionary=json.load(open(labels_path, 'r'))

    df_label=pd.DataFrame.from_dict(average_activation_dictionary)
    
    dictionary_neurons=json.load(open(dictionary_neurons_path,'r'))
    dictionary_hypo={str(i): [] for i in range(5000)}

    path_hypo =Path(embedding_path+'dictionary_hypo_'+str(id_loader)+'_'+modality+'_.json')
    
    conv_template = "qwen_1_5"
    system="You are a meticulous AI researcher conducting an important investigation into a certain neuron in a vision language model. Your task is to analyze the neuron and provide an explanation that thoroughly encapsulates its behavior."
    conv = copy.deepcopy(conv_templates[conv_template])

    conv.system=system
    portion_dictionary_neurons=take(id_loader, dictionary_neurons.items())
    
    actual_conversation=conv
    
  
    for neuron_number,batch in tqdm(portion_dictionary_neurons,desc=' Generate the Text Hypotheses ' ,total=len(portion_dictionary_neurons),leave=False):
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
    
        
                dictionary_hypo[neuron_number]=text_outputs[0]
                
        else:
            dictionary_hypo[neuron_number]='No textual concept'
            
 
    with open(path_hypo, 'w') as json_file:
        json.dump(dictionary_hypo, json_file, indent=4)

@gin.configurable
def generate_visual_hypotheses_llava(dataset_path:Path,embedding_path:Path, dictionary_neurons_path:Path,id_loader:int=0)->None:
    """
    Generate possibles hypotheses of concept for each neuron saved in the df_neuron variables

    Args:
        dataset_path (Path): path to folder where is saved the dataset  
        dictionary_neurons_path (Path): path to folder where the dictionary with the outputs of the backbone is saved 
        embedding_path (Path): path to folder where the latent activations of the SAE are saved 
        labels_path (Path): path to folder were the labels are collected
        device (torch.device, optional): name of the device. Defaults to 'cpu'.
    Result:
        For each neuron in df_neurons, the function engender a txt file with the LlaVa prompt and besides and a summary.json 
        with a short version of each of all of them.
       


    """    
    modality='visual'
    if not os.path.exists(dictionary_neurons_path):
        neuron_list=range(5000)
        dictionary_neurons_path=create_dictionary_neurons(embedding_path,neuron_list,modality=modality)
        # dictionary_neurons_path=create_dictionary_neurons_llava_next(embedding_path,list_neurons)
        

    
    data = load_dataset("lmms-lab/LLaVA-NeXT-Data", split="train[:15%]", cache_dir=dataset_path, num_proc=10)
    path_hypo =Path(embedding_path+'dictionary_hypo_'+str(id_loader)+'_'+modality+'_.json')
   

    # Load model
    pretrained = "lmms-lab/llava-onevision-qwen2-72b-ov"
    model_name = "llava_qwen"
   
    llava_model_args = {
            "multimodal": True,
        }
    overwrite_config = {}
    overwrite_config["image_aspect_ratio"] = "pad"
    llava_model_args["overwrite_config"] = overwrite_config
    # tokenizer, model, image_processor, _ = load_pretrained_model(pretrained, None, model_name, device_map='auto', **llava_model_args,load_4bit=True)
    
    tokenizer, model, image_processor, _ = load_pretrained_model(pretrained, None, model_name, device_map='auto', **llava_model_args)

    # Used to retrive the question and the image_id

    model.generation_config.temperature=None
    model.generation_config.top_p=None
    model.eval()
    
    dictionary_neurons=json.load(open(dictionary_neurons_path,'r'))
    dictionary_hypo={str(i): [] for i in range(5000)}

    conv_template = "qwen_1_5"
    system="You are a meticulous AI researcher conducting an important investigation into a certain neuron in a vision language model. Your task is to analyze the neuron and provide an explanation that thoroughly encapsulates its behavior."
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.system=system
    

    portion_dictionary_neurons=take(id_loader, dictionary_neurons.items())
  
    needed_ids = set()
    for _, batch in portion_dictionary_neurons:  # limit to 1000 for progress bar
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
    
    for neuron_number, batch in tqdm(portion_dictionary_neurons, desc="Generate Visual Hypotheses", total=1000, leave=False):
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
                dictionary_hypo[neuron_number] = output
                torch.cuda.empty_cache()


        
    with open(path_hypo, 'a') as json_file:
        json.dump(dictionary_hypo, json_file, indent=4)

@gin.configurable
def generate_textual_hypotheses_llava(dataset_path:Path,embedding_path:Path, dictionary_neurons_path:Path,id_loader:int=0)->None:
    """
    Generate possibles hypotheses of concept for each neuron saved in the df_neuron variables

    Args:
        dataset_path (Path): path to folder where is saved the dataset  
        dictionary_neurons_path (Path): path to folder where the dictionary with the outputs of the backbone is saved 
        embedding_path (Path): path to folder where the latent activations of the SAE are saved 
        labels_path (Path): path to folder were the labels are collected
        device (torch.device, optional): name of the device. Defaults to 'cpu'.
        
    Result:
        For each neuron in df_neurons, the function engender a txt file with the LlaVa prompt and besides and a summary.json 
        with a short version of each of all of them.
       


    """    
    modality='textual'
    if not os.path.exists(dictionary_neurons_path):
        neuron_list=range(5000)
        dictionary_neurons_path=create_dictionary_neurons(embedding_path,neuron_list,modality=modality)
      
        
       
     # train[:15%]
    data = load_dataset("lmms-lab/LLaVA-NeXT-Data", split="train[:15%]", cache_dir=dataset_path, num_proc=10)
    
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
    # tokenizer, model, image_processor, _ = load_pretrained_model(pretrained, None, model_name, device_map='auto', **llava_model_args)

    # Used to retrive the question and the image_id

    model.generation_config.temperature=None
    model.generation_config.top_p=None
    model.eval()
    

    
    dictionary_neurons=json.load(open(dictionary_neurons_path,'r'))
    dictionary_hypo={str(i): [] for i in range(5000)}

    path_hypo =Path(embedding_path+'dictionary_hypo_'+str(id_loader)+'_'+modality+'_.json')
    
    conv_template = "qwen_1_5"
    system="You are a meticulous AI researcher conducting an important investigation into a certain neuron in a vision language model. Your task is to analyze the neuron and provide an explanation that thoroughly encapsulates its behavior."
    portion_dictionary_neurons=take(id_loader, dictionary_neurons.items())
    ##where to cut
  
    needed_ids = set()
    for _, batch in portion_dictionary_neurons:  # limit to 1000 for progress bar
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
    
    processor = LlavaNextProcessor.from_pretrained("llava-hf/llama3-llava-next-8b-hf")
    
    
    for neuron_number,batch in tqdm(portion_dictionary_neurons,desc='Generate the Textual Hypotheses' ,total=1000,leave=False):
        
        #Used to memorize which images are used to derive the hypothesis
        texts=[]
        
        

        for id_sample_str, feats in batch.items():
            id_sample = int(id_sample_str)
            entry = lookup.get(id_sample)
            
            if entry is None:
                continue
            
            input_ids = processor(feats["textual_features"]["final_output"])
            token_ids = input_ids["input_ids"][0]  # Assumes batch size = 1

            # Decode all tokens at once for efficiency
            decoded_tokens = processor.batch_decode([[tok_id] for tok_id in token_ids], skip_special_tokens=True)

            final_string = []

            for i, (token_str, neuron_ids) in enumerate(zip(decoded_tokens, feats["textual_features"]["latent_indices"])):
                if i == 0:
                    continue  # skip first token if needed

                if int(neuron_number) in neuron_ids:
                    final_string.append(f"[{token_str}]")
                else:
                    final_string.append(token_str)
            
            
            # build prompt text
            convo = entry["conversations"][0]["value"]
            text_feat = ''.join(final_string)
            
            texts.append(convo.replace("<image>", " ").replace("\n", " ")+ f" {text_feat}")

            # mask out patches
           
        
            # TypeError: color must be int or single-element tuple
       

            # Prepare the template

            questions= prompt_routine(texts,modality=modality)
                
            with torch.inference_mode():
                

                conv.append_message(conv.roles[0], questions[0])
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
                # Generate response
                
                output = model.generate(

                    input_ids,
                    images=None,
                    image_sizes=None,
                    do_sample=False,
                    
                    max_new_tokens=4096
                )
                
                output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
                conv.messages.pop()
                conv.messages.pop()
                dictionary_hypo[neuron_number] = output
                torch.cuda.empty_cache()
                

            
 
    with open(path_hypo, 'w') as json_file:
        json.dump(dictionary_hypo, json_file, indent=4)
        