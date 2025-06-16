import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["HF_HUB_CACHE"]="/data/lanza/hub"
os.environ["SAE_DISABLE_TRITON"] = "0"
os.environ["TOKENIZERS_PARALLELISM"]="false"

os.environ["PATH"] += os.pathsep + "/sbin/"
import torch
import json
from sparsify.sparsify.sparse_coder import SparseCoder as SAE
from typing import  List
import pandas as pd 
from tqdm import tqdm
import numpy as np
import copy
from pathlib import Path
from llava.model.builder import load_pretrained_model
from llava.mm_utils import  process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from PIL import Image
from utils.utils_image import reconstruct_image, create_image_patches
from utils.api  import create_dictionary_neurons
import gin
import glob
from datasets import load_dataset

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

def check_neuron_in_text(id_neuron:int, list_indices:List):
    for patch in list_indices:
        if id_neuron in patch:
            return True
    return False

def questions_routine_vqa(list_texts:List,modality:str='visual'):
    """
    Prepares a prompt for the VQA routine to extract shared concepts from a list of texts and images.

    Args:
        list_texts (List): List of text entries to analyze for concept extraction.

    Returns:
        List: A list containing the formatted prompt string for the model.
    """
    if modality=='visual':
        GUIDELINES = """ 
            [REQUIREMENTS]

            1. Focus only on the highlighted region in each image. If no region is highlighted or if the highlighted region is minimal (e.g., a few bright spots), ignore the image.
            2. Identify common visual patterns, objects, or concepts in the activated regions. For example, note if highlighted areas show consistent structures, such as mesh patterns or similar objects.
            
            [GUIDELINES]
            
            1.You will receive a series of images and correlated texts, and you have to identify the shared concept between them.he images will be masked, so you will have to describe only on the visible portion of image to generate a concept.
            These are samples taken from a Visual Question Answering dataset, so for each image there is a question and an answer.
            
            
            2. Concise Description Only: Provide a short, direct description of the common features within the highlighted regions. Avoid any interpretive language—simply state what you see, such as “mesh-like structures” or “actions related to joy or happiness”. 
            Concepts can be only visual concepts so related to the image, the text can be only used to guide the generation, such as if you see a series of images regarding a specific race of dog look also if the all texts, or part of them, mention the race dog.

            
            3. If no clear concept emerges from the images to any visual concept, for example if the pixels are too far sparse that cannot form any understandable figure, write: No visual concept 

            [OUTPUT EXAMPLES]
            - Concept: "A tennis racket"   
        
            - Concept: "No visual concept"   
            
            
            
            Remember,Write always only one Concept for the entire set of inputs
        """
    
    else:
        GUIDELINES=""" 
            [REQUIREMENTS]

                Focus only on the text content provided with each example. If the text is missing, irrelevant, or extremely minimal (e.g., a few unrelated words), ignore that example.

                Identify common themes, objects, or concepts mentioned across the text snippets. Pay special attention to any highlighted word in each text—this word should be treated as the most important cue for concept identification.

            [GUIDELINES]

                1.You will receive a series of text snippets, sometimes accompanied by images. Only use the text, and in particular the word between parentheses, to identify the shared concept. Images should not be considered in your analysis.
                These examples are derived from a Visual Question Answering dataset, so each text is in the form of a question or an answer.

                2.Concise Description Only: Provide a short, direct description of the common concept emerging from the texts. Avoid speculation or abstract interpretation—simply state what is explicitly or implicitly repeated, especially in relation to the highlighted words (e.g., “vehicles,” “cooking actions,” “types of animals”).
                Use the image only for reference if absolutely necessary; the main analysis must be text-driven, with words in parentheses as priority.

                3.If no clear concept emerges from the texts (e.g., if they are too diverse or vague), write: No textual concept

            [OUTPUT EXAMPLES]

                Concept: "A tennis match"

                Concept: "Descriptions of birds"

                Concept: "No textual concept"
                
            Remember,Write always only one Concept for the entire set of inputs
            """

    GUIDELINES += """\n\n"""
    
    for text  in list_texts:
        GUIDELINES += " image {}: question and answer {}.\n\n".format(DEFAULT_IMAGE_TOKEN,text)

    return [GUIDELINES]

def questions_routine_llava(list_texts:List,modality:str='visual'):
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

            Identify common themes, objects, or concepts mentioned across the text snippets. Pay special attention to any highlighted word in each text—this word should be treated as the most important cue for concept identification.

        [GUIDELINES]

            1.Consider Visual Context: While maintaining primary focus on the text, with words in parentheses as priority,  you may marginally consider the associated image to support or refine your textual observations. 
                However, the final concept should be predominantly based on textual patterns.


            2.Concise Description Only: Provide a short, direct description of the common concept emerging from the texts. Avoid speculation or abstract interpretation—simply state what is explicitly or implicitly repeated, especially in relation to the highlighted words (e.g., "vehicles," "cooking actions," "types of animals").
            

            3.If no clear concept emerges from the texts (e.g., if they are too diverse or vague), write:  \"Concept:  `No textual concept`\"

  
            """


    GUIDELINES += """\n\n"""
    
    for text  in list_texts:
        GUIDELINES += " image {}: question and answer {}.\n\n".format(DEFAULT_IMAGE_TOKEN,text)

    return [GUIDELINES]
     
@gin.configurable
def generate_hypotheses_image(path_dataset:Path,path_embedding:Path, path_dictionary_neurons:Path,path_labels:Path,id_loader:int=0,device:torch.device='cuda:0')->None:
    """
    Generate possibles hypotheses of concept for each neuron saved in the df_neuron variables

    Args:
        path_dataset (Path): path to folder where is saved the dataset  
        path_dictionary_neurons (Path): path to folder where the dictionary with the outputs of the backbone is saved 
        path_embedding (Path): path to folder where the latent activations of the SAE are saved 
        path_labels (Path): path to folder were the labels are collected
        device (torch.device, optional): name of the device. Defaults to 'cpu'.
    Result:
        For each neuron in df_neurons, the function engender a txt file with the LlaVa prompt and besides and a summary.json 
        with a short version of each of all of them.
       


    """    
    modality='visual'
    if not os.path.exists(path_dictionary_neurons):
        list_neurons=range(5000)
        path_dictionary_neurons=create_dictionary_neurons(path_embedding,list_neurons,modality=modality)
        # path_dictionary_neurons=create_dictionary_neurons_llava_next(path_embedding,list_neurons)
        
       
    


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

    # Used to retrive the question and the image_id

    model.generation_config.temperature=None
    model.generation_config.top_p=None
    model.eval()
    
    

    average_activation_dictionary=json.load(open(path_labels, 'r'))

    df_label=pd.DataFrame.from_dict(average_activation_dictionary)
    
    dictionary_neurons=json.load(open(path_dictionary_neurons,'r'))
    dictionary_hypo={str(i): [] for i in range(5000)}

    path_hypo =Path(path_embedding+'dictionary_hypo_'+str(id_loader)+'_'+modality+'_.json')

    
    conv_template = "qwen_1_5"
    system="You are a meticulous AI researcher conducting an important investigation into a certain neuron in a vision language model. Your task is to analyze the neuron and provide an explanation that thoroughly encapsulates its behavior."
    conv = copy.deepcopy(conv_templates[conv_template])

    conv.system=system
    
    actual_conversation=conv
    
    for neuron_number,batch in tqdm(take(id_loader,dictionary_neurons.items()),desc=' Generate the Image Hypotheses  ' ,total=1000,leave=False):
    
        ids_list=batch.keys()   
        #Used to memorize which images are used to derive the hypothesis
        masked_image=[]
        images=[]
        texts=[]
        indices_test=[]
        for id in ids_list:
            # Extract and clean text
            text_features = batch[id]["text_features"]
            tmp_text = text_features['final_output'].replace('assistant', '').replace('\n', '')
            indices_test.append(check_neuron_in_text(neuron_number, text_features['latent_indices']))
            texts.append(df_label[id]['question'] + tmp_text)

            # Get image path
            img_name = df_label[id]['image_name']
            is_train = 'train' in img_name
            folder_tmp = path_dataset + ('train2014/' if is_train else 'val2014/')
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

            questions= questions_routine_vqa(texts,modality)

            
            
            with torch.no_grad():
                
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
                # prompt.append(text_outputs[0])
                # #Used to distinguish between the different answer generation
                # prompt.append('\n<End>\n\n')
                
                # actual_conversation.messages[-1][1]=text_outputs[0]
                output_bool=True
                if not all(indices_test):
                    output_bool=False
                dictionary_hypo[neuron_number]=(text_outputs[0],'Presence in text '+str(output_bool))
        
    with open(path_hypo, 'w') as json_file:
        json.dump(dictionary_hypo, json_file, indent=4)

@gin.configurable
def generate_hypotheses_text(path_dataset:Path,path_embedding:Path, path_dictionary_neurons:Path,path_labels:Path,id_loader:int=0,device:torch.device='cuda:0')->None:
    """
    Generate possibles hypotheses of concept for each neuron saved in the df_neuron variables

    Args:
        path_dataset (Path): path to folder where is saved the dataset  
        path_dictionary_neurons (Path): path to folder where the dictionary with the outputs of the backbone is saved 
        path_embedding (Path): path to folder where the latent activations of the SAE are saved 
        path_labels (Path): path to folder were the labels are collected
        device (torch.device, optional): name of the device. Defaults to 'cpu'.
        
    Result:
        For each neuron in df_neurons, the function engender a txt file with the LlaVa prompt and besides and a summary.json 
        with a short version of each of all of them.
       


    """    
    modality='textual'
    if not os.path.exists(path_dictionary_neurons):
        list_neurons=range(5000)
        path_dictionary_neurons=create_dictionary_neurons(path_embedding,list_neurons,modality=modality)
      
        
       
    


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

    # Used to retrive the question and the image_id

    model.generation_config.temperature=None
    model.generation_config.top_p=None
    model.eval()
    
    

    average_activation_dictionary=json.load(open(path_labels, 'r'))

    df_label=pd.DataFrame.from_dict(average_activation_dictionary)
    
    dictionary_neurons=json.load(open(path_dictionary_neurons,'r'))
    dictionary_hypo={str(i): [] for i in range(5000)}

    path_hypo =Path(path_embedding+'dictionary_hypo_'+str(id_loader)+'_'+modality+'_.json')
    
    conv_template = "qwen_1_5"
    system="You are a meticulous AI researcher conducting an important investigation into a certain neuron in a vision language model. Your task is to analyze the neuron and provide an explanation that thoroughly encapsulates its behavior."
    conv = copy.deepcopy(conv_templates[conv_template])

    conv.system=system
    
    actual_conversation=conv
    
  
    for neuron_number,batch in tqdm(take(id_loader,dictionary_neurons.items()),desc=' Generate the Text Hypotheses ' ,total=1000,leave=False):
        ids_list=batch.keys()   
        #Used to memorize which images are used to derive the hypothesis
        
        texts=[]
        images=[]
       
        for id in ids_list:
            tmp_text=batch[id]["text_features"]['final_output'].replace('assistant','')
            tmp_text=tmp_text.replace('\n','')
            texts.append(df_label[id]['question'] + tmp_text)

            img_name=df_label[id]['image_name']
            folder_tmp = path_dataset+'train2014/' if 'train' in img_name else path_dataset+'val2014/'
            images.append(Image.open(folder_tmp+img_name).convert("RGB"))
            
        
    # TypeError: color must be int or single-element tuple

        if images:
        
            image_tensors = process_images(images, image_processor, model.config)
            
            image_tensors = [_image.to(dtype=torch.float16, device=model.device) for _image in image_tensors]
            image_sizes = [image.size for image in images]

                    # Prepare the template

            questions= questions_routine_vqa(texts,modality='text')
            
            with torch.no_grad():
                
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
    
        
                dictionary_hypo[neuron_number]=(text_outputs[0])
        else:
            dictionary_hypo[neuron_number]='No textual concept'
            
 
    with open(path_hypo, 'w') as json_file:
        json.dump(dictionary_hypo, json_file, indent=4)

@gin.configurable
def generate_hypotheses_image_llava(path_dataset:Path,path_embedding:Path, path_dictionary_neurons:Path,id_loader:int=0,device:torch.device='cuda:0')->None:
    """
    Generate possibles hypotheses of concept for each neuron saved in the df_neuron variables

    Args:
        path_dataset (Path): path to folder where is saved the dataset  
        path_dictionary_neurons (Path): path to folder where the dictionary with the outputs of the backbone is saved 
        path_embedding (Path): path to folder where the latent activations of the SAE are saved 
        path_labels (Path): path to folder were the labels are collected
        device (torch.device, optional): name of the device. Defaults to 'cpu'.
    Result:
        For each neuron in df_neurons, the function engender a txt file with the LlaVa prompt and besides and a summary.json 
        with a short version of each of all of them.
       


    """    
    modality='visual'
    if not os.path.exists(path_dictionary_neurons):
        list_neurons=range(5000)
        path_dictionary_neurons=create_dictionary_neurons(path_embedding,list_neurons,modality=modality)
        # path_dictionary_neurons=create_dictionary_neurons_llava_next(path_embedding,list_neurons)
        


    data = load_dataset("lmms-lab/LLaVA-NeXT-Data", split="train[:15%]", cache_dir=path_dataset, num_proc=10)
    path_hypo =Path(path_embedding+'dictionary_hypo_'+str(id_loader)+'_'+modality+'_.json')
 
   

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

    # Used to retrive the question and the image_id

    model.generation_config.temperature=None
    model.generation_config.top_p=None
    model.eval()
    
    dictionary_neurons=json.load(open(path_dictionary_neurons,'r'))
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
    
    for neuron_number, batch in tqdm(portion_dictionary_neurons, desc="Generate Image Hypotheses", total=1000, leave=False):
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
          
            questions= questions_routine_llava(texts,modality)

    
            with torch.no_grad():
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
                print(output)
                output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
                # print(text_outputs[0])
                conv.messages.pop()
                conv.messages.pop()
                dictionary_hypo[neuron_number] = output
                torch.cuda.empty_cache()


        
    with open(path_hypo, 'a') as json_file:
        json.dump(dictionary_hypo, json_file, indent=4)

@gin.configurable
def generate_hypotheses_text_llava(path_dataset:Path,path_embedding:Path, path_dictionary_neurons:Path,id_loader:int=0,device:torch.device='cuda:0')->None:
    """
    Generate possibles hypotheses of concept for each neuron saved in the df_neuron variables

    Args:
        path_dataset (Path): path to folder where is saved the dataset  
        path_dictionary_neurons (Path): path to folder where the dictionary with the outputs of the backbone is saved 
        path_embedding (Path): path to folder where the latent activations of the SAE are saved 
        path_labels (Path): path to folder were the labels are collected
        device (torch.device, optional): name of the device. Defaults to 'cpu'.
        
    Result:
        For each neuron in df_neurons, the function engender a txt file with the LlaVa prompt and besides and a summary.json 
        with a short version of each of all of them.
       


    """    
    modality='textual'
    if not os.path.exists(path_dictionary_neurons):
        list_neurons=range(5000)
        path_dictionary_neurons=create_dictionary_neurons(path_embedding,list_neurons,modality=modality)
      
        
       
     # train[:15%]
    data = load_dataset("lmms-lab/LLaVA-NeXT-Data", split="train[:15%]", cache_dir=path_dataset, num_proc=10)
    
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

    # Used to retrive the question and the image_id

    model.generation_config.temperature=None
    model.generation_config.top_p=None
    model.eval()
    
    



    
    
    dictionary_neurons=json.load(open(path_dictionary_neurons,'r'))
    dictionary_hypo={str(i): [] for i in range(5000)}

    path_hypo =Path(path_embedding+'dictionary_hypo_'+str(id_loader)+'_'+modality+'_.json')
    
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
                "image": example["image"].convert('RGB')
            }
        if len(lookup) >= len(needed_ids):
            break
    conv = copy.deepcopy(conv_templates[conv_template])

    conv.system=system
    
  
    
    for neuron_number,batch in tqdm(portion_dictionary_neurons,desc=' Generate the Text Hypotheses ' ,total=1000,leave=False):
        
        #Used to memorize which images are used to derive the hypothesis
        images=[]
        texts=[]
        
        

        for id_sample_str, feats in batch.items():
            id_sample = int(id_sample_str)
            entry = lookup.get(id_sample)
            
            if entry is None:
                continue

            # build prompt text
            convo = entry["conversations"][0]["value"]
            text_feat = feats["textual_features"]["final_output"]
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

            questions= questions_routine_llava(texts,modality=modality)
                
            with torch.no_grad():
                

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
                dictionary_hypo[neuron_number] = output
                torch.cuda.empty_cache()
                

            
 
    with open(path_hypo, 'w') as json_file:
        json.dump(dictionary_hypo, json_file, indent=4)
