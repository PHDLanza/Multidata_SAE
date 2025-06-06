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

def check_neuron_in_text(id_neuron:int, list_text_indices:List):
    for patch in list_text_indices:
        if id_neuron in patch:
            return True
    return False

def questions_routine_vqa(texts_list:List,modality:str='visual'):
    """
    Prepares a prompt for the VQA routine to extract shared concepts from a list of texts and images.

    Args:
        texts_list (List): List of text entries to analyze for concept extraction.

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
    
    for text  in texts_list:
        GUIDELINES += " image {}: question and answer {}.\n\n".format(DEFAULT_IMAGE_TOKEN,text)

    return [GUIDELINES]

def questions_routine_llava(texts_list:List,modality:str='visual'):
    """
    Prepares a prompt for the Llava-Next routine to extract shared concepts from a list of texts and images.

    Args:
        texts_list (List): List of text entries to analyze for concept extraction.

    Returns:
        List: A list containing the formatted prompt string for the model.
    """ 
    if modality=='visual':
    
        GUIDELINES = """ 
            [REQUIREMENTS]
                Focus only on the highlighted region in each image. If no region is highlighted or if the highlighted region is minimal (e.g., a few bright spots), ignore the image.
            
                Identify common visual patterns, objects, or concepts in the activated regions. For example, note if highlighted areas show consistent structures, such as mesh patterns or similar objects.\
            
            [GUIDELINES]
            Consider Text Context: While maintaining primary focus on the highlighted regions in images, you may marginally consider the associated text (questions and answers) to support or refine your visual observations. 
            However, the final concept should be predominantly based on visual patterns.
                Concise Description Only: Provide a short, direct description of the common features within the highlighted regions. Avoid any interpretive language—simply state what you see, such as “mesh-like structures” or “actions related to joy or happiness”
            
            1. Describe Only the Highlighted Regions: Generate captions solely based on the highlighted regions. If no meaningful pattern is visible, or if only a few scattered spots are highlighted,
                output: \"Concept:  `No visual concept`\"
                
            2. Consider Text Context: While maintaining primary focus on the highlighted regions in images, you may marginally consider the associated text (questions and answers) to support or refine your visual observations. However, the final concept should be predominantly based on visual patterns.
                
            3. Concise Description Only: Provide a short, direct description of the common features within the highlighted regions.
              Avoid any interpretive language—simply state what you see, such as “mesh-like structures” or “actions related to joy or happiness”
        """
    else:
        GUIDELINES=""" 
            [REQUIREMENTS]
               Focus only on the text content provided with each example. If the text is missing, irrelevant, or extremely minimal (e.g., a few unrelated words), ignore that example.

            Identify common themes, objects, or concepts mentioned across the text snippets. Pay special attention to any highlighted word in each text—this word should be treated as the most important cue for concept identification.

        [GUIDELINES]

          1.You will receive a series of text snippets, sometimes accompanied by images. Only use the text, and in particular the word between parentheses, to identify the shared concept. Images should not be considered in your analysis.
            These examples are derived from a Visual Question Answering dataset, so each text is in the form of a question or an answer.

            2.Concise Description Only: Provide a short, direct description of the common concept emerging from the texts. Avoid speculation or abstract interpretation—simply state what is explicitly or implicitly repeated, especially in relation to the highlighted words (e.g., "vehicles," "cooking actions," "types of animals").
            Use the image only for reference if absolutely necessary; the main analysis must be text-driven, with words in parentheses as priority.

            3.If no clear concept emerges from the texts (e.g., if they are too diverse or vague), write: No textual concept

  
            """


    GUIDELINES += """\n\n"""
    
    for text  in texts_list:
        GUIDELINES += " image {}: question and answer {}.\n\n".format(DEFAULT_IMAGE_TOKEN,text)

    return [GUIDELINES]
def create_dictionary_neurons(folder_save_embedding:Path, list_neurons:List[int],modality:str='visual')->None:
        """
        Create a dictionary mapping each neuron to its top-5 most activated samples.

        Args:
            folder_save_embedding (Path): Path to the folder where embeddings are saved.
            list_neurons (List[int]): List of neuron indices to process.
            modality (str): Modality type, either 'image' or 'text':Default.

        Returns:
            Path: Path to the saved dictionary JSON file.
        """
        
        json_files = glob.glob(os.path.join(folder_save_embedding, 'llava_15_block_*.json'))

        # json_files = glob.glob(os.path.join(folder_save_embedding, 'vqa_block*.json'))

        # Create a combined dictionary for all files
        combined_data = {}

        # Read and combine all json files
        for file_path in json_files:
            with open(file_path, 'r') as f:
                data = json.load(f)
                combined_data.update(data)

        dictionary_neurons={}
        name_dictionary='dictionary_neurons_'+modality+'.json'
        average_activation_dictionary = json.load(open(folder_save_embedding+'average_activation_dictionary_'+modality+'.json'))
        for neuron in tqdm(list_neurons,desc='Sorting the activations and creating the '+name_dictionary):
            sorted_list = sorted(average_activation_dictionary[str(neuron)].items(), key=lambda x: x[1][1], reverse=True)

            new_sorted_list=[el[0] for el in sorted_list[0:5]]

            dictionary_neurons[neuron]={el:combined_data[el] for el in new_sorted_list}
            
            
        with open(os.path.join(folder_save_embedding, name_dictionary), 'a') as f:
            json.dump(dictionary_neurons, f)
        return folder_save_embedding+name_dictionary
 
@gin.configurable
def generate_hypotheses_image(folder_dataset:Path,folder_save_embedding:Path, dictionary_neurons_path:Path,folder_labels:Path,id_loader:int=0,device:torch.device='cuda:0')->None:
    """
    Generate possibles hypotheses of concept for each neuron saved in the df_neuron variables

    Args:
        folder_dataset (Path): path to folder where is saved the dataset  
        dictionary_neurons_path (Path): path to folder where the dictionary with the outputs of the backbone is saved 
        folder_save_embedding (Path): path to folder where the latent activations of the SAE are saved 
        folder_labels (Path): path to folder were the labels are collected
        device (torch.device, optional): name of the device. Defaults to 'cpu'.
    Result:
        For each neuron in df_neurons, the function engender a txt file with the LlaVa prompt and besides and a summary.json 
        with a short version of each of all of them.
       


    """    
    modality='visual'
    if not os.path.exists(dictionary_neurons_path):
        list_neurons=range(5000)
        dictionary_neurons_path=create_dictionary_neurons(folder_save_embedding,list_neurons,modality=modality)
        # dictionary_neurons_path=create_dictionary_neurons_llava_next(folder_save_embedding,list_neurons)
        
       
    


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
    
    

    average_activation_dictionary=json.load(open(folder_labels, 'r'))

    df_label=pd.DataFrame.from_dict(average_activation_dictionary)
    
    dictionary_neurons=json.load(open(dictionary_neurons_path,'r'))
    dictionary_hypo={str(i): [] for i in range(5000)}

    path_hypo =Path(folder_save_embedding+'dictionary_hypo_'+str(id_loader)+'_'+modality+'_.json')

    
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
            folder_tmp = folder_dataset + ('train2014/' if is_train else 'val2014/')
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
def generate_hypotheses_text(folder_dataset:Path,folder_save_embedding:Path, dictionary_neurons_path:Path,folder_labels:Path,id_loader:int=0,device:torch.device='cuda:0')->None:
    """
    Generate possibles hypotheses of concept for each neuron saved in the df_neuron variables

    Args:
        folder_dataset (Path): path to folder where is saved the dataset  
        dictionary_neurons_path (Path): path to folder where the dictionary with the outputs of the backbone is saved 
        folder_save_embedding (Path): path to folder where the latent activations of the SAE are saved 
        folder_labels (Path): path to folder were the labels are collected
        device (torch.device, optional): name of the device. Defaults to 'cpu'.
        
    Result:
        For each neuron in df_neurons, the function engender a txt file with the LlaVa prompt and besides and a summary.json 
        with a short version of each of all of them.
       


    """    
    modality='textual'
    if not os.path.exists(dictionary_neurons_path):
        list_neurons=range(5000)
        dictionary_neurons_path=create_dictionary_neurons(folder_save_embedding,list_neurons,modality=modality)
      
        
       
    


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
    
    

    average_activation_dictionary=json.load(open(folder_labels, 'r'))

    df_label=pd.DataFrame.from_dict(average_activation_dictionary)
    
    dictionary_neurons=json.load(open(dictionary_neurons_path,'r'))
    dictionary_hypo={str(i): [] for i in range(5000)}

    path_hypo =Path(folder_save_embedding+'dictionary_hypo_'+str(id_loader)+'_'+modality+'_.json')
    
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
            folder_tmp = folder_dataset+'train2014/' if 'train' in img_name else folder_dataset+'val2014/'
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

def unite_dictionaries(dictionaries_neuron_path:Path,modality:str='visual')->None:

    """Unites multiple dictionary files into a single complete dictionary file.
        Execute this function after generate hypotheses for each neuron in SAE (thorugh generate_hypotheses)  
    Args:
        dictionaries_neuron_path (Path): Path to the directory containing the dictionary files 
            and where the output file will be saved.
        modality (str, optional): Modality of the input ('image' or 'text'). Default is 'image'.
    Notes:
        
        - Input files should follow the pattern 'dictionary_hypo_*{modality}_.json'
        - Each input dictionary is filtered based on numerical ranges (1000 numbers per file)
        - The final dictionary is sorted by numerical keys
        - Output is saved as 'dictionary_hypotheses_complete_{modality}.json' in the same directory
     Returns:
        None
    
    """    
    
    files=glob.glob(dictionaries_neuron_path+'dictionary_hypo_*'+modality+'_.json')
    dictionary={}
    for el in files:
        
        num=el.split('_')[-3]
        index_start=int(num)*1000
        index_end=int(index_start)+999
        tmp_dict = json.load(open(el))
        filtered_dict = {k: v for k, v in tmp_dict.items() if int(k) >= index_start and int(k) <= index_end}
        if dictionary=={}:
            dictionary=filtered_dict  
        else:
            dictionary.update(filtered_dict)
    sorted_dictionary = dict(sorted(dictionary.items(), key=lambda x: int(x[0])))
    with open(dictionaries_neuron_path+'dictionary_hypotheses_complete_'+modality+'.json', 'a') as f:
        json.dump(sorted_dictionary, f, indent=4)
        
@gin.configurable
def generate_hypotheses_image_llava(folder_dataset:Path,folder_save_embedding:Path, dictionary_neurons_path:Path,id_loader:int=0,device:torch.device='cuda:0')->None:
    """
    Generate possibles hypotheses of concept for each neuron saved in the df_neuron variables

    Args:
        folder_dataset (Path): path to folder where is saved the dataset  
        dictionary_neurons_path (Path): path to folder where the dictionary with the outputs of the backbone is saved 
        folder_save_embedding (Path): path to folder where the latent activations of the SAE are saved 
        folder_labels (Path): path to folder were the labels are collected
        device (torch.device, optional): name of the device. Defaults to 'cpu'.
    Result:
        For each neuron in df_neurons, the function engender a txt file with the LlaVa prompt and besides and a summary.json 
        with a short version of each of all of them.
       


    """    
    modality='visual'
    if not os.path.exists(dictionary_neurons_path):
        list_neurons=range(5000)
        dictionary_neurons_path=create_dictionary_neurons(folder_save_embedding,list_neurons,modality=modality)
        # dictionary_neurons_path=create_dictionary_neurons_llava_next(folder_save_embedding,list_neurons)
        


    data = load_dataset("lmms-lab/LLaVA-NeXT-Data", split="train[:15%]", cache_dir=folder_dataset, num_proc=10)
    path_hypo =Path(folder_save_embedding+'dictionary_hypo_'+str(id_loader)+'_'+modality+'_.json')
 
   

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
                
                output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
                # print(text_outputs[0])
                conv.messages.pop()
                conv.messages.pop()
                dictionary_hypo[neuron_number] = output
                torch.cuda.empty_cache()

        else:
            dictionary_hypo[neuron_number]='No visual concept'
        
    with open(path_hypo, 'a') as json_file:
        json.dump(dictionary_hypo, json_file, indent=4)

@gin.configurable
def generate_hypotheses_text_llava(folder_dataset:Path,folder_save_embedding:Path, dictionary_neurons_path:Path,id_loader:int=0,device:torch.device='cuda:0')->None:
    """
    Generate possibles hypotheses of concept for each neuron saved in the df_neuron variables

    Args:
        folder_dataset (Path): path to folder where is saved the dataset  
        dictionary_neurons_path (Path): path to folder where the dictionary with the outputs of the backbone is saved 
        folder_save_embedding (Path): path to folder where the latent activations of the SAE are saved 
        folder_labels (Path): path to folder were the labels are collected
        device (torch.device, optional): name of the device. Defaults to 'cpu'.
        
    Result:
        For each neuron in df_neurons, the function engender a txt file with the LlaVa prompt and besides and a summary.json 
        with a short version of each of all of them.
       


    """    
    modality='text'
    path_hypo =Path(folder_save_embedding+'dictionary_hypo_'+str(id_loader)+'_'+modality+'_.json')
    if not os.path.exists(dictionary_neurons_path):
        list_neurons=range(5000)
        dictionary_neurons_path=create_dictionary_neurons(folder_save_embedding,list_neurons,modality=modality)
      
        
       
     # train[:15%]
    data = load_dataset("lmms-lab/LLaVA-NeXT-Data", split="train[:15%]", cache_dir=folder_dataset, num_proc=10)
    
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
    
    



    
    
    dictionary_neurons=json.load(open(dictionary_neurons_path,'r'))
    dictionary_hypo={str(i): [] for i in range(5000)}

    path_hypo =Path(folder_save_embedding+'dictionary_hypo_'+str(id_loader)+'_'+modality+'_.json')
    
    conv_template = "qwen_1_5"
    system="You are a meticulous AI researcher conducting an important investigation into a certain neuron in a vision language model. Your task is to analyze the neuron and provide an explanation that thoroughly encapsulates its behavior."
    portion_dictionary_neurons=take(id_loader, dictionary_neurons.items())
    ##where to cut
  
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
    conv = copy.deepcopy(conv_templates[conv_template])

    conv.system=system
    
    actual_conversation=conv
    
  
    for neuron_number,batch in tqdm(portion_dictionary_neurons,desc=' Generate the Text Hypotheses ' ,total=1000,leave=False):
        ids_list=batch.keys()   
        #Used to memorize which images are used to derive the hypothesis
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
            
    else:
        dictionary_hypo[neuron_number]='No textual concept'
            
 
    with open(path_hypo, 'w') as json_file:
        json.dump(dictionary_hypo, json_file, indent=4)
