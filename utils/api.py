import os
os.environ["SAE_DISABLE_TRITON"] = "0"
os.environ["TOKENIZERS_PARALLELISM"]="false"
import torch
import json
from sparsify.sparsify.sparse_coder import SparseCoder as SAE
from typing import Dict, List
from torchvision.transforms.functional import to_pil_image
from typing import Tuple
from transformers import LlavaNextProcessor ,LlavaNextForConditionalGeneration
from torch.utils.data import DataLoader,Dataset
from utils.dataset import VQAXDataset
from tqdm import tqdm
import numpy as np
from pathlib import Path
import gin
import glob
from datasets import load_dataset
    
def model_generation(model:LlavaNextForConditionalGeneration,actual_conversation:Dict,
                   content:str,processor:LlavaNextProcessor,images:List=None,max_new_tokens=20)->tuple[Dict,str]:
    """
    Call the model to generate a response based on the conversation history and the provided content.

    Args:
        model (LlavaLlamaForCausalLM): LLava model to be used.
        actual_conversation (Dict): Conversation dictionary to be updated.
        content (str): Text to be passed to the model.
        images (List, optional): List of images passed to the model. Defaults to "None".
        process(LlavaNextProcessor,optional): Processor object used both for process text+image(s) or simple text.
        max_new_tokens (int, optional): Maximum number of tokens to generate in the response. Defaults to 20.


    Returns:
       tuple[Dict, str]: A tuple containing:
                    - Dictionary with the updated conversation
                    - String containing the generated text output
                    - Raw model output tokens
    """ 
    
    if images:
        
        conversation_with_images=[
                    {

                    "role": "user",
                    "content": [
                        {"type": "text", "text": content},
                        {"type": "image"},
                        ],
                    },
                ]
        
        actual_conversation.extend(conversation_with_images)
        
        prompt = processor.apply_chat_template(actual_conversation, add_generation_prompt=True)
        
        inputs = processor(images=images, text=prompt, return_tensors="pt")
        
        with torch.inference_mode():
            output = model.generate(
                input_ids=inputs["input_ids"].to(model.device),
                pixel_values=inputs["pixel_values"].to(model.device),
                image_sizes=inputs["image_sizes"].to(model.device),
                attention_mask=inputs["attention_mask"].to(model.device),
                max_new_tokens=max_new_tokens,

        )
    else:
        conversation_witout_images=[
                    {

                    "role": "user",
                    "content": [
                        {"type": "text", "text": content},
                        
                        ],
                    },
                ]
        actual_conversation.extend(conversation_witout_images)
        prompt = processor.apply_chat_template(actual_conversation, add_generation_prompt=True)

        inputs = processor( text=prompt, return_tensors="pt").to(model.device)
        
        inputs_embeds =  model.get_input_embeddings()(inputs["input_ids"])
        with torch.inference_mode():
            output = model.generate(
                inputs_embeds= inputs_embeds,
                max_new_tokens=max_new_tokens
        )
            
    # Generate output
    
        
    # Clean up
    if images:
        del images
    
    text_output=processor.decode(output[0][-max_new_tokens:], skip_special_tokens=True)

    # Extend the converstation with the answer
    tmp_conversation=[
                {

                "role": "assitant",
                "content": [
                    {"type": "text", "text": text_output},
                    
                    ],
                },
            ]
    actual_conversation.extend(tmp_conversation)

    return actual_conversation,text_output,output[0]

def initialize_llava(model:LlavaNextForConditionalGeneration,actual_conversation:Dict,processor:LlavaNextProcessor,max_new_tokens=20):
    """
    Initialize a conversation for the LLaVA model to extract features from the VQA.

    Args:
        model (LlavaNextForConditionalGeneration): The LLaVA model instance
        actual_conversation (Dict): Current conversation state/history
        processor (LlavaNextProcessor): Processor for handling inputs
        max_new_tokens (int, optional): Maximum number of tokens to generate. Defaults to 20.

    Returns:
        Dict: Updated conversation with system instructions and initial response
    """   

    # Initializate the prompt
    prompt = processor.apply_chat_template(actual_conversation, add_generation_prompt=True)
    
    inputs = processor( text=prompt, return_tensors="pt").to(model.device)
    inputs_embeds =  model.get_input_embeddings()(inputs["input_ids"])
    #Generate the response
    with torch.inference_mode():
        output = model.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=max_new_tokens
        )
     # Extend the converstation with the answer
    output_final=processor.decode(output[0][-max_new_tokens:], skip_special_tokens=True)
    tmp_conversation=[
        {

            "role": "assitant",
            "content": [
                {"type": "text", "text": output_final},
                
                ],
        },
    ]
    # Add the response to the conversation
    actual_conversation.extend(tmp_conversation)
    

    return actual_conversation

def extract_matching_neuron_values_indices(indices_tensor: torch.tensor, acts_tensor:torch.tensor, target_tensor: torch.tensor):
    """
    Extract indices and activation values for neurons present in the target_tensor.

      Args:
        list_indices (torch.Tensor): Tensor of shape (num_patches, k) containing neuron indices for each patch.
        list_acts (torch.Tensor): Tensor of shape (num_patches, k) containing activation values for each patch.
        target_tensor (torch.Tensor): 1D tensor of neuron indices to filter for.
    Note:
        Standard set up:
        num_patches=576
        k=256
    Returns:
        List[int], List[int]
    """    
    
    # Pre-allocate lists with estimated size
    matching_indices = []
    matching_values = []
    
    # Process all patches at once using torch operations
    mask = torch.isin(indices_tensor, target_tensor)
    indices_where = torch.where(mask)
    
    # Group by patch number
    for patch_idx in range(len(indices_tensor)):
        patch_mask = indices_where[0] == patch_idx
        if patch_mask.any():
            matching_indices.append(indices_tensor[patch_idx][indices_where[1][patch_mask]].tolist())
            matching_values.append(acts_tensor[patch_idx][indices_where[1][patch_mask]].tolist())
    
    return matching_indices, matching_values
  
@gin.configurable
def create_average_activation_dictionary(embedding_file_path:Path,embedding_folder_path,modality:str)->None:
    """Create a dictionary mapping neurons to their activation statistics in VQA samples.

    For each analyzed neuron, store information about which VQA samples activated it,
    including the average activation value and count of image patches involved.

    Args:
        embedding_file_path (Path):Path containing embedding results
        embedding_path_path (Path):Path to the embedding folder
        modality (str, optional): Modality to process ('visual' or 'textual').
    Returns:
        None. Saves results to 'average_activation_dictionary.json' with structure:
        {
            neuron_id: {
                vqa_sample_id: [activation_average, patch_count],
                ...
            },
            ...
        }
    """

    neuron_activation_stats_dict = {str(i): {} for i in range(5000)}

    res=json.load(open(embedding_file_path))
    for key, value in res.items():
       
        neuron_activation_stats_dict=average_values_indices(value[modality+'_features']['latent_indices'],value[modality+'_features']['latent_acts'],neuron_activation_stats_dict,key)  
       
    with open(embedding_folder_path+"average_activation_"+modality+"_dictionary.json", "a") as f:
            json.dump(neuron_activation_stats_dict, f, indent=4)
    
        
def average_values_indices(lists_indices: List[List[int]], lists_acts:List[List[int]],neuron_activation_stats_dict:dict,id_sample:str)->Dict:

    """Compute the averages activations values for all active neurons for each patch, or token generated, in a single image, or single text, and adding to 
        the total dictionary

    Args:
        lists_indices (List[List[int]]): List of lists, where each inner list contains indices of activated neurons for a patch.
        lists_acts (List[List[int]]): List of lists, where each inner list contains activation values for the corresponding neurons in a patch.
        neuron_activation_stats_dict (dict): Dictionary to accumulate activation statistics for each neuron.
        id_sample (str): Identifier for the current sample (e.g., VQA sample ID or LlaVA sample ID).
    Comments:
        both list have the same shape (576,[range between 0 to 256]) for images  and for text (num_token_analyzed,[range between 0 to 256])
        The [range between 0 to 256] is the intersection between the neurons designed to be analized (generally first 5000) and the best 256 k highest valued neuron in the patch or string of tokens
      
    Returns:
        Dictionary updated (see above for the structure)
    """
    """
    

    Notes:
        - Both lists have length 576 (number of patches), with each inner list containing up to 256 activated neurons per patch.
        - Only neurons present in both the analysis set and the top-256 activations per patch are included.
        - The function updates the dictionary with the average activation and count of patches for each neuron in the current sample.

    Returns:
        dict: Updated neuron activation statistics dictionary.
    
    """
    # Use numpy arrays for faster computation
    neuron_sums = {}
    neuron_counts = {}
    
    # Process all patches at once
    for patch_indices, patch_acts in zip(lists_indices, lists_acts):
        for neuron_idx, activation in zip(patch_indices, patch_acts):
            if neuron_idx not in neuron_sums.keys():
                neuron_sums[neuron_idx] = 0.0
                neuron_counts[neuron_idx] = 0
                
            neuron_sums[neuron_idx] += activation
            neuron_counts[neuron_idx] += 1

    # Calculate averages and update stats dictionary
    for key,value in neuron_sums.items():
        # Store [average_activation, patch_count]
        avg_activation = value / 576  # Total patches
        neuron_activation_stats_dict[str(key)][id_sample] = [
            avg_activation,
            neuron_counts[key]
        ]

    return neuron_activation_stats_dict
@gin.configurable
def llava_extract(embedding_path,dataset_path,layer:str='24',device=torch.device('cuda:0')):  
    """
    Extract sparse autoencoder (SAE) features from the LLaVA-NeXT model for both image and text modalities.

    Args:
        embedding_path (str): Directory where the extracted feature JSON files will be saved.
        dataset_path (str): Directory for caching/loading the LLaVA-NeXT dataset.
        layer (str, optional): Model layer to extract features from. Defaults to '24'.
        device (torch.device, optional): Device to run computations on. Defaults to cuda:0.

    Output:
        None. Saves results to a JSON file 

    Notes:
        - Processes the LLaVA-NeXT-Data dataset (15% of the training split).
        - Uses the LLaVA-NeXT 8B model with SDPA attention and 16-bit precision.
        - Extracts features from layer 24 using a pre-trained SAE.
        - Handles both image and text inputs, extracting features for each.
        - Results are saved incrementally to avoid data loss during processing.
    """
    # The first 5000 neurons
    target_tensor = torch.tensor(range(5000), device=device)
  
    data=load_dataset("lmms-lab/LLaVA-NeXT-Data", split="train[:15%]", cache_dir=dataset_path, num_proc=10)
    processor = LlavaNextProcessor.from_pretrained("llava-hf/llama3-llava-next-8b-hf")
    model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llama3-llava-next-8b-hf",attn_implementation="sdpa", torch_dtype=torch.float16, device_map="auto")
    model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
    model.eval()
    
    
    
    def forward_hook(model, input, output):
        if hooked_res["hidden_states"] is not None:
            hooked_res["hidden_states"].append(output)
        else:
            hooked_res["hidden_states"] = [output]
        return output


    # Register the hook
    hook_gen=model.language_model.model.layers[int(layer)].register_forward_hook(forward_hook)
    #Tag used to identify the image and the text
    image_tag=processor(text="<image>")
    image_tag=image_tag["input_ids"][0][1]
    hookpoint="model.layers."+str(layer)
    sae = SAE.load_from_hub("lmms-lab/llama3-llava-next-8b-hf-sae-131k", hookpoint=hookpoint).to(device)
    sae.eval()
    
    results_dict={}
    
    for batch in tqdm(data,desc='Extract from LlaVA'):
        hooked_res = {"hidden_states": None} 
        with torch.inference_mode():
            
            id_dictionary=batch['id']
            text=batch['conversations'][1]['value']
            if id_dictionary not in results_dict.keys():
                
                if batch['image']:
        
                    
                    img=batch['image'].resize((336,336))
                    img_conversation=[
                                {
                                    "role": "system",
                                    "content": [
                                        {"type": "image"},
                                        {"type": "text", "text":text},
                                    ],
                                },
                        ]
                    
                    prompt = processor.apply_chat_template(img_conversation, add_generation_prompt=True)     
                    
                    inputs = processor( images=img,text=prompt, return_tensors="pt").to(model.device)
                    
                    
                    _ = model(
                        
                            input_ids=inputs["input_ids"].to(model.device),
                            pixel_values=inputs["pixel_values"].to(model.device),
                            image_sizes=inputs["image_sizes"].to(model.device),
                            attention_mask=inputs["attention_mask"].to(model.device)
                    )
                    
                    
                    hidden_state=hooked_res["hidden_states"][0]
                    
                    indices_visual_tags=torch.where(inputs['input_ids'][0] == image_tag)[0][:576]
                    indices_textual_tags=torch.where(inputs['input_ids'][0] != image_tag)[0][5:]
                    result_sae_visual = sae(hidden_state[0][0][indices_visual_tags].to(sae.device))
                    result_sae_textual = sae(hidden_state[0][0][indices_textual_tags].to(sae.device))
                    
                         
                    latent_indices_visual,latent_acts_visual =extract_matching_neuron_values_indices(result_sae_visual.latent_indices.to(device),
                                                                                                     result_sae_visual.latent_acts.to(device),target_tensor)
                    latent_indices_textual,latent_acts_textual =extract_matching_neuron_values_indices(result_sae_textual.latent_indices.to(device),
                                                                                                       result_sae_textual.latent_acts.to(device),target_tensor) 
                    
                    
                else:
                    latent_acts_visual=None
                    latent_indices_visual=None
                    
                if text is None:
                    latent_indices_textual=None
                    latent_acts_textual=None
            else:
                
                latent_acts_visual=results_dict[batch['id']]["visual_features"]["latent_acts"]
                latent_indices_visual=results_dict[batch['id']]["visual_features"]["latent_indices"]
                latent_acts_textual=results_dict[batch['id']]["textual_features"]["latent_acts"]
                latent_indices_textual=results_dict[batch['id']]["textual_features"]["latent_indices"]
        
            results_dict[id_dictionary]={
                "visual_features":
                    {"latent_acts":latent_acts_visual,
                    "latent_indices":latent_indices_visual},
                "textual_features": 
                    {"latent_acts":latent_acts_textual,
                    "latent_indices":latent_indices_textual,
                    "final_output":text}
                }
          
            
            
            torch.cuda.empty_cache()

    hook_gen.remove()
    with open(embedding_path+"llava_embeddings.json", "a") as f:
        json.dump(results_dict, f, indent=4)
@gin.configurable
def coco_extract(vqa_dataset:VQAXDataset,path_sae:Path,path_embedding:Path,device:torch.device='cuda:0'):
    """
    Extract sparse autoencoder features from a VQA dataset using the LLaVA-NeXT model.

    Args:
        vqa_dataset (VQAXTrainDataset): VQA dataset to process.
        
        path_sae (Path): Path to the pre-trained sparse autoencoder.
        path_embedding (Path): Path to the save folder.
        id_loader (int, optional): Index of the data subset to process. Defaults all data subset, i.e. id_loader=-1.
        device (torch.device): The cuda device. Defaults to cuda:0.

    Returns:
        None. The function saves extracted features to a JSON file.
    """
    
    
    #Donwload the dataset and partition it
    vqa_dataset=VQAXDataset()
    

    # All in once single run
    train_loader=DataLoader(vqa_dataset, batch_size=1, shuffle=False, num_workers=4)
        
   

    # Load SAE and LLaVA models
    sae = SAE.load_from_hub(path_sae, hookpoint="model.layers.24").to(device)
    sae.eval()
    
    processor = LlavaNextProcessor.from_pretrained("llava-hf/llama3-llava-next-8b-hf")
    model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llama3-llava-next-8b-hf",attn_implementation="sdpa", torch_dtype=torch.float16, device_map="auto")
    model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
    model.eval()
    #Prepare the LLaVA model, initializing the prompt and the system prompt
  
    
    
   

    max_new_tokens=20
    # Standard, always extract the first 5000 neurons from the SAE
    target_tensor = torch.tensor(range(5000), device=device)

    # Creation of the hook
    def forward_hook(model, input, output):
        if hooked_res["hidden_states"] is not None:
            hooked_res["hidden_states"].append(output)
        else:
            hooked_res["hidden_states"] = [output]
        return output


    # Register the hook
    hook_gen=model.language_model.model.layers[24].register_forward_hook(forward_hook)
    
    #Tag used to identify the image and the text
    image_tag=processor(text="<image>")
    image_tag=image_tag["input_ids"][0][1]
    
    results_dict = {
    }
    with torch.inference_mode():
        hooked_res = {"hidden_states": None}
        for batch in tqdm(train_loader, desc="Extraction embedding", leave=True):
            img,question,_,_,_,id_sample=batch
            id_sample=id_sample[0]
            images = to_pil_image(img.squeeze(0))
            

        

            img_conversation=[
                        {
                            "role": "system",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text":question},
                            ],
                        },
                ]
            
            prompt = processor.apply_chat_template(img_conversation, add_generation_prompt=True)     
            
            inputs = processor( images=images,text=prompt, return_tensors="pt").to(model.device)
            
            
            _ = model(
                
                    input_ids=inputs["input_ids"].to(model.device),
                    pixel_values=inputs["pixel_values"].to(model.device),
                    image_sizes=inputs["image_sizes"].to(model.device),
                    attention_mask=inputs["attention_mask"].to(model.device)
                    )
              # Extract the hidden states linked with the text output, after the processing of the image (first hidden_state chunk) and before the 

          
            hidden_state_textual = [state[0] for state, _ in hooked_res["hidden_states"][1:-1]]
            
            hidden_state_textual = torch.cat(hidden_state_textual, dim=0)
            
            # Extract the hidden states linked with the image + question input (first hidden_state chunk)
            hidden_state_visual = hooked_res["hidden_states"][0][0][0]
            
            
            indices_image_tags=torch.where(output_ids == image_tag)[0][:576]
            
            # Apply sparse autoencoder to the hidden state
            result_sae_visual = sae(hidden_state_visual[indices_image_tags].to(sae.device))
            result_sae_textual = sae(hidden_state_textual.to(sae.device))
            #Pop the last two messages, question and answer to re-use actual_conversation
         
            #Clean the hook response
            hooked_res = {"hidden_states": None}

            latent_indices_visual,latent_acts_visual =extract_matching_neuron_values_indices(result_sae_visual.latent_indices.to(device),
                                                                                             result_sae_visual.latent_acts.to(device),target_tensor) 
            
            latent_indices_textual,latent_acts_textual =extract_matching_neuron_values_indices(result_sae_textual.latent_indices.to(device),
                                                                                         result_sae_textual.latent_acts.to(device),target_tensor) 
            
            results_dict[id_sample]={
                "visual_features":
                    {"latent_acts":latent_acts_visual,
                    "latent_indices":latent_indices_visual},
                "textual_features": 
                    {"latent_acts":latent_acts_textual,
                    "latent_indices":latent_indices_textual,
                    }
                }

            del result_sae_visual,result_sae_textual,batch,output_ids,images
            torch.cuda.empty_cache()
            
    hook_gen.remove()
    
   
 
    with open(path_embedding+"coco_embeddings.json", "a") as f:
        json.dump(results_dict, f, indent=4)

def compute_fvu(path_file:str)->None:
    """Compute the mean, variance, and number of elements from values in a JSON file.
    
    Args:
        path_file (str): Path to input JSON file containing numerical data.

    Returns:
        tuple: Contains the following elements:
            - float: Mean of the values
            - float: Variance of the values 
            - int: Number of elements analyzed
    """
    data = json.load(open(path_file))
    values = np.array(list(data))
    mean = np.mean(values)
    variance = np.var(values) 
    print(f"Mean text: {mean}")
    print(f"Variance text: {variance}")
    print(f"Num elements text: {len(values)}")
    return mean,variance,len(values)

def create_neuron_top5_dictionary(embedding_path:Path, neuron_list:List[int],modality:str='visual')->None:
        """
        Create a dictionary mapping each neuron to its top-5 most activated samples.

        Args:
            embedding_path (Path): Path to the folder where embeddings are saved.
            neuron_list (List[int]): List of neuron indices to process.
            modality (str): Modality type, either 'visual' or 'textual':Default.

        Returns:
            Path: Path to the saved dictionary JSON file.
        """
        # Determine which JSON files to load based on the folder name
        if 'coco' in embedding_path:
            json_pattern = 'coco_embbeddings*.json'
        else:
            json_pattern = 'llava_embeddings*.json'
        
        dictionary_neurons={}
        name_dictionary='neuron_top5_'+modality+'_dictionary.json'
        #Check if the dictionary with all the average neuron activations of exists  
        avg_act_dict_path = os.path.join(embedding_path, f'average_activation__{modality}_dictionary.json')
        if not os.path.exists(avg_act_dict_path):
            #create the average neruon activations dictionary for the modality
            
            create_average_activation_dictionary(embedding_path=embedding_path,embedding_file_path=embedding_path,modality=modality)


        embedding_file_path = os.path.join(embedding_path, json_pattern)
        data=json.load(open(embedding_file_path))
        average_activation_dictionary = json.load(open(avg_act_dict_path))
        for neuron in tqdm(neuron_list,desc='Sorting activations and creating '+name_dictionary+' file'):
            # Sort by activation value (descending)
            sorted_list = sorted(average_activation_dictionary[str(neuron)].items(), key=lambda x: x[1][1], reverse=True)
            # Take the top 5 activated samples
            top_5_samples=[sample_id[0] for sample_id in sorted_list[0:5]]

            dictionary_neurons[neuron]={sample_id:data[sample_id] for sample_id in top_5_samples}
            
        result_path=os.path.join(embedding_path, name_dictionary)   
        with open(result_path, 'a') as f:
            json.dump(dictionary_neurons, f)
        return result_path

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
    for el in tqdm(files, desc='Unite the dictionaries'):
        
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
    