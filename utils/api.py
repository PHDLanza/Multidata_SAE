import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["HF_HUB_CACHE"]="/data/lanza/hub"
os.environ["SAE_DISABLE_TRITON"] = "0"
os.environ["TOKENIZERS_PARALLELISM"]="false"
os.environ["PATH"] += os.pathsep + "/sbin/"

import torch
import json

from sparsify.sparsify.sparse_coder import SparseCoder as SAE
from typing import Dict, List

from torchvision.transforms.functional import to_pil_image



from typing import Tuple
from transformers import LlavaNextProcessor ,LlavaNextForConditionalGeneration
from torch.utils.data import DataLoader,Dataset


from utils.dataset import VQAXTrainDataset
from tqdm import tqdm
import numpy as np

from pathlib import Path
import gin
import glob
from datasets import load_dataset

@gin.configurable
def vqa_extract(train_dataset:VQAXTrainDataset,device:torch.device,path_sae:Path,folder_save_embedding:Path,id_loader=-1):
    """
    Extract features from the VQA_dataset using a sparse autoencoder and LLaVA-NeXT model.

    Args:
        train_dataset (VQAXTrainDataset): Dataset containing VQA samples.
        device (torch.device): Device to run computations on.
        path_sae (Path): Path to the pre-trained sparse autoencoder.
        folder_save_embedding (Path): Directory to save extracted embeddings.
        id_loader (int, optional): Index of the data subset to process. Defaults to -1.

    Returns:
        None. Saves extracted features to a JSON file.
    """
    target_tensor = torch.tensor(range(5000), device=device)
    
    max_new_tokens=10
    
    train_dataset=VQAXTrainDataset()
    if id_loader!=-1:
      
        generator1 = torch.Generator().manual_seed(42)
        tmp_dataset=torch.utils.data.random_split(train_dataset,[0.16, 0.16, 0.16, 0.16, 0.16, 0.20],generator1)
        train_loader=DataLoader(tmp_dataset[id_loader], batch_size=1, shuffle=False, num_workers=4)
        
    else:
        
        train_loader=DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=4)


    
    processor = LlavaNextProcessor.from_pretrained("llava-hf/llama3-llava-next-8b-hf")
    model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llama3-llava-next-8b-hf",attn_implementation="sdpa", torch_dtype=torch.float16, device_map="auto",load_in_4bit=True)
    model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
    model.eval()
    system="""You are a Visual Question Answering (VQA) model. Your task is to analyze an input image and answer questions about it. Your answers must always be a single word, without explanations, punctuation, or additional text.

    If the question requires a yes/no answer, respond with "yes" or "no" only. If the question asks for an object, color, action, or any other descriptor, respond with the most relevant single word.
    Examples:

        Question: What color is the car?
        Answer: Red

        Question: What animal is in the picture?
        Answer: Dog

        Question: Is the person smiling?
        Answer: Yes

        Question: What is the person doing?
        Answer: Running

        Question: What object is on the table?
        Answer: Laptop

        Question: What is the weather like?
        Answer: Sunny

        Question: How many apples are there?
        Answer: Three

        Question: What shape is the object?
        Answer: Circle
            
    """
    actual_conversation=[
                        {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": system},
                            
                            ],
                        },
                    ]
    actual_conversation=initialize_llava_vqa(model=model,processor=processor,actual_conversation=actual_conversation)
    hooked_res = {"hidden_states": None}
    
 

    sae = SAE.load_from_hub(path_sae, hookpoint="model.layers.24").to(device)
    sae.eval()

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
    # header_end=processor(text="<|end_header_id|>")
    # header_end=header_end["input_ids"][0][1]
    
    results_dict = {
    }
    with torch.no_grad():
        for batch in tqdm(train_loader, desc="Extraction embedding", leave=True):
            img,question,_,_,_,id_sample=batch
            
            images = [to_pil_image(img.squeeze(0))]
            final_question=question[0]+" Rembember your answers must always be a single word, without explanations, punctuation, or additional text. "
            actual_conversation,text_output,output=model_generation(model,actual_conversation,
                            final_question,processor,images,max_new_tokens=max_new_tokens)
        
              # Extract the hidden states linked with the text output, after the processing of the image (first hidden_state chunk) and before the 
            # end of token message (last hidden_state chunk)
          
                 
            hidden_state_text = [state[0] for state, _ in hooked_res["hidden_states"][1:-1]]
            
            hidden_state_text = torch.cat(hidden_state_text, dim=0)
            
            # Extract the hidden states linked with the image + question input (first hidden_state chunk)
            hidden_state_image = hooked_res["hidden_states"][0][0][0]
            
            indices_image_tags=torch.where(output == image_tag)[0][:576]
            
            # Apply sparse autoencoder to the hidden state
            result_sae_image = sae(hidden_state_image[indices_image_tags].to(sae.device))
            
            result_sae_text = sae(hidden_state_text.to(sae.device))
            actual_conversation.pop()
            
            actual_conversation.pop()

            hooked_res = {"hidden_states": None}

            latent_indices_visual,latent_acts_visual =extract_matching_neuron_values_indices(result_sae_image.latent_indices.to(device),
                                                                                             result_sae_image.latent_acts.to(device),target_tensor) 
            
            latent_indices_text,latent_acts_text =extract_matching_neuron_values_indices(result_sae_text.latent_indices.to(device),
                                                                                         result_sae_text.latent_acts.to(device),target_tensor) 
            
            
            results_dict[id_sample[0]]={
                "visual_features":
                    {"latent_acts":latent_acts_visual,
                    "latent_indices":latent_indices_visual},
                "text_features": 
                    {"latent_acts":latent_acts_text,
                    "latent_indices":latent_indices_text,
                    "final_output":text_output}
                }
    
            
            del result_sae_image,result_sae_text,batch,output,images
            torch.cuda.empty_cache()
            
    hook_gen.remove()

   
    
    with open(folder_save_embedding+"vqa_res_block_"+str(id_loader)+".json", "a") as f:
        json.dump(results_dict, f, indent=4)


    
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
 
    if images is not None:

        
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
        
        with torch.no_grad():
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
        with torch.no_grad():
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

# def initialize_llava_converstation_HONDA(model:LlavaNextForConditionalGeneration,actual_conversation:Dict,processor:LlavaNextProcessor,max_new_tokens=20)->Dict:
#     """
#     Initialize a conversation with the LLaVA model by setting up the system instructions.

#     Args:
#         model (LlavaNextForConditionalGeneration): The LLaVA model instance
#         actual_conversation (Dict): Current conversation state/history
#         processor (LlavaNextProcessor): Processor for handling inputs
#         max_new_tokens (int, optional): Maximum number of tokens to generate. Defaults to 20.

#     Returns:
#         Dict: Updated conversation with system instructions and initial response
#     """    



    
#     prompt = processor.apply_chat_template(actual_conversation, add_generation_prompt=True)

#     inputs = processor( text=prompt, return_tensors="pt").to(model.device)
#     inputs_embeds =  model.get_input_embeddings()(inputs["input_ids"])
    
#     with torch.no_grad():
#         output = model.generate(
#             inputs_embeds=inputs_embeds,
#             max_new_tokens=max_new_tokens
#         )
#      # Extend the converstation with the answer
#     output_final=processor.decode(output[0][-max_new_tokens:], skip_special_tokens=True)
#     tmp_conversation=[
#                 {

#                 "role": "assitant",
#                 "content": [
#                     {"type": "text", "text": output_final},
                    
#                     ],
#                 },
#             ]
#     actual_conversation.extend(tmp_conversation)
    
    
#     function_file=json.load(open("/data/lanza/HONDA/function_structure_1.json"))    
#     function_structure_info="""  Now you will receive structure of the functions that you will use to describe the actions that must be accomplished.
#     Here the format that the function will have:
#     EXAMPLE
#     "id_function": {
#         "name": "exampleFunction",
#         "description": "Function that detect the types of two arguments",
#         "args": [
#             {
#                 "name": "arg1",
#                 "type": "string"
#             },
#             {
#                 "name": "arg2",
#                 "type": "number"
#             }
#         ],
#         "return": {
#             "type": "boolean"
#             "description": "This function will return true if the first argument is a string and the second argument is a number"
#         }
#     }
#     FINISH EXAMPLE
#     HERE THE FUNCTION DEFINTIONS:

# """ 
    
    
#     function_structure_info+=str(function_file)

#     conversation_witout_images=[
#                     {

#                     "role": "user",
#                     "content": [
#                         {"type": "text", "text": function_structure_info},
                        
#                         ],
#                     },
#                 ]
#     actual_conversation.extend(conversation_witout_images)
    
#     prompt = processor.apply_chat_template(actual_conversation, add_generation_prompt=True)

#     inputs = processor( text=prompt, return_tensors="pt").to(model.device)
#     inputs_embeds =  model.get_input_embeddings()(inputs["input_ids"])
#     with torch.no_grad():
#         output = model.generate(
#             inputs_embeds=inputs_embeds,
#             max_new_tokens=max_new_tokens,
#         )
#      # Extend the converstation with the answer
#     output_final=processor.decode(output[0][-max_new_tokens:], skip_special_tokens=True)
#     tmp_conversation=[
#                 {

#                 "role": "assitant",
#                 "content": [
#                     {"type": "text", "text": output_final},
                    
#                     ],
#                 },
#             ]
#     actual_conversation.extend(tmp_conversation)
   
    


#     return actual_conversation

def initialize_llava_vqa(model:LlavaNextForConditionalGeneration,actual_conversation:Dict,processor:LlavaNextProcessor,max_new_tokens=20):
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

   
    prompt = processor.apply_chat_template(actual_conversation, add_generation_prompt=True)

    inputs = processor( text=prompt, return_tensors="pt").to(model.device)
    inputs_embeds =  model.get_input_embeddings()(inputs["input_ids"])
    
    with torch.no_grad():
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
def create_average_activation_dictionary(folder_save_embedding:Path)->None:
    """Create a dictionary mapping neurons to their activation statistics in VQA samples.

    For each analyzed neuron, store information about which VQA samples activated it,
    including the average activation value and count of image patches involved.

    Args:
        folder_save_embedding (Path): Directory containing VQA task results

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
    neuron_activation_stats_dict_image = {str(i): {} for i in range(5000)}
    neuron_activation_stats_dict_text = {str(i): {} for i in range(5000)}
    
    json_files = glob.glob(os.path.join(folder_save_embedding, 'vqa_res_block_*.json'))
    
    for file_path in tqdm(json_files, desc="Average activations of SAE's neurons", leave=True):
    
        res=json.load(open(file_path))
        for key, value in res.items():
            neuron_activation_stats_dict_image=average_values_indices(value['visual_features']['latent_indices'],value['visual_features']['latent_acts'],neuron_activation_stats_dict_image,key)  
            neuron_activation_stats_dict_text=average_values_indices(value['text_features']['latent_indices'],value['text_features']['latent_acts'],neuron_activation_stats_dict_text,key)  


    with open(folder_save_embedding+"average_activation_dictionary_text.json", "a") as f:
        json.dump(neuron_activation_stats_dict_text, f, indent=4)
        
    with open(folder_save_embedding+"average_activation_dictionary_image.json", "a") as f:
        json.dump(neuron_activation_stats_dict_image, f, indent=4)
        
    
@gin.configurable
def create_average_activation_dictionary_llava_next(folder_save_embedding:Path)->None:
    """Create a dictionary mapping neurons to their activation statistics in llava samples.

    For each analyzed neuron, store information about which llava samples activated it,
    including the average activation value and count of image patches involved.

    Args:
        folder_save_embedding (Path): Directory containing llava results

    Returns:
        None. Saves results to 'average_activation_dictionary.json' with structure:
        {
            neuron_id: {
                llava_id: [activation_average, patch_count],
                ...
            },
            ...
        }
    """
    neuron_activation_stats_dict_image = {str(i): {} for i in range(5000)}
    neuron_activation_stats_dict_text = {str(i): {} for i in range(5000)}
    
    json_files = glob.glob(os.path.join(folder_save_embedding, 'llava_15_block_*.json'))
    
    for file_path in tqdm(json_files, desc="Average activations of SAE's neurons", leave=True):
    
        res=json.load(open(file_path))
        for key, value in res.items():
            neuron_activation_stats_dict_image=average_values_indices(value['visual_features']['latent_indices'],value['visual_features']['latent_acts'],neuron_activation_stats_dict_image,key)  
            neuron_activation_stats_dict_text=average_values_indices(value['textual_features']['latent_indices'],value['textual_features']['latent_acts'],neuron_activation_stats_dict_text,key)  


    with open(folder_save_embedding+"average_activation_dictionary_textual.json", "a") as f:
        json.dump(neuron_activation_stats_dict_text, f, indent=4)
        
    with open(folder_save_embedding+"average_activation_dictionary_visual.json", "a") as f:
        json.dump(neuron_activation_stats_dict_image, f, indent=4)
        

def average_values_indices(list_indices: List[List[int]], list_acts:List[List[int]],neuron_activation_stats_dict:dict,id_sample:str)->Dict:

    """Compute the averages activations values for all active neurons for each patch in a single image and adding to 
        the total dictionary

    Args:
        list_indices (List[List[int]]): List with the indices of neurons activate for each patch  
        list_acts (List[List[int]]): List with the activations of neurons activate for each patch 
        neuron_activation_stats_dict (dict): Dictionary 
        id_sample (str): VQA id_sample
    Comments:
        both list have shape(576,[range between 0 to 256])-> num of patches and
        active neurons has intersection between the neurons designed to be analized and the best 256 k highest valued neuron in the patch
    Returns:
        Dictionary updated (see above for the structure)
    """
    # Use numpy arrays for faster computation
    neuron_sums = {}
    neuron_counts = {}
    
    # Process all patches at once
    for patch_indices, patch_acts in zip(list_indices, list_acts):
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
def llava_extract(folder_save_embedding,folder_dataset,id_loader=0,device=torch.device('cuda:0')):  
    """Extract sparse autoencoder features from LLaVA-NeXT model for both images and text.
    This function processes a dataset using the LLaVA-NeXT model and extracts sparse autoencoder
    features from a specific layer (layer 24) for both visual and textual inputs.
     Args:
        folder_save_embedding (str): Path to save the extracted features JSON file
        folder_dataset (str): Path to store/load the LLaVA-NeXT dataset cache
        id_loader (int, optional): Index of the data subset to process. Defaults to 0
        device (torch.device, optional): Device to run computations on. Defaults to cuda:0

      
        None: Results are saved to a JSON file with structure:
                "image_id": {
                    "visual_features": {
                        "latent_acts": tensor or None,
                        "latent_indices": tensor or None
                    "text_features": {
                        "latent_acts": tensor or None, 
                        "latent_indices": tensor or None,
                        "final_output": str or None
    Notes:
        - Processes LLaVA-NeXT-Data dataset (15% training split)
        - Uses 8-bit LLaVA-NeXT model with SDPA attention
        - Extracts features from layer 24 using a pre-trained sparse autoencoder
        - Handles both image and text inputs separately
        - Results are saved incrementally to prevent data loss
    """
    target_tensor = torch.tensor(range(5000), device=device)
    # Calculate the size of each subset
    
    
  
    full_dataset = load_dataset("lmms-lab/LLaVA-NeXT-Data", split="train[:15%]", cache_dir=folder_dataset, num_proc=10)
    
    factor=5
    subset_size = len(full_dataset) // factor
    
    data_subsets = [
        full_dataset.select(range(i * subset_size, (i + 1) * subset_size))
        for i in range(factor)
    ]
    # Use the first subset for this run (you can change the index to use different subsets)
    data = data_subsets[id_loader]
    
    processor = LlavaNextProcessor.from_pretrained("llava-hf/llama3-llava-next-8b-hf")
    model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llama3-llava-next-8b-hf",attn_implementation="sdpa", torch_dtype=torch.float16, device_map="auto",load_in_4bit=True)
    model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
    model.eval()

    
    
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

    sae = SAE.load_from_hub("lmms-lab/llama3-llava-next-8b-hf-sae-131k", hookpoint="model.layers.24").to(device)
    sae.eval()
    results_dict={}
    
    for batch in tqdm(data,desc='Extract from LlaVA'):
        hooked_res = {"hidden_states": None} 
        with torch.no_grad():
            
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
                    
                    
                    output = model(
                        
                            input_ids=inputs["input_ids"].to(model.device),
                            pixel_values=inputs["pixel_values"].to(model.device),
                            image_sizes=inputs["image_sizes"].to(model.device),
                            attention_mask=inputs["attention_mask"].to(model.device)
                    )
                    
                    
                    el=hooked_res["hidden_states"][0]
                    
                    indices_image_tags=torch.where(inputs['input_ids'][0] == image_tag)[0][:576]
                    indices_text_tags=torch.where(inputs['input_ids'][0] != image_tag)[0][5:]
                    result_sae_image = sae(el[0][0][indices_image_tags].to(sae.device))
                    result_sae_text = sae(el[0][0][indices_text_tags].to(sae.device))
                    
                         
                    latent_indices_visual,latent_acts_visual =extract_matching_neuron_values_indices(result_sae_image.latent_indices.to(device),
                                                                                                     result_sae_image.latent_acts.to(device),target_tensor)
                    latent_indices_textual,latent_acts_textual =extract_matching_neuron_values_indices(result_sae_text.latent_indices.to(device),
                                                                                                       result_sae_text.latent_acts.to(device),target_tensor) 
                    
                    
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

            
    with open(folder_save_embedding+"llava_15_block_"+str(id_loader)+".json", "a") as f:
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