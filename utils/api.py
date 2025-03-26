import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["HF_HUB_CACHE"]="/data/lanza/hub"
os.environ["SAE_DISABLE_TRITON"] = "0"
os.environ['TOKENIZERS_PARALLELISM']="false"
os.environ["PATH"] += os.pathsep + "/sbin/"

import torch
import json

from sparsify.sparsify.sparse_coder import SparseCoder as SAE
from typing import Dict, List

from torchvision.transforms.functional import to_pil_image

from torch.utils.data import Subset


from transformers import LlavaNextProcessor ,LlavaNextForConditionalGeneration
from torch.utils.data import DataLoader


from utils.dataset import VQAXTrainDataset
from tqdm import tqdm
import numpy as np
import base64
import cv2
from pathlib import Path
import gin

@gin.configurable
def vqa_extract(train_dataset:VQAXTrainDataset,device:str,path_sae:Path,folder_save_embedding:Path,id_loader=-1, split_num=-1):
    """_summary_

    Args:
        train_dataset (VQAXTrainDataset): _description_
        device (str): _description_
        path_sae (Path): _description_
        folder_save_embedding (Path): _description_
        id_loader (int, optional): _description_. Defaults to -1.
        split_num (int, optional): _description_. Defaults to -1.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """    
    train_dataset=VQAXTrainDataset()
    if id_loader!=-1:
        if split_num == -1:
            raise ValueError("split_num must not be -1")
        dataset_size = len(train_dataset)
        
        split_size = dataset_size // split_num
        start_idx = id_loader * split_size
        end_idx = start_idx + split_size if id_loader < split_num else dataset_size
        train_dataset = Subset(train_dataset, range(start_idx, end_idx))
   
    train_loader=DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=4)



    processor = LlavaNextProcessor.from_pretrained("llava-hf/llama3-llava-next-8b-hf")
    model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llama3-llava-next-8b-hf",attn_implementation='sdpa', torch_dtype=torch.float16, device_map="auto",load_in_4bit=True)
    model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
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
    id_sample=0
    hooked_res = {'hidden_states': None}
    device='cuda:0'
 

    sae = SAE.load_from_hub(path_sae, hookpoint="model.layers.24").to(device)
    sae.eval()

    def forward_hook(model, input, output):
        if hooked_res['hidden_states'] is not None:
            hooked_res['hidden_states'].append(output)
        else:
            hooked_res['hidden_states'] = [output]
        return output


    # Register the hook
    hook_gen=model.language_model.model.layers[24].register_forward_hook(forward_hook)
    results_dict = {
    }
    with torch.no_grad():
        for batch in tqdm(train_loader, desc='Extraction embedding', leave=True):
            img,question,answer,_,_,id_sample=batch
            
            images = [to_pil_image(img.squeeze(0))]
            final_question=question[0]+' Rembember your answers must always be a single word, without explanations, punctuation, or additional text. '
            actual_conversation,text_output,output=model_fake_api(model,actual_conversation,
                            final_question,processor,images,max_new_tokens=10)
        
            # Extract hidden states from the first token of the first batch
            hidden_state = hooked_res['hidden_states'][0][0][0]

            # Apply sparse autoencoder to the hidden state
            result_sae = sae(hidden_state.to(sae.device))
            # output=processor.decode(outputs[0], skip_special_tokens=True)
    
            actual_conversation.pop()
            actual_conversation.pop()

            hooked_res = {'hidden_states': None}
            results_dict[id_sample[0]]={"latent_acts":result_sae.latent_acts[0].tolist(),"latent_indices":result_sae.latent_indices[0].tolist(),"final_output":text_output}
            del result_sae,batch,output,images
            torch.cuda.empty_cache()
    hook_gen.remove()

   

    with open(folder_save_embedding+'vqa_results_block_'+str(id_loader)+'.json', 'a') as f:
        f.write(str(results_dict) + '\n')

def model_fake_api(model:LlavaNextForConditionalGeneration,actual_conversation:Dict,
                   content:str,processor:LlavaNextProcessor,images:List=None,max_new_tokens=20)->tuple[Dict,str]:
    
    """
    Api to generate the next response of the model

    Args:
        model (LlavaLlamaForCausalLM): LLava model to be used.
        actual_conversation (Dict): Conversation dictionary to be updated.
        content (str): Text to be passed to the model.
        images (List, optional): List of images passed to the model. Defaults to 'None'.
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
                temperature=0.0001,
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
        
        inputs_embeds =  model.get_input_embeddings()(inputs['input_ids'])
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

def initialize_llava_converstation(model:LlavaNextForConditionalGeneration,actual_conversation:Dict,processor:LlavaNextProcessor,max_new_tokens=20)->Dict:
    """
    Initialize a conversation with the LLaVA model by setting up the system instructions.

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
    inputs_embeds =  model.get_input_embeddings()(inputs['input_ids'])
    
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
    
    
    function_file=json.load(open('/data/lanza/HONDA/function_structure_1.json'))    
    function_structure_info="""  Now you will receive structure of the functions that you will use to describe the actions that must be accomplished.
    Here the format that the function will have:
    EXAMPLE
    "id_function": {
        "name": "exampleFunction",
        "description": "Function that detect the types of two arguments",
        "args": [
            {
                "name": "arg1",
                "type": "string"
            },
            {
                "name": "arg2",
                "type": "number"
            }
        ],
        "return": {
            "type": "boolean"
            "description": "This function will return true if the first argument is a string and the second argument is a number"
        }
    }
    FINISH EXAMPLE
    HERE THE FUNCTION DEFINTIONS:

""" 
    
    
    function_structure_info+=str(function_file)

    conversation_witout_images=[
                    {

                    "role": "user",
                    "content": [
                        {"type": "text", "text": function_structure_info},
                        
                        ],
                    },
                ]
    actual_conversation.extend(conversation_witout_images)
    
    prompt = processor.apply_chat_template(actual_conversation, add_generation_prompt=True)

    inputs = processor( text=prompt, return_tensors="pt").to(model.device)
    inputs_embeds =  model.get_input_embeddings()(inputs['input_ids'])
    with torch.no_grad():
        output = model.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=max_new_tokens,
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

def initialize_llava_vqa(model:LlavaNextForConditionalGeneration,actual_conversation:Dict,processor:LlavaNextProcessor,max_new_tokens=20):
    """
    Initialize a conversation with the LLaVA model by setting up the system instructions.

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
    inputs_embeds =  model.get_input_embeddings()(inputs['input_ids'])
    
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
def extract_text_image_vectors(LLMfile):
    img_array=[]
    text_array=[]
    img_encoded_array=[]
    for  entry in LLMfile:
        entry_data = entry.get("data", {})
        contents = entry_data["content"] if entry_data.get("content", None) else []
        
            
        for content in contents:
            if isinstance(content, dict) and content.get("type", None) == "text":
                text_array.append(content["text"])

            if isinstance(content, dict) and content.get("type", None) == "image_url":

                image_string = content["image_url"]["url"]
                encoded_data = image_string.split(',')[1]
                nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
                # img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                image = cv2.imdecode(nparr, cv2.COLOR_BGR2RGB)
                img_array.append(image)
                img_encoded_array.append(encoded_data)
                break
    return text_array,img_array,img_encoded_array