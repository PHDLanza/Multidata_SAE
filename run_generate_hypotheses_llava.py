import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["HF_HUB_CACHE"]="/data/lanza/hub"
os.environ["SAE_DISABLE_TRITON"] = "0"
os.environ['TOKENIZERS_PARALLELISM']="false"
os.environ["PATH"] += os.pathsep + "/sbin/"


from utils.utils_generation import generate_textual_hypotheses_llava ,generate_visual_hypotheses_llava
import argparse
import gin
import torch



if __name__=='__main__':


    gin.parse_config_file('config_file/config_generate_hypotheses_llava.gin')
    parser = argparse.ArgumentParser()
    parser.add_argument('--id_loader','-id' ,type=int, default=0, help='Identifier for the data portion to process')
    parser.add_argument('--modality','-m' ,type=str, default='visual', help='Modality to evaluate: visual or textual. Default visual')
    
    
    args = parser.parse_args()
    

    gin.bind_parameter('generate_textual_hypotheses_llava.id_loader', args.id_loader)
    
    gin.bind_parameter('generate_visual_hypotheses_llava.id_loader', args.id_loader)
    
    

    if args.modality == 'visual':
        generate_visual_hypotheses_llava()
        
    elif args.modality == 'textual':
        generate_textual_hypotheses_llava()
        
  
    else:
        raise ValueError(f"Unknown modality: {args.modality}. Choose from 'visual' or 'textual' ")



    # # Optionally save the combined data to a new file

    