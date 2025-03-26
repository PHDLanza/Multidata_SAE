import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["HF_HUB_CACHE"]="/data/lanza/hub"
os.environ["SAE_DISABLE_TRITON"] = "0"
os.environ['TOKENIZERS_PARALLELISM']="false"
os.environ["PATH"] += os.pathsep + "/sbin/"


import argparse
import gin
from utils.api import vqa_extract


if __name__=='__main__':
    
    gin.parse_config_file('/export/home/lanza/infhome/MyScript/Multidata_SAE/config_file/config_vqa_extract.gin')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--id_loader', type=int, default=-1, help='ID of the test loader to use')
    parser.add_argument('--split_num', type=int, default=-1, help='Num of splits for the dataset')
    
    args = parser.parse_args()
    
    if len(vars(args)) == 0:
        gin.bind_parameter('vqa_extract.id_loader', args.id_loader)
        gin.bind_parameter('vqa_extract.split_num', args.split_num)
        

    vqa_extract()
    