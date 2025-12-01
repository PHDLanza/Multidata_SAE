import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["HF_HUB_CACHE"]="/data/lanza/hub"
os.environ["SAE_DISABLE_TRITON"] = "0"
os.environ["TOKENIZERS_PARALLELISM"]="false"
os.environ["PATH"] += os.pathsep + "/sbin/"



from utils.api import coco_extract
import gin
import argparse

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--id_loader','-id', type=int, default=-1, help='Identifier for the data portion to process')
    parser.add_argument('--layer','-l', type=int, default='24', help='Identifier the layer where extract the activations')
    gin.parse_config_file('config_file/config_coco_extract.gin')
    
    args = parser.parse_args()
    for i in range(5):

        gin.bind_parameter('coco_extract.id_loader', i)
        gin.bind_parameter('coco_extract.layer', args.layer)

        print('Id loader ',i)
        coco_extract()
        
    
    # Load the full dataset

