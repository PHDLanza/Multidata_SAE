from utils.api import coco_extract
import gin
import argparse

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    gin.parse_config_file('config_file/config_coco_extract.gin')
    
    coco_extract()
        
    
    

