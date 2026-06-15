import gin
from utils.utils_evaluation import detection_score_llava, fuz_score_llava
from utils.utils_evaluation import detection_score_coco, fuz_score_coco

if __name__ == "__main__":
 
    gin.parse_config_file('config_file/config_autointerpretability.gin')

    detection_score_llava()
    fuz_score_llava()


        
    

    
