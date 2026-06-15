from utils.utils_generation import generate_visual_hypotheses_coco ,generate_textual_hypotheses_coco
import gin




if __name__=='__main__':


    gin.parse_config_file('config_file/config_generate_hypotheses_coco.gin')


    
    generate_visual_hypotheses_coco()
    generate_textual_hypotheses_coco()
        
  
  


