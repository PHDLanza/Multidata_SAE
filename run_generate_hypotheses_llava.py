from utils.utils_generation import generate_textual_hypotheses_llava ,generate_visual_hypotheses_llava
import gin

if __name__=='__main__':


    gin.parse_config_file('config_file/config_generate_hypotheses_llava.gin')
   
    generate_visual_hypotheses_llava()
    generate_textual_hypotheses_llava()



    