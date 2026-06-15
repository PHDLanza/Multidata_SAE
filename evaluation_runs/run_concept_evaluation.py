import gin
from utils.utils_evaluation import eval_visual_hypotheses_CLIP,eval_visual_hypotheses_ALIGN
from utils.utils_evaluation import eval_textual_hypotheses_CLIP,eval_textual_hypotheses_ALIGN

if __name__=='__main__':
    
    gin.parse_config_file('config_file/config_evaluation.gin')
    

    eval_visual_hypotheses_CLIP()
    eval_visual_hypotheses_ALIGN()
    
    eval_textual_hypotheses_CLIP()
    eval_textual_hypotheses_ALIGN()
        
    

