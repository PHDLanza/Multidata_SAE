from utils.api import llava_extract
import gin


if __name__ == "__main__":
    
   
    gin.parse_config_file('config_file/config_llava_extract.gin')
    


   
    llava_extract()


        
    


