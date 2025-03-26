

from torch.utils.data import Dataset
from PIL import Image
import json
import torchvision.transforms as transforms 
import gin


    
    
@gin.configurable
class VQAXTrainDataset(Dataset):


    def __init__(self, path_labels,path_folder):
        self.transform = transforms.Compose([
            # you can add other transformations in this list
            transforms.Resize((336,336)),
            transforms.ToTensor(),
          
        ])
        self.data = json.load(open(path_labels, 'r'))
        self.ids_list = list(self.data.keys())
        self.path_folder=path_folder
        for k,v in self.data.items():   
            if len(v['explanation']) > 1:   # some questions have more than one explanation
                # duplicate them for loading. -1 because one explanation is already in ids_list
                self.ids_list += [str(k)] * (len(v['explanation']) - 1)    

        self.index_tracker = {k: len(v['explanation']) - 1 for k,v in self.data.items()}
        


        
    def __getitem__(self, i):
        
        quention_id = self.ids_list[i]
        sample = self.data[quention_id]
        img_name = sample['image_name']
        question_text = sample['question']    
        answer_text = sample['answers'][0]['answer'] 

        exp_idx = self.index_tracker[quention_id]    # the index of the explanation for questions with multiple explanations
        if exp_idx > 0:
            self.index_tracker[quention_id] -= 1    # decrease usage
                
        explanation_text = sample['explanation'][exp_idx]   # explanation
        
    
        
        folder = self.path_folder+'train2014/' if 'train' in img_name else self.path_folder+'val2014/' 
        img_path = folder + img_name
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        
        complete = question_text + ' The answer is ' + answer_text + ' because ' + explanation_text
        # complete ='Filler'

        return (img, question_text, answer_text, explanation_text,complete,quention_id)

    def __len__(self):
        return len(self.ids_list)
