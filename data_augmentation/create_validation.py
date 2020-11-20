from os.path import join, exists
from os import listdir, makedirs
import cv2
from shutil import move
import random

species = [
    '004.Groove_billed_Ani',
    '009.Brewer_Blackbird',
    '010.Red_winged_Blackbird',
    '011.Rusty_Blackbird',
    '012.Yellow_headed_Blackbird',
    '013.Bobolink',
    '014.Indigo_Bunting',
    '015.Lazuli_Bunting',
    '016.Painted_Bunting',
    '019.Gray_Catbird',
    '020.Yellow_breasted_Chat',
    '021.Eastern_Towhee',
    '023.Brandt_Cormorant',
    '026.Bronzed_Cowbird',
    '028.Brown_Creeper',
    '029.American_Crow',
    '030.Fish_Crow',
    '031.Black_billed_Cuckoo',
    '033.Yellow_billed_Cuckoo',
    '034.Gray_crowned_Rosy_Finch'
]



def create_validation(data_path = '../aug_bird_dataset/train_images/', val_path = '../aug_bird_dataset/val_images/'):
    
    for bird in species:
        
        dir_path = join(data_path, bird)
        
        train_imgs = listdir(dir_path)
        
        number = len(train_imgs)
        
        validation_sample = random.sample(train_imgs,6)
        
        for img in validation_sample:
            if img == '.ipynb_checkpoints':
                continue
            if not exists(join(val_path,bird)):
                makedirs(join(val_path,bird))
                
            move(join(dir_path,img),
                join(val_path,bird,img))
            

def fusion_train_val(train_path = '../bird_dataset/train_images/', val_path = '../bird_dataset/val_images/', output_path = '../aug_bird_dataset/train_images/' ):
    
    for bird in species:
        
        train_dir = join(train_path, bird)
        val_dir = join(val_path, bird)
        
        train_imgs = listdir(train_dir)
        val_imgs = listdir(val_dir)
        
        dir_obj = join(output_path, bird)
        
        for img in train_imgs:
        
            if img == '.ipynb_checkpoints':
                continue
            img_path = join(train_dir, img)
            image = cv2.imread(img_path) 
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if not exists(dir_obj):
                makedirs(dir_obj)
                
            cv2.imwrite(join(
                    dir_obj,
                    str(img)
                ), image)
            
        for img in val_imgs:
        
            if img == '.ipynb_checkpoints':
                continue
            img_path = join(val_dir, img)
            image = cv2.imread(img_path) 
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if not exists(dir_obj):
                makedirs(dir_obj)
            
            cv2.imwrite(join(
                    dir_obj,
                    str(img)
                ), image)