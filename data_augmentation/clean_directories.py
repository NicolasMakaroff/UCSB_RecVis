from os.path import join, exists
from os import listdir, makedirs, remove, path

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

def clean_dir(data_path = '../aug_bird_dataset/train_images/', val_path = '../aug_bird_dataset/val_images/'):
    for bird in species:
        
        dir_path = join(data_path, bird)
        if not exists(dir_path):
            makedirs(dir_path)
        
        source_img = listdir(dir_path)
        
        if not exists(join(val_path,bird)):
            makedirs(join(val_path,bird))
        
        val_img = listdir(join(val_path,bird))
        
        
        
        for img in source_img:
                
            if path.exists(join(dir_path,img)):
                if not img == '.ipynb_checkpoints':
                    remove(join(dir_path,img))
                    
        for img in val_img:    
            if path.exists(join(val_path,bird,img)):
                if not img == '.ipynb_checkpoints':
                    remove(join(val_path,bird,img))

