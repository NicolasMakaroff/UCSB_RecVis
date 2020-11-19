from imgaug import augmenters as iaa
import os
from os.path import join, exists
import cv2
from tqdm.notebook import tqdm



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

gauss = iaa.AdditiveGaussianNoise(scale = 0.2 * 255)
affine = iaa.Affine(translate_px = {'x': (-50,50), 'y': (-50,50)})
blur = iaa.GaussianBlur(sigma=(3.0))
flip = iaa.Fliplr(1.0)

def data_aug(data_path = '../bird_dataset/train_images', output_path = '../aug_bird_dataset/train_images/', counter=0):
    
    for bird in tqdm(species) :
        
        dir_path = join(data_path, bird)
        source_img = os.listdir(dir_path)
        #print(source_img)
        
        
        counter = counter
        
        for image in tqdm(source_img):
            
            if image == '.ipynb_checkpoints':
                continue
            
            aug_img = []
            image_path = join(dir_path, image)
            
            img = cv2.imread(image_path)
            #print(type(img))
            aug_img.append(img)

            gauss_img = gauss.augment_image(img)

            affine_img = affine.augment_image(img)
            blur_img = blur.augment_image(img)
            flip_img = flip.augment_image(img)

            #second_aug_img = second_aug.augment_images(img)
            
            aug_img.append(gauss_img)
            aug_img.append(sharp_img)

            aug_img.append(blur_img)
            aug_img.append(flip_img)

            
            aug_dir_path = join(output_path, bird)
            
            if not exists(join(output_path, bird)):
                    makedirs(join(output_path, bird))
                    
            for aug_images in aug_img:
                cv2.imwrite(join(
                    aug_dir_path,
                    bird[4:] +
                    '_' +
                    str(counter)
                    + '.jpg'
                ),
                aug_images
                )
                counter += 1
                


    
    
    
