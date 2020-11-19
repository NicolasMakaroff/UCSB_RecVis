import os 
import torch
import random 
import numpy as np
import cv2
from os.path import isfile, join, exists
from os import listdir, rename, makedirs
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

from joblib import Parallel, delayed
import multiprocessing

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

class_names = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


# load a model pre-trained pre-trained on COCO


def crop_bird(data_path = '../bird_dataset/train_images/', output_path = '../aug_bird_dataset/train_images/',gpu_available = False):
    
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    if gpu_available:
        model.to('gpu')
    
    model.eval()
    
    
    for bird in tqdm(species):
        
        dir_path = join(data_path,bird)
        source_dir = listdir(dir_path)

        counter = 0
        for img in source_dir:
            if img == '.ipynb_checkpoints':
                continue


            img_path = join(dir_path, img)
            image = cv2.imread(img_path)


            transform = transforms.Compose([transforms.ToTensor()])
            image = transform(image)

            res = model([image])
            pred_score = list(res[0]['scores'].detach().numpy())

            masks = (res[0]['masks']>0.5).squeeze().detach().cpu().numpy()
            pred_class = [class_names[i] for i in list(res[0]['labels'].numpy())]

            pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(res[0]['boxes'].detach().numpy())]

            if not pred_score:
                continue
            pred_t = [pred_score.index(x) for x in pred_score if x>0.5][-1]
            masks = masks[:pred_t+1]
            boxes = pred_boxes[:pred_t+1]
            pred_cls = pred_class[:pred_t+1]

            img = cv2.imread(img_path) 
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            aug_dir_path = join(output_path, bird)

            for i in range(len(masks)):
                if pred_cls[i] == 'bird':

                    x1, y1 = np.floor(boxes[i][0][0]),np.floor(boxes[i][0][1])
                    x2, y2 = np.floor(boxes[i][1][0]), np.floor(boxes[i][1][1])
                    crop = img[int(y1):int(y2),int(x1):int(x2)]

                    if not exists(join(output_path, bird)):
                        makedirs(join(output_path, bird))

                    cv2.imwrite(join(
                    aug_dir_path,
                    bird[4:] +
                    '_crop_' +
                    str(counter)
                    + '.jpg'
                    ),crop)

            counter += 1


        

        

        

        