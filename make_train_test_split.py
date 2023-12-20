from glob import glob
from os import path
import os
import shutil
import random


IMAGE_PATH = 'data/images'
ANNOTATION_PATH = 'data/annotations'
TRAIN_IMAGE_PATH = 'data/train/images'
TRAIN_ANNOTATION_PATH = 'data/train/annotations'
TEST_IMAGE_PATH = 'data/test/images'
TEST_ANNOTATION_PATH = 'data/test/annotations'
SHAPE_X = 300
SHAPE_Y = 300

def create_train_test_split(test_ratio = 0.2):
    # Clear train folder
    for img, ant in zip(glob(path.join(TRAIN_IMAGE_PATH, '*.png')), glob(path.join(TRAIN_ANNOTATION_PATH, '*.xml'))):
        os.remove(img)
        os.remove(ant)
    # Clear test folder
    for img, ant in zip(glob(path.join(TEST_IMAGE_PATH, '*.png')), glob(path.join(TEST_ANNOTATION_PATH, '*.xml'))):
        os.remove(img)
        os.remove(ant)
    # Repopulate both folders
    for img, ant in zip(glob(path.join(IMAGE_PATH, '*.png')), glob(path.join(ANNOTATION_PATH, '*.xml'))):
        if random.random() < test_ratio:
            shutil.copy(img, TEST_IMAGE_PATH)
            shutil.copy(ant, TEST_ANNOTATION_PATH)
        else:
            shutil.copy(img, TRAIN_IMAGE_PATH)
            shutil.copy(ant, TRAIN_ANNOTATION_PATH)

if __name__ == '__main__':
    create_train_test_split()