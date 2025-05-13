import os

ROOT_DIR = os.getcwd()
VIT_WEIGHTS_PATH = "/content/drive/MyDrive/model-vit-ang-loss.h5"

SAVE_IMAGE_DIR = os.path.join("/tmp/")

COCO_TRAIN_DIR = "/content/DataOAD/train"
COCO_VALIDATION_DIR = "/content/DataOAD/validation-test"
COCO_VALIDATION_TEST_LABEL_CSV_PATH = "/content/DataOAD/validation-test.csv"
BATCH_SIZE = 16