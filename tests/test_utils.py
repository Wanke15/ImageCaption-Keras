import glob
import os
import pickle
import time

from tensorflow.python.keras import Model
from tensorflow.python.keras.applications import InceptionV3
from tqdm import tqdm

from utils import load_flicker8k_token, get_base_dir, clean_descriptions
from utils import get_all_vocab, save_descriptions
from utils import load_image_names, get_image_paths, load_clean_descriptions

from extract_image_feature import image_preprocess
from extract_image_feature import encode

base_dir = get_base_dir()
text_file_base_dir = os.path.join(base_dir, "data/captioning/TextFiles/")

filename = os.path.join(text_file_base_dir, "Flickr8k.token.txt")

# parse descriptions
descriptions = load_flicker8k_token(filename)

# clean descriptions
cleaned_descriptions = clean_descriptions(descriptions)

# summarize vocabulary
vocabulary = get_all_vocab(descriptions)

save_descriptions(descriptions, 'descriptions.txt')
##################################################################

# load training dataset (6K)
filename = os.path.join(text_file_base_dir, 'Flickr_8k.trainImages.txt')
train = load_image_names(filename)
print('Num of data samples: %d' % len(train))

# Below path contains all the images
images_dir = os.path.join(base_dir, 'data/captioning/Flicker8k_Dataset/')
# Below file conatains the names of images to be used in train data
train_image_txt_file = os.path.join(text_file_base_dir,
                                    'Flickr_8k.trainImages.txt')

train_image_paths = get_image_paths(images_dir, train_image_txt_file)

test_image_txt_file = os.path.join(text_file_base_dir,
                                   'Flickr_8k.testImages.txt')

test_image_paths = get_image_paths(images_dir, test_image_txt_file)

# descriptions
train_descriptions = load_clean_descriptions('descriptions.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))

# Load the inception v3 model
model = InceptionV3(weights='imagenet', include_top=True)
model_new = Model(model.input, model.layers[-2].output)
# model_new = InceptionV3(weights='imagenet', include_top=False)
# Create a new model, by removing the last layer (output layer) from the inception v3
# model_new = Model(model.input, model.layers[-2].output)

# Call the funtion to encode all the train images
# This will take a while on CPU - Execute this only once
start = time.time()
encoding_train = {
    img[len(images_dir):]: encode(model_new, img)
    for img in tqdm(train_image_paths)
}
print("Train feature taken in seconds: ", time.time() - start)

# Save the bottleneck train features to disk
train_feat_path = os.path.join(
    base_dir, "data/captioning/pickle/encoded_train_images.pkl")
with open(train_feat_path, "wb") as encoded_pickle:
    pickle.dump(encoding_train, encoded_pickle)

# Call the funtion to encode all the test images - Execute this only once
start = time.time()
encoding_test = {
    img[len(images_dir):]: encode(model_new, img)
    for img in tqdm(test_image_paths)
}
print("Test feature taken in seconds: ", time.time() - start)

# Save the bottleneck test features to disk
test_feat_path = os.path.join(
    base_dir, "data/captioning/pickle/encoded_test_images.pkl")
with open(test_feat_path, "wb") as encoded_pickle:
    pickle.dump(encoding_test, encoded_pickle)
