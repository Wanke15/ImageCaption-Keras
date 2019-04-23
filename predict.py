import os
import pickle

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from utils import build_model


from utils import load_flicker8k_token, get_base_dir, clean_descriptions
from utils import get_all_vocab, save_descriptions
from utils import load_image_names, get_image_paths, load_clean_descriptions, max_length

from extract_image_feature import image_preprocess
from extract_image_feature import encode


base_dir = get_base_dir()

test_features_path = os.path.join(base_dir, "data/captioning/pickle/encoded_test_images.pkl")
test_features = pickle.load(
    open(test_features_path, "rb"))

glove_dir='data/captioning/glove'
embedding_dim = 200
with open("data/captioning/pickle/embedding_matrix.pkl", 'rb') as f:
    embedding_matrix = pickle.load(f)
print("embedding_matrix.shape: ", embedding_matrix.shape)

with open("data/captioning/pickle/vocab.pkl", 'rb') as f:
    vocab = pickle.load(f)
print("Vocab size: ", len(vocab))

ixtoword = {}
wordtoix = {}

ix = 1
for w in vocab:
    wordtoix[w] = ix
    ixtoword[ix] = w
    ix += 1

max_length = 34

model = build_model(input_dim=2048, max_length=max_length, vocab_size=(len(vocab)+1),
                    embedding_dim=embedding_dim, embedding_matrix=embedding_matrix)
model.load_weights("models/model_final_weights.h5")

def greedySearch(photo):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final

print(greedySearch(np.expand_dims(test_features['3458211052_bb73084398.jpg'], axis=0)))
