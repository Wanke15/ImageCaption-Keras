import glob
import pickle
import string
import os

import numpy as np
from tensorflow.keras.layers import add, Concatenate

from tensorflow.keras.preprocessing import image
from tensorflow.python.keras.applications.inception_v3 import preprocess_input

from tensorflow.keras import Input, Model
from tensorflow.python.keras.layers import Dropout, Dense, Embedding, LSTM, merge, CuDNNLSTM
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm


def get_base_dir():
    return os.path.dirname(__file__)


def load_flicker8k_token(file_name):
    # read text file
    with open(file_name, 'r') as f:
        lines = f.read()
    img2desc = dict()
    # process lines
    for line in lines.split('\n'):
        # split line by white space
        tokens = line.split()
        if len(line) < 2:
            continue
        # take the first token as the image id, the rest as the description
        image_id, image_desc = tokens[0], tokens[1:]
        # extract filename from image id
        image_id = image_id.split('.')[0]
        # convert description tokens back to string
        image_desc = ' '.join(image_desc)
        # create the list if needed
        if image_id not in img2desc:
            img2desc[image_id] = list()
        # store description
        img2desc[image_id].append(image_desc)
    return img2desc


def clean_descriptions(descs):
    # prepare translation table for removing punctuation
    table = str.maketrans('', '', string.punctuation)
    for key, desc_list in descs.items():
        for i in range(len(desc_list)):
            desc = desc_list[i]
            # tokenize
            desc = desc.split()
            # convert to lower case
            desc = [word.lower() for word in desc]
            # remove punctuation from each token
            desc = [w.translate(table) for w in desc]
            # remove hanging 's' and 'a'
            desc = [word for word in desc if len(word) > 1]
            # remove tokens with numbers in them
            desc = [word for word in desc if word.isalpha()]
            # store as string
            desc_list[i] = ' '.join(desc)
    return descs


# convert the loaded descriptions into a vocabulary of words
def get_all_vocab(descriptions):
    # build a list of all description strings
    vocabs = set()
    for key in descriptions.keys():
        [vocabs.update(d.split()) for d in descriptions[key]]
    return vocabs


# save descriptions to file, one per line
def save_descriptions(descriptions, filename):
    lines = list()
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key + ' ' + desc)
    data = '\n'.join(lines)
    with open(filename, 'w') as f:
        f.write(data)


# load a pre-defined list of photo identifiers
def load_image_names(filename):
    with open(filename, 'r') as f:
        lines = f.read()
    image_name = list()
    # process line by line
    for line in lines.split('\n'):
        # skip empty lines
        if len(line) < 1:
            continue
        # get the image identifier
        identifier = line.split('.')[0]
        image_name.append(identifier)
    return set(image_name)


def get_image_paths(image_dir, text_file):
    all_images = glob.glob(image_dir + '*.jpg')

    # Read the train image names in a set
    image_names = set(open(text_file, 'r').read().strip().split('\n'))

    image_paths = [
        img_path for img_path in all_images
        if img_path[len(image_dir):] in image_names
    ]
    return image_paths


# load clean descriptions into memory
def load_clean_descriptions(filename, dataset):
    # load document
    with open(filename, 'r') as f:
        doc = f.read()
    _descriptions = dict()
    for line in doc.split('\n'):
        # split line by white space
        tokens = line.split()
        # split id from description
        image_id, image_desc = tokens[0], tokens[1:]
        # skip images not in the set
        if image_id in dataset:
            # create list
            if image_id not in _descriptions:
                _descriptions[image_id] = list()
            # wrap description in tokens
            desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
            # store
            _descriptions[image_id].append(desc)
    return _descriptions


def image_preprocess(image_path):
    # Convert all the images to size 299x299 as expected by the inception v3 model
    img = image.load_img(image_path, target_size=(299, 299))
    # Convert PIL image to numpy array of 3-dimensions
    x = image.img_to_array(img)
    # Add one more dimension
    x = np.expand_dims(x, axis=0)
    # preprocess the images using preprocess_input() from inception module
    x = preprocess_input(x)
    return x


# convert a dictionary of clean descriptions to a list of descriptions
def to_lines(descriptions):
    all_desc = list()
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc


# calculate the length of the description with the most words
def max_length(descriptions):
    lines = to_lines(descriptions)
    return max(len(d.split()) for d in lines)


# data generator, intended to be used in a call to model.fit_generator()
def data_generator(descriptions, photos, wordtoix, max_length, vocab_size,
                   num_photos_per_batch):
    X1, X2, y = list(), list(), list()
    n = 0
    # loop for ever over images
    while 1:
        for key, desc_list in descriptions.items():
            n += 1
            # retrieve the photo feature
            photo = photos[key + '.jpg']
            for desc in desc_list:
                # encode the sequence
                seq = [
                    wordtoix[word] for word in desc.split(' ')
                    if word in wordtoix
                ]
                # split one sequence into multiple X, y pairs
                for i in range(1, len(seq)):
                    # split into input and output pair
                    in_seq, out_seq = seq[:i], seq[i]
                    # pad input sequence
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    # encode output sequence
                    out_seq = to_categorical([out_seq],
                                             num_classes=vocab_size)[0]
                    # store
                    X1.append(photo)
                    X2.append(in_seq)
                    y.append(out_seq)
            # yield the batch data
            if n == num_photos_per_batch:
                yield [[np.array(X1), np.array(X2)], np.array(y)]
                X1, X2, y = list(), list(), list()
                n = 0


def build_model(input_dim, max_length, vocab_size, embedding_dim,
                embedding_matrix):
    inputs1 = Input(shape=(input_dim, ))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    inputs2 = Input(shape=(max_length, ))
    se1 = Embedding(vocab_size, embedding_dim, mask_zero=False)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    # decoder1 = Concatenate()([fe2, se3])
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    # attention_probs = Dense(256, activation='softmax', name='attention_probs')(decoder2)
    # attention_mul = merge.Multiply()([decoder1, attention_probs])
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)

    model.layers[2].set_weights([embedding_matrix])
    model.layers[2].trainable = False

    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

def save_embedding_matrix(glove_dir, vocab_size, wordtoix, embedding_dim = 200):
    # glove_dir='data/captioning/glove'
    # Load Glove vectors
    embeddings_index = {}  # empty dictionary
    with open(os.path.join(glove_dir, 'glove.6B.200d.txt'), encoding="utf-8") as f:
        for line in tqdm(f.readlines()):
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    print('Found %s word vectors.' % len(embeddings_index))

    # Get 200-dim dense vector for each of the 10000 words in out vocabulary
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    for word, i in wordtoix.items():
        # if i < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Words not found in the embedding index will be all zeros
            embedding_matrix[i] = embedding_vector
    with open("data/captioning/pickle/embedding_matrix.pkl", 'wb') as f:
        pickle.dump(embedding_matrix, f)
