import pickle
import os

from tqdm import tqdm
import tensorflow.keras.backend as K

from utils import get_base_dir, load_clean_descriptions, load_image_names
from utils import max_length, data_generator, build_model

base_dir = get_base_dir()

train_features_path = os.path.join(base_dir, "data/captioning/pickle/encoded_train_images.pkl")
train_features = pickle.load(
    open(train_features_path, "rb"))

print('Photos: train=%d' % len(train_features))

text_file_base_dir = os.path.join(base_dir, "data/captioning/TextFiles/")
filename = os.path.join(text_file_base_dir, 'Flickr_8k.trainImages.txt')
train_image_names = load_image_names(filename)
# descriptions
train_descriptions = load_clean_descriptions('data/captioning/descriptions.txt', train_image_names)
print('Descriptions: train=%d' % len(train_descriptions))

# Create a list of all the training captions
all_train_captions = []
for key, val in train_descriptions.items():
    for cap in val:
        all_train_captions.append(cap)
print(len(all_train_captions))

# Consider only words which occur at least 10 times in the corpus
word_count_threshold = 10
word_counts = {}
nsents = 0
for sent in tqdm(all_train_captions):
    nsents += 1
    for w in sent.split(' '):
        word_counts[w] = word_counts.get(w, 0) + 1

vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
print('preprocessed words %d -> %d' % (len(word_counts), len(vocab)))

ixtoword = {}
wordtoix = {}

ix = 1
for w in vocab:
    wordtoix[w] = ix
    ixtoword[ix] = w
    ix += 1

with open("data/captioning/pickle/vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)

vocab_size = len(ixtoword) + 1  # one for appended 0's
print("vocab size:, ", vocab_size)

# determine the maximum sequence length
max_length = max_length(train_descriptions)
print('Description Length: %d' % max_length)

glove_dir='data/captioning/glove'
embedding_dim = 200
with open("data/captioning/pickle/embedding_matrix.pkl", 'rb') as f:
    embedding_matrix = pickle.load(f)
print("embedding_matrix.shape: ", embedding_matrix.shape)

model = build_model(input_dim=2048, max_length=max_length, vocab_size=vocab_size,
                    embedding_dim=embedding_dim, embedding_matrix=embedding_matrix)

print(model.summary())

epochs = 10
number_pics_per_bath = 6
steps = len(train_descriptions) // number_pics_per_bath

for i in range(epochs):
    print("Epoch :", str(i+1))
    generator = data_generator(descriptions=train_descriptions, photos=train_features,
                               wordtoix=wordtoix,
                               vocab_size=vocab_size,
                               max_length=max_length, num_photos_per_batch=number_pics_per_bath)
    model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
    model.save('models/model_' + str(i+1) + '.h5')

for i in range(epochs):
    generator = data_generator(descriptions=train_descriptions, photos=train_features,
                               wordtoix=wordtoix,
                               vocab_size=vocab_size,
                               max_length=max_length, num_photos_per_batch=number_pics_per_bath)
    model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
    model.save('models/model_' + str(i+11) + '.h5')

K.set_value(model.optimizer.lr, 0.0001)
epochs = 30
number_pics_per_bath = 12
steps = len(train_descriptions) // number_pics_per_bath

for i in range(epochs):
    print("Epoch :", str(i + 21))
    generator = data_generator(descriptions=train_descriptions, photos=train_features,
                               wordtoix=wordtoix,
                               vocab_size=vocab_size,
                               max_length=max_length, num_photos_per_batch=number_pics_per_bath)
    model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
    model.save('models/model_' + str(i + 21) + '.h5')

model.save_weights('models/model_final_weights.h5')