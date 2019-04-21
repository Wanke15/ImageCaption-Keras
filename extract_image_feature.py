import numpy as np

from tensorflow._api.v1.keras.preprocessing import image
from tensorflow.python.keras.applications.inception_v3 import preprocess_input


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


# Function to encode a given image into a vector of size (2048, )
def encode(base_model, input_image):
    input_image = image_preprocess(input_image)  # preprocess the image
    fea_vec = base_model.predict(
        input_image)  # Get the encoding vector for the image
    fea_vec = np.reshape(
        fea_vec, fea_vec.shape[1])  # reshape from (1, 2048) to (2048, )
    return fea_vec
