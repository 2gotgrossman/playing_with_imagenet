import sys
sys.path.insert(0, '/home/davidg/style_transfer_project/deep-learning-models')

from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import top_images
import numpy as np




model = VGG19(weights='imagenet')

img_path = 'elephant.jpg'



images = top_images.gimme_em()

IMAGE_FOLDER_ORIGINAL = '/home/neuroShare/data/ImageNet/ILSVRC2012_data/val/'
IMAGE_FOLDER_TEXTURIZED = '/home/davidg/style_transfer_project/generated_photos_round_1/'



def test_images(folder):
    classes_and_predictions = {}

    for i, im in enumerate(images):

        img_path = folder + im
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        preds = model.predict(x)
        if i % 100 == 0:
            print ('Image #', i)
            print('Predicted:', decode_predictions(preds, top=3)[0])

        classes_and_predictions[im] = preds.tolist()[0] # Unnecessarily nested list
    return classes_and_predictions

predictions_natural = test_images(IMAGE_FOLDER_ORIGINAL)
predictions_texturized = test_images(IMAGE_FOLDER_TEXTURIZED)

import pickle
with open('pickled/compare_predictions_VGG19' +'.p', 'wb') as f:
     p = {'predictions_natural' : predictions_natural, 'predictions_texturized' : predictions_texturized}
     pickle.dump(file=f, obj=p)
