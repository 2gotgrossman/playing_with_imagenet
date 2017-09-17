import sys
sys.path.insert(0, '/home/davidg/style_transfer_project/deep-learning-models')

from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input, decode_predictions
import top_images
import numpy as np
import os



model = InceptionV3(weights='imagenet')

def rotate_image(image, by_x_degrees):
    return image.rotate(by_x_degrees)

def get_image(path_to_image):
    return image.load_img(path_to_image, target_size=(224, 224))


def get_image_prob_vector(image):
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    preds = preds.tolist()[0] # Unnecessarily nested list
    return preds

def get_argsort(vector):
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(vector)), key=vector.__getitem__)

def get_image_top_1(image_vector, image_name, vals_dict):
    """
    returns Boolean: True if true class in Top 1
    """
    correct_index = vals_dict[image_name]
    if get_argsort(image_vector)[-1] == correct_index:
        return True
    else:
        return False
    pass

def get_image_top_5(image_vector, image_name, vals_dict):
    """
    returns Boolean: True if true class in Top 5
    """
    correct_index = vals_dict[image_name]
    if correct_index in get_argsort(image_vector)[-5:]:
        return True
    else:
        return False
    pass

def get_vals_dict(path_to_val_txt):
    with open(path_to_val_txt, 'r') as f:
        vals = f.read().split("\n")

    vals_dict = dict(map(lambda x: x.split(" "), vals ))
    return vals_dict

def test_imagenet_validation_set():
    classes_and_predictions = {}

    IMAGE_FOLDER_ORIGINAL = '/home/neuroShare/data/ImageNet/ILSVRC2012_data/val/'

    vals_dict = get_vals_dict("val.txt")

    images = os.listdir(IMAGE_FOLDER_ORIGINAL)

    arr_lookup =  {'original_top_1' : 0,
                    'original_top_5' : 1,
                    'rotate_90_top_1' : 2,
                    'rotate_90_top_5' : 3,
                    'rotate_180_top_1' : 4,
                    'rotate_180_top_5' : 5,
                    'rotate_270_top_1' : 6,
                    'rotate_270_top_5' : 7 }

    performance = [0]*8


    for i, im in enumerate(images):
        image = get_image(path_to_image=IMAGE_FOLDER_ORIGINAL + im)
        rotations = map(lambda x: rotate_image(image, x), [0, 90, 180, 270])
        predictions = map( lambda x: get_image_prob_vector(x), rotations)

        top_one_predictions = map(lambda x: get_image_top_1(x, im, vals_dict), predictions)
        top_five_predictions = map(lambda x: get_image_top_5(x, im, vals_dict), predictions)


        performance = map(lambda (x,y): x+y, zip(performance, top_five_predictions+top_one_predictions))
        print i

        if i % 100 == 0:
            print performance
            print ('Image #', i)
            print('Predicted:', decode_predictions(preds, top=3)[0])

        print performance

if __name__ == '__main__':
    test_imagenet_validation_set()
