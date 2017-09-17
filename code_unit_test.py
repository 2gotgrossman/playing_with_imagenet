import numpy as np

# Make sure that caffe is on the python path:
import sys
CAFFE_ROOT = '/home/davidg/caffe/' # CHANGE THIS LINE TO YOUR Caffe PATH
sys.path.insert(0, CAFFE_ROOT + 'python')

import caffe

# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
MODEL_FILE = CAFFE_ROOT + 'models/bvlc_reference_caffenet/deploy.prototxt'
PRETRAINED = CAFFE_ROOT + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel' #Weights to be used w/MODEL_FILE

# Use GPU or CPU
# caffe.set_mode_cpu()
caffe.set_mode_gpu()
caffe.set_device(1)

# Load network
# Note arguments to preprocess input
#  mean subtraction switched on by giving a mean array
#  input channel swapping takes care of mapping RGB into the reference ImageNet model's BGR order
#  raw scaling multiplies the feature scale from the input [0,1] to the ImageNet model's [0,255]
net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                       mean=np.load(CAFFE_ROOT + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1),
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))

IMAGE_FOLDER = '/home/neuroShare/data/ImageNet/ILSVRC2012_data/val/'
import os
all_images = os.listdir(IMAGE_FOLDER)


# Create Dictionary of true classes
# Format: dictionary of key: file name string, value: class number integer
with open('val.txt') as file:
    true_classes = file.read().split('\n')
    true_classes = map(lambda x: (x.split()[0], int(x.split()[1])), true_classes)
    true_classes = dict(true_classes)


# Dictionary of classes with a list of (picture_ID, probability_prediction)
# Example classes_and_predictions[553] = [('ILSVRC2012_val_00018239.JPEG', .53) ... ] 
classes_and_predictions = dict( [ (i, []) for i in range(1000) ] )


# Make batches of 100 images (50000/100 batches = 500 batches)
input_batches = []
for i in range(0, 500, 100):
    print('batch %d' % i)
    
    # batch 1 of images
    current_image_names = all_images[i: i + 100]
    input_images = [ caffe.io.load_image(IMAGE_FOLDER + im) for im in current_image_names ]

    # Classify image
    prediction = net.predict(input_images)  # predict takes any number of images, and formats them for the Caffe net automatically
    
    
    for i, image in enumerate(current_image_names):
        # Get true label of image and prediction for true label
        true_label = true_classes[image]
        prediction_of_true_label = prediction[i][true_label]
        
        classes_and_predictions[true_label].append( (image, prediction_of_true_label) )
        # print ('predicted class for Image %s: '%image, prediction[i].argmax())

import pickle
with open('classes_and_predictions.p', 'wb') as f:
    pickle.dump(classes_and_predictions, f)
