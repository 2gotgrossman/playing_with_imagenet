def test(MODEL_FILE, PRETRAINED_WEIGHTS, output_file_name):

    import top_images
    import numpy as np

    images = top_images.gimme_em()

    # Make sure that caffe is on the python path:
    import sys
    CAFFE_ROOT = '/home/davidg/caffe/' # CHANGE THIS LINE TO YOUR Caffe PATH
    sys.path.insert(0, CAFFE_ROOT + 'python')

    import caffe

    # Set the right path to your model definition file, pretrained model weights,
    # and the image you would like to classify.

    # Use GPU or CPU
    # caffe.set_mode_cpu()
    caffe.set_mode_gpu()
    caffe.set_device(1)



    # Load network
    # Note arguments to preprocess input
    #  mean subtraction switched on by giving a mean array
    #  input channel swapping takes care of mapping RGB into the reference ImageNet model's BGR order
    #  raw scaling multiplies the feature scale from the input [0,1] to the ImageNet model's [0,255]
    net = caffe.Classifier(MODEL_FILE, PRETRAINED_WEIGHTS,
                           mean=np.load(CAFFE_ROOT + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1),
                           channel_swap=(2,1,0),
                           raw_scale=255,
                           image_dims=(256, 256))

    IMAGE_FOLDER_ORIGINAL = '/home/neuroShare/data/ImageNet/ILSVRC2012_data/val/'
    IMAGE_FOLDER_TEXTURIZED = '/home/davidg/style_transfer_project/generated_photos_round_1/'


    # Create Dictionary of true classes
    # Format: dictionary of key: file name string, value: class number integer
    with open('val.txt') as file:
        true_classes = file.read().split('\n')
        true_classes = map(lambda x: (x.split()[0], int(x.split()[1])), true_classes)
        true_classes = dict(true_classes)




    def test_images(folder, all_images):
        """
        Returns Dictionary of classes with a probability aray of prediction
        Example classes_and_predictions['ILSVRC2012_val_00018239.JPEG'] = [.543, .00001, ... ]

        """
        classes_and_predictions = dict(  )

        # Make batches of 100 images (1000/100 batches = 10 batches)
        input_batches = []
        for i in range(0, 1000, 100):
            print('batch %d' % i)

            # batch 1 of images
            current_image_names = all_images[i: i + 100]
            input_images = [ caffe.io.load_image(folder + im) for im in current_image_names ]

            # Classify image
            prediction = net.predict(input_images)  # predict takes any number of images, and formats them for the Caffe net automatically


            for i, image in enumerate(current_image_names):
                classes_and_predictions[image] = prediction[i]
        return  classes_and_predictions


    predictions_natural = test_images(IMAGE_FOLDER_ORIGINAL, images)
    predictions_texturized = test_images(IMAGE_FOLDER_TEXTURIZED, images)

    import pickle
    with open(output_file_name +'.p', 'wb') as f:
        p = {'predictions_natural' : predictions_natural, 'predictions_texturized' : predictions_texturized}
        pickle.dump(p, f)

# if __name__ == '__main__':
#     test(MODEL_FILE='~/caffe/models/bvlc_reference_caffenet/deploy.prototxt',
#          PRETRAINED_WEIGHTS='~/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')
