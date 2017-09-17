import test

models = ['/home/davidg/caffe/models/bvlc_googlenet/deploy.prototxt']
weights = ['/home/davidg/caffe/models/bvlc_googlenet/bvlc_googlenet.caffemodel']
output = ['compare_predictions_googlenet']

for i in range(len(models)):
    test.test(models[i], weights[i], output[i])
