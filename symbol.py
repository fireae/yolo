import mxnet as mx

def ConvFactory(data, num_filter, kernel, stride, pad):
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad)
    bn = mx.sym.BatchNorm(data=conv)
    act = mx.sym.LeakyReLU(data=bn, slope=0.1)
    return act


def get_imagenet_symbol(n_class=1000):
    data = mx.sym.Variable(name='data')
    conv1 = ConvFactory(data, 64, (7,7), (2,2), (3,3))
    pool1 = mx.sym.Pooling(data=conv1, kernel=(2,2), stride=(2,2), pool_type='max')
    conv2 = ConvFactory(pool1, 192, (3,3), (1,1), (1,1))
    pool2 = mx.sym.Pooling(data=conv2, kernel=(2,2), stride=(2,2), pool_type='max')
    conv3_1r = ConvFactory(pool2, 128, (1,1), (1,1), (0,0))
    conv3_1 = ConvFactory(conv3_1r, 256, (3,3), (1,1), (1,1))
    conv3_2r = ConvFactory(conv3_1, 256, (1,1), (1,1), (0,0))
    conv3_2 = ConvFactory(conv3_2r, 512, (3,3), (1,1), (1,1))
    pool3 = mx.sym.Pooling(data=conv3_2, kernel=(2,2), stride=(2,2), pool_type='max')
    conv4_1r = ConvFactory(pool3, 256, (1,1), (1,1), (0,0))
    conv4_1 = ConvFactory(conv4_1r, 512, (3,3), (1,1), (1,1))
    conv4_2r = ConvFactory(conv4_1, 256, (1,1), (1,1), (0,0))
    conv4_2 = ConvFactory(conv4_2r, 512, (3,3), (1,1), (1,1))
    conv4_3r = ConvFactory(conv4_2, 256, (1,1), (1,1), (0,0))
    conv4_3 = ConvFactory(conv4_3r, 512, (3,3), (1,1), (1,1))
    conv4_4r = ConvFactory(conv4_3, 256, (1,1), (1,1), (0,0))
    conv4_4 = ConvFactory(conv4_4r, 512, (3,3), (1,1), (1,1))
    conv4_5r = ConvFactory(conv4_4, 512, (1,1), (1,1), (0,0))
    conv4_5 = ConvFactory(conv4_5r, 1024, (3,3), (1,1), (1,1))
    pool4 = mx.sym.Pooling(data=conv4_5, kernel=(2,2), stride=(2,2), pool_type='max')
    conv5_1r = ConvFactory(pool4, 512, (1,1), (1,1), (0,0))
    conv5_1 = ConvFactory(conv5_1r, 1024, (3,3), (1,1), (1,1))
    conv5_2r = ConvFactory(conv5_1, 512, (1,1), (1,1), (0,0))
    conv5_2 = ConvFactory(conv5_2r, 1024, (3,3), (1,1), (1,1))
    pool5 = mx.sym.Pooling(data=conv5_2, kernel=(7,7), stride=(1,1), pool_type='avg')
    flatten = mx.sym.Flatten(data=pool5)
    fc = mx.sym.FullyConnected(data=flatten, num_hidden=n_class)
    softmax = mx.sym.SoftmaxOutput(data=fc, name='softmax')
    return softmax


def get_yolo_symbol(n_out=7*7*30):
    data = mx.sym.Variable(name='data')
    conv1 = ConvFactory(data, 64, (7,7), (2,2), (3,3))
    pool1 = mx.sym.Pooling(data=conv1, kernel=(2,2), stride=(2,2), pool_type='max')
    conv2 = ConvFactory(pool1, 192, (3,3), (1,1), (1,1))
    pool2 = mx.sym.Pooling(data=conv2, kernel=(2,2), stride=(2,2), pool_type='max')
    conv3_1r = ConvFactory(pool2, 128, (1,1), (1,1), (0,0))
    conv3_1 = ConvFactory(conv3_1r, 256, (3,3), (1,1), (1,1))
    conv3_2r = ConvFactory(conv3_1, 256, (1,1), (1,1), (0,0))
    conv3_2 = ConvFactory(conv3_2r, 512, (3,3), (1,1), (1,1))
    pool3 = mx.sym.Pooling(data=conv3_2, kernel=(2,2), stride=(2,2), pool_type='max')
    conv4_1r = ConvFactory(pool3, 256, (1,1), (1,1), (0,0))
    conv4_1 = ConvFactory(conv4_1r, 512, (3,3), (1,1), (1,1))
    conv4_2r = ConvFactory(conv4_1, 256, (1,1), (1,1), (0,0))
    conv4_2 = ConvFactory(conv4_2r, 512, (3,3), (1,1), (1,1))
    conv4_3r = ConvFactory(conv4_2, 256, (1,1), (1,1), (0,0))
    conv4_3 = ConvFactory(conv4_3r, 512, (3,3), (1,1), (1,1))
    conv4_4r = ConvFactory(conv4_3, 256, (1,1), (1,1), (0,0))
    conv4_4 = ConvFactory(conv4_4r, 512, (3,3), (1,1), (1,1))
    conv4_5r = ConvFactory(conv4_4, 512, (1,1), (1,1), (0,0))
    conv4_5 = ConvFactory(conv4_5r, 1024, (3,3), (1,1), (1,1))
    pool4 = mx.sym.Pooling(data=conv4_5, kernel=(2,2), stride=(2,2), pool_type='max')
    conv5_1r = ConvFactory(pool4, 512, (1,1), (1,1), (0,0))
    conv5_1 = ConvFactory(conv5_1r, 1024, (3,3), (1,1), (1,1))
    conv5_2r = ConvFactory(conv5_1, 512, (1,1), (1,1), (0,0))
    conv5_2 = ConvFactory(conv5_2r, 1024, (3,3), (1,1), (1,1))
    conv5_3 = ConvFactory(conv5_2, 1024, (3,3), (1,1), (1,1))
    conv5_4 = ConvFactory(conv5_3, 1024, (3,3), (2,2), (1,1))
    conv6_1 = ConvFactory(conv5_4, 1024, (3,3), (1,1), (1,1))
    conv6_2 = ConvFactory(conv6_1, 1024, (3,3), (1,1), (1,1))
    flatten = mx.sym.Flatten(data=conv6_2)
    fc1 = mx.sym.FullyConnected(data=flatten, num_hidden=4096)
    act1 = mx.sym.LeakyReLU(data=fc1, slope=0.1)
    dp = mx.sym.Dropout(data=act1, p=0.5)
    fc2 = mx.sym.FullyConnected(data=dp, num_hidden=n_out)
    return fc2 