import cv2
import numpy as np
import mxnet as mx

from symbol import get_yolo_symbol
import data
from matplotlib import pyplot as plt


# im = cv2.imread('sample.jpg')
# im = cv2.resize(im, (448,448))
# im = np.swapaxes(im, 0, 2)
# im = np.swapaxes(im, 1, 2)
# im[0] -= 124
# im[1] -= 117
# im[2] -= 104
# im = np.expand_dims(im, 0)

CLASSES =  ["aeroplane", "bicycle", "bird", "boat", "bottle",
            "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person",
            "pottedplant", "sheep", "sofa", "train","tvmonitor"]



def load():
    d = data.Data(1)
    symbol = get_yolo_symbol()
    args = mx.nd.load('args2.nd')
    auxs = mx.nd.load('auxs2.nd')
    return d, symbol, args, auxs
# args['data'] = mx.nd.array(im, mx.gpu())

def main(data, symbol, args, auxs):
    for i in range(100):
        d = data.provide_validation()
        args['data'] = d[0]

        executor = symbol.bind(ctx=mx.gpu(), args=args, aux_states=auxs, grad_req='null')

        executor.forward(is_train=True)
        x = executor.outputs[0].asnumpy()
        x = np.reshape(x, [7,7,30])
        thres = 0.05
        im = d[0].asnumpy()
        im += 130
        im = np.squeeze(im)
        im = np.swapaxes(im, 1, 2)
        im = np.swapaxes(im, 0, 2)
        im[im>255] = 255
        im[im<0] = 0
        plt.imshow(im.astype(np.uint8))
        for ii in range(7):
            for jj in range(7):
                for kk in range(20):
                    if x[ii,jj,4]*x[ii,jj,kk+10] > thres:
                        xyhw = x[ii,jj,0:4]
                        xc = xyhw[0]*448/7+448/7*ii
                        yc = xyhw[1]*448/7+448/7*jj
                        h = xyhw[2]**2*448
                        w = xyhw[3]**2*448
                        xmin = xc-w/2
                        xmax = xc+w/2
                        ymin = yc-h/2
                        ymax = yc+h/2
                        plt.plot([xmin,xmin], [ymin,ymax], 'b-', lw=3)
                        plt.plot([xmin,xmax], [ymax,ymax], 'b-', lw=3)
                        plt.plot([xmax,xmax], [ymax,ymin], 'b-', lw=3)
                        plt.plot([xmax,xmin], [ymin,ymin], 'b-', lw=3)
                        print ii, jj, CLASSES[kk], max(x[ii,jj,4], x[ii,jj,9])*x[ii,jj,kk+10]
        plt.show()
        print '=='
