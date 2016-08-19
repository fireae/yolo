import os
import mxnet as mx
import numpy as np
import cv2
import xml.etree.ElementTree as ET

from copy import deepcopy


class Data():

    CLASSES =  ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]
    IMAGE_PATH = '/home/zw/dataset/VOC2012/JPEGImages'
    ANNO_PATH = '/home/zw/dataset/VOC2012/Annotations'
    IMAGESET_PATH = '/home/zw/dataset/VOC2012/ImageSets/Main'
    N = 1000000
    Nv = 1000000
    mean_rgb = [124, 117, 104]

    def __init__(self, batchsize):
        self.batchsize = batchsize
        self.get_data()
        self.idx = 0
        self.idxv = 0
        self.order = np.random.permutation(self.N)
        self.orderv = np.random.permutation(self.Nv)
        self.epoch = 0
        self.batch_img = mx.nd.empty((self.batchsize, 3, 448, 448), ctx=mx.gpu())

    def provide_batch(self):
        if self.idx+self.batchsize > self.N:
            self.epoch += 1
            self.idx = 0
            self.idxv = 0
            self.order = np.random.permutation(self.N)
            return 'new epoch'
        else:
            batch_img = np.zeros([self.batchsize, 3, 448, 448], np.float32)
            batch_anno = [[] for i in range(self.batchsize)]
            for i in range(self.batchsize):
                batch_img[i], batch_anno[i] = self.augment_img(self.train_img[self.order[i+self.idx]], self.train_anno[self.order[i+self.idx]])
            batch_img[:,0] -= self.mean_rgb[0]
            batch_img[:,1] -= self.mean_rgb[1]
            batch_img[:,2] -= self.mean_rgb[2]
            self.batch_img[:] = batch_img
            self.idx += self.batchsize
            return [self.batch_img, batch_anno]

    def provide_validation(self):
        if self.idxv+self.batchsize > self.Nv:
            return 'new epoch'
        else:
            batch_img = np.zeros([self.batchsize, 3, 448, 448], np.float32)
            batch_anno = [[] for i in range(self.batchsize)]
            for i in range(self.batchsize):
                batch_img[i], batch_anno[i] = self.augment_img(self.test_img[self.orderv[i+self.idxv]], self.test_anno[self.orderv[i+self.idxv]], nochange=1)
            batch_img[:,0] -= self.mean_rgb[0]
            batch_img[:,1] -= self.mean_rgb[1]
            batch_img[:,2] -= self.mean_rgb[2]
            self.batch_img[:] = batch_img
            self.idxv += self.batchsize
            return [self.batch_img, batch_anno]


    @staticmethod
    def random_crop(img, anno, bd):
        oh, ow, _ = img.shape
        flip = np.random.randint(0,2)
        if flip:
            img[:,:,:] = img[:,::-1,:]
            new_anno = []
            for obj in anno:
                new_obj = []
                new_obj.append(obj[0])
                new_obj.append([ow-obj[1][2], obj[1][1], ow-obj[1][0], obj[1][3]])
                new_anno.append(new_obj)
            anno = new_anno
        bd[:2] *= oh
        bd[2:] *= ow
        bd = bd.astype(int)
        newh = oh+bd[0]+bd[1]
        neww = ow+bd[2]+bd[3]
        new_im = np.zeros([newh,neww,3], np.float32)
        new_im[max(bd[0],0):newh-max(bd[1],0), max(bd[2],0):neww-max(bd[3],0)] = \
                img[-min(bd[0],0):oh+min(bd[1],0), -min(bd[2],0):ow+min(bd[3],0)]
        new_anno = []
        for obj in anno:
            if obj[1][0]+bd[2] > newh or obj[1][1]+bd[0] > neww or obj[1][2]+bd[2] < 0 or obj[1][3]+bd[0] < 0:
                continue
            new_obj = []
            new_obj.append(obj[0])
            new_obj.append([max(0,obj[1][0]+bd[2]), max(0,obj[1][1]+bd[0]), min(neww,obj[1][2]+bd[2]), min(newh,obj[1][3]+bd[0])])
            if new_obj[1][2] < new_obj[1][0]+15 or new_obj[1][3] < new_obj[1][1]+15:
                continue
            new_anno.append(new_obj)
        return new_im, new_anno

    
    def augment_img(self, old_img, anno, nochange=0):
        img = np.zeros_like(old_img).astype(np.float32)
        img[:] = old_img
        if not nochange:
            img[:,:,1] *= np.random.uniform(0.66, 1.5)
            img[:,:,2] *= np.random.uniform(0.66, 1.5)
            bd = np.random.uniform(-0.1, 0.1, [4])
            [img, new_anno] = self.random_crop(img, anno, bd) 
        else:
            new_anno = deepcopy(anno) 
        img[img>255] = 255
        img[img<0] = 0
        img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        old_shape = img.shape
        new_img = cv2.resize(img, (448, 448))
        new_img = np.swapaxes(new_img, 0, 2)
        new_img = np.swapaxes(new_img, 1, 2)
        for obj in new_anno:
            obj[1][0] *= 1.*448/old_shape[1]
            obj[1][1] *= 1.*448/old_shape[0]
            obj[1][2] *= 1.*448/old_shape[1]
            obj[1][3] *= 1.*448/old_shape[0]
            if min(obj[1]) < 0 or max(obj[1]) > 448.01:
                print 'wrongwrongwrong', old_img.shape, img.shape, anno, new_anno, bd
                obj[1][0] = min(448, obj[1][0])
                obj[1][1] = min(448, obj[1][1])
                obj[1][2] = min(448, obj[1][2])
                obj[1][3] = min(448, obj[1][3])
        return new_img, new_anno
        
    def parse_anno(self, xml_file):
        anno = []
        root = ET.parse(xml_file)
        objs = root.findall('object')
        for obj in objs:
            c = self.CLASSES.index(obj[0].text)
            bb = obj.find('bndbox')
            bndbox = map(lambda x:int(x.text), [bb.find('xmin'), bb.find('ymin'), bb.find('xmax'), bb.find('ymax')])
            anno.append([c, bndbox])
        return anno

    def get_data(self):
        with open(os.path.join(self.IMAGESET_PATH, 'train.txt')) as f:
            lines = f.readlines()
        self.train_img = []
        self.train_anno = []
        for line in lines[:self.N]:
            self.train_img.append(cv2.cvtColor(cv2.imread(os.path.join(self.IMAGE_PATH, line.strip()+'.jpg')), cv2.COLOR_BGR2HSV))
            self.train_anno.append(self.parse_anno(os.path.join(self.ANNO_PATH, line.strip()+'.xml')))
#             s = self.train_img[-1].shape
#             for obj in self.train_anno[-1]:
#                 if obj[1][0] > s[0] or obj[1][1] > s[1] or obj[1][2] > s[0] or obj[1][3] > s[1]:
#                     print s, obj
#                     print line
        self.N = len(self.train_img)
        with open(os.path.join(self.IMAGESET_PATH, 'val.txt')) as f:
            lines = f.readlines()
        self.test_img = []
        self.test_anno = []
        for line in lines[:self.Nv]:
            self.test_img.append(cv2.cvtColor(cv2.imread(os.path.join(self.IMAGE_PATH, line.strip()+'.jpg')), cv2.COLOR_BGR2HSV))
            self.test_anno.append(self.parse_anno(os.path.join(self.ANNO_PATH, line.strip()+'.xml')))
#             s = self.test_img[-1].shape
#             for obj in self.test_anno[-1]:
#                 if obj[1][0] > s[0] or obj[1][1] > s[1] or obj[1][2] > s[0] or obj[1][3] > s[1]:
#                     print line
        self.Nv = len(self.test_img)
