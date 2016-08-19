import time
import numpy as np
import mxnet as mx

from data import Data
from symbol import get_yolo_symbol


CLASSES =  ["aeroplane", "bicycle", "bird", "boat", "bottle",
            "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person",
            "pottedplant", "sheep", "sofa", "train","tvmonitor"]

PRETRAINED = ['models/imagenet', 30]
BATCHSIZE = 32 
EPOCH = 100000
DISPLAY_INTERVAL = 10

LAMBDA_XY = 5.
LAMBDA_HW = 5.
LAMBDA_NOOBJ = .5
np.set_printoptions(precision=2)



def init_yolo():
#     initializer = mx.init.Xavier(factor_type="in", magnitude=2.34)
    initializer = mx.init.Uniform(5e-4)
    pretrained_model = mx.model.FeedForward.load(PRETRAINED[0], PRETRAINED[1], ctx=mx.gpu())
#     args = mx.nd.load('args2.nd')
#     auxs = mx.nd.load('auxs2.nd')
    arg_params = pretrained_model.arg_params
    aux_params = pretrained_model.aux_params
    symbol = get_yolo_symbol()

    arg_shapes, output_shapes, aux_shapes = symbol.infer_shape(data=(BATCHSIZE, 3, 448, 448))
    arg_names = symbol.list_arguments()
    aux_names = symbol.list_auxiliary_states()
    arg_dict = dict(zip(arg_names, [mx.nd.zeros(shape, ctx=mx.gpu()) for shape in arg_shapes]))
    aux_dict = dict(zip(aux_names, [mx.nd.zeros(shape, ctx=mx.gpu()) for shape in aux_shapes]))
    grad_dict = dict(zip(arg_names, [mx.nd.zeros(shape, ctx=mx.gpu()) for shape in arg_shapes]))
    for name in arg_dict:
        if name.endswith('label'):
            continue
        if name.startswith('data'):
            continue
        if name in arg_params and not name.startswith('fullyconnected'):
            arg_params[name].copyto(arg_dict[name])
        else:
            print name
            initializer(name, arg_dict[name])
    for name in aux_dict:
        if 0 and name in aux_params and not name.startswith('fullyconnected'):
            aux_params[name].copyto(aux_dict[name])
        else:
            initializer(name, aux_dict[name])
    executor = symbol.bind(ctx=mx.gpu(), args=arg_dict, args_grad=grad_dict, aux_states=aux_dict, grad_req='write')
#     executor = symbol.bind(ctx=mx.gpu(), args=args, args_grad=grad_dict, aux_states=auxs, grad_req='write')
    return executor


def iou(xyhw, xyxy, gridx, gridy):
    xpmi = gridx*448/7 + xyhw[0] - xyhw[3]*xyhw[3]*224 # xmin pred
    xpma = gridx*448/7 + xyhw[0] + xyhw[3]*xyhw[3]*224
    ypmi = gridy*448/7 + xyhw[1] - xyhw[2]*xyhw[2]*224
    ypma = gridy*448/7 + xyhw[1] + xyhw[2]*xyhw[2]*224
    min_yma = min(ypma, xyxy[3])
    max_ymi = max(ypmi, xyxy[1])
    min_xma = min(xpma, xyxy[2])
    max_xmi = max(xpmi, xyxy[0])
    inter = (min_xma-max_xmi) * (min_yma-max_ymi)
    if min_xma <= max_xmi or min_yma <= max_ymi:
        return 0
    union = (xpma-xpmi)*(ypma-ypmi) + (xyxy[2]-xyxy[0])*(xyxy[3]-xyxy[1])
    if inter > (union-inter):
        print xyxy, [xpmi, ypmi, xpma, ypma]
    return inter / (union-inter)


def xyxy2xyhw(xyxy):
    center = [(xyxy[0]+xyxy[2])/2, (xyxy[1]+xyxy[3])/2]
    gridx = int(center[0]/448*7)
    gridx = min(gridx, 6)
    gridy = int(center[1]/448*7)
    gridy = min(gridy, 6)
    new_x = center[0] - gridx*448/7
    new_y = center[1] - gridy*448/7
    new_x /= 448/7
    new_y /= 448/7
    new_h = (xyxy[3]-xyxy[1]) / 448
    new_w = (xyxy[2]-xyxy[0]) / 448
    return np.array([new_x, new_y, np.sqrt(new_h), np.sqrt(new_w)]), gridx, gridy


def show_loss(out, anno):
    if len(anno) == 0:
        return
    out = np.reshape(out, [7,7,30])
    avg_iou = 0
    avg_conf0 = 0
    avg_conf1 = 0
    avg_acc0 = 0
    avg_acc1 = 0
    for obj in anno:
        xyhw, gridx, gridy = xyxy2xyhw(obj[1])
        bbox0 = out[gridx, gridy, 0:4]
        bbox1 = out[gridx, gridy, 5:9]
        iou0 = iou(bbox0, obj[1], gridx, gridy)
        iou1 = iou(bbox1, obj[1], gridx, gridy)
        avg_iou += max(iou0, iou1)
        if iou0 > iou1:
            avg_conf0 += out[gridx, gridy, 4]
        else:
            avg_conf0 += out[gridx, gridy, 9]
        avg_acc0 += out[gridx, gridy, 10+obj[0]]
        avg_conf1 += out[:,:,4].mean()/2
        avg_conf1 += out[:,:,9].mean()/2
        avg_acc1 += out[:,:,10:].mean()
    avg_iou /= len(anno)
    avg_conf0 /= len(anno)
    avg_conf1 /= len(anno)
    avg_acc0 /= len(anno)
    avg_acc1 /= len(anno)
    print "avg iou: %f, pos conf: %f, avg conf: %f, pos class: %f, avg class: %f" % (avg_iou, avg_conf0, avg_conf1, avg_acc0, avg_acc1)



def get_loss(out, anno):
    out = np.reshape(out, [7,7,30])
    loss = 0
    has_obj = np.zeros([7,7,2])
    xy_loss = 0
    hw_loss = 0
    conf_loss = 0
    class_loss = 0
    noobj_loss = 0
    for obj in anno:
        xyhw, gridx, gridy = xyxy2xyhw(obj[1])
        bbox0 = out[gridx, gridy, 0:4]
        bbox1 = out[gridx, gridy, 5:9]
        iou0 = iou(bbox0, obj[1], gridx, gridy)
        iou1 = iou(bbox1, obj[1], gridx, gridy)
        if iou0 > iou1:
            has_obj[gridx, gridy, 0] = 1
            xy_loss += LAMBDA_XY * np.square(out[gridx,gridy,0:2]-xyhw[:2]).sum()
            hw_loss += LAMBDA_HW * np.square(out[gridx,gridy,2:4]-xyhw[2:]).sum()
            conf_loss += np.square(out[gridx,gridy,4]-iou0)
        else:
            has_obj[gridx, gridy, 1] = 1
            xy_loss += LAMBDA_XY * np.square(out[gridx,gridy,5:7]-xyhw[:2]).sum()
            hw_loss += LAMBDA_HW * np.square(out[gridx,gridy,7:9]-xyhw[2:]).sum()
            conf_loss += np.square(out[gridx,gridy,9]-iou1)
#             print [iou1, out[gridx,gridy,9]],
        for c in range(20):
            if c != obj[0]:
                class_loss += np.square(out[gridx,gridy,c+10])
            else:
                class_loss += np.square(out[gridx,gridy,c+10]-1)
    for x in range(7):
        for y in range(7):
            if not has_obj[x,y,0]:
                noobj_loss += LAMBDA_NOOBJ * np.square(out[x,y,4])
            if not has_obj[x,y,1]:
                noobj_loss += LAMBDA_NOOBJ * np.square(out[x,y,9])
    return np.array([xy_loss, hw_loss, conf_loss, class_loss, noobj_loss])


def get_grad(out, anno):
    out = np.reshape(out, [7,7,30])
    grad = np.zeros([7,7,30])
    has_obj = np.zeros([7,7,2])
    for obj in anno:
        xyhw, gridx, gridy = xyxy2xyhw(obj[1])
        bbox0 = out[gridx, gridy, 0:4]
        bbox1 = out[gridx, gridy, 5:9]
        iou0 = iou(bbox0, obj[1], gridx, gridy)
        iou1 = iou(bbox1, obj[1], gridx, gridy)
        if iou0 > iou1:
            has_obj[gridx, gridy, 0] = 1
            grad[gridx,gridy,0:2] += 2*LAMBDA_XY * (out[gridx,gridy,0:2]-xyhw[:2])
            grad[gridx,gridy,2:4] += 2*LAMBDA_HW * (out[gridx,gridy,2:4]-xyhw[2:])
            grad[gridx,gridy,4] += 2 * (out[gridx,gridy,4]-iou0)
        else:
            has_obj[gridx, gridy, 1] = 1
            grad[gridx,gridy,5:7] += 2*LAMBDA_XY * (out[gridx,gridy,5:7]-xyhw[:2])
            grad[gridx,gridy,7:9] += 2*LAMBDA_HW * (out[gridx,gridy,7:9]-xyhw[2:])
            grad[gridx,gridy,9] += 2 * (out[gridx,gridy,9]-iou1)
        for c in range(20):
            if c != obj[0]:
                grad[gridx,gridy,c+10] += 2 * out[gridx,gridy,c+10]
            else:
                grad[gridx,gridy,c+10] += 2 * (out[gridx,gridy,c+10]-1)
    for x in range(7):
        for y in range(7):
            if not has_obj[x,y,0]:
                grad[x,y,4] += 2 * LAMBDA_NOOBJ * out[x,y,4]
            if not has_obj[x,y,1]:
                grad[x,y,9] += 2 * LAMBDA_NOOBJ * out[x,y,9]
    return grad.flatten()


def train_yolo():
    data = Data(BATCHSIZE)
    executor = init_yolo()
    optimizer = mx.optimizer.SGD(
            learning_rate=1e-5,
            momentum=0.9,
            wd=5e-4)
    states = {}
    needed = set() 
    for idx, k in enumerate(executor.arg_dict.keys()):
        if 1 or k.startswith('fullyconnected') or k.startswith('convolution20'):
            needed.add(idx)
    for idx, w in enumerate(executor.arg_arrays):
        if idx in needed:
            states[idx] = optimizer.create_state(idx, w)
    grad = mx.nd.empty([BATCHSIZE, 7*7*30], ctx=mx.gpu())
#     grad_arrays = {} 
#     for idx in needed:
#         grad_arrays[idx] = mx.nd.zeros(executor.grad_arrays[idx].shape, ctx=mx.gpu())
    for epoch in range(EPOCH):
        if epoch == 200:
            optimizer.lr /= 10
#         if epoch == 1:
#             optimizer.lr *= 10
#         if epoch % 50 == 49:
#             optimizer.lr *= 3
        interval = 0
        loss = np.array([0,0,0,0,0], dtype=np.float64)
        print 'EPOCH: %d'%epoch
        timer = time.clock()
        batch_idx = 0
        total_loss = np.array([0,0,0,0,0], dtype=np.float64)
        while True:
            d = data.provide_batch()
            if d == 'new epoch':
                break
            d[0].copyto(executor.arg_dict['data'])
            executor.forward(is_train=True)
            out = executor.outputs[0].asnumpy()
            grad_np = np.zeros([BATCHSIZE,7*7*30])
            i = 0
            for single_out, single_anno in zip(out, d[1]):
                grad_np[i] = get_grad(single_out, single_anno)
                i += 1
                l = get_loss(single_out, single_anno)
                loss += l
                total_loss += l
            show_loss(single_out, single_anno)
            interval += 1
            if interval == DISPLAY_INTERVAL:
                interval = 0
                loss /= DISPLAY_INTERVAL*BATCHSIZE
                print '\t Batch[%d]: Loss %s, %f' % (batch_idx+1, loss, loss.sum()),
                print out.max(), out.min(), out.mean()
                loss = np.array([0,0,0,0,0], dtype=np.float64) 
            grad_np /= BATCHSIZE
            grad[:] = grad_np
            executor.backward(grad)
#             for idx in needed:
#                 grad_arrays[idx] += executor.grad_arrays[idx]
#             if interval % 8 == 7:
            for idx, w in enumerate(executor.arg_arrays):
                if idx in needed:
                    optimizer.update(idx, w, executor.grad_arrays[idx], states[idx])
#                 for idx in needed:
#                     grad_arrays[idx] *= 0
            batch_idx += 1
        t = time.clock() - timer
        print 'TIME ELAPSED: %fs, %.2f SAMPLES PER SECOND' % (t, batch_idx*BATCHSIZE/t)
        print 'TOTAL LOSS: %s,%f' % (str(total_loss/BATCHSIZE/batch_idx), total_loss.sum()/BATCHSIZE/batch_idx)
        if epoch % 10 == 0:
            mx.nd.save('args2.nd', executor.arg_dict)
            mx.nd.save('auxs2.nd', executor.aux_dict)

#         batch_idx = 0
#         total_loss = np.array([0,0,0,0], dtype=np.float64)
#         while True:
#             d = data.provide_validation()
#             if d == 'new epoch':
#                 break
#             d[0].copyto(executor.arg_dict['data'])
#             executor.forward()
#             out = executor.outputs[0].asnumpy()
#             for single_out, single_anno in zip(out, d[1]):
#                 to_p = np.reshape(single_out, [7,7,30])
#                 l = get_loss(single_out, single_anno)
#                 total_loss += l
#             batch_idx += 1
#         print 'VALIDATION LOSS: %s,%f' % (str(total_loss/BATCHSIZE/batch_idx), total_loss.sum()/BATCHSIZE/batch_idx)
#             
train_yolo()
