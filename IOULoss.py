import logging
import math
import sys
import time
import pdb
sys.path.append('/home/wcj/mxnet/python')
import mxnet as mx
import numpy

class IOULoss(mx.operator.CustomOp):
    def forward(self, is_train, req, in_data, out_data, aux):
        data_in = mx.nd.slice_axis(in_data[0], axis=1, begin=0, end=5)
        slda = mx.nd.slice_axis(data_in[0], axis=0, begin=0, end=5)
        data_sc = slda[0]
        data_tp = slda[1]
        data_bt = slda[2]
        data_lf = slda[3]
        data_rt = slda[4]
        pred_sc = mx.nd.divide(1, 1 + mx.nd.exp(-data_sc))
        pred_tp = data_tp
        pred_bt = data_bt
        pred_lf = data_lf
        pred_rt = data_rt
        #print 'pred_sc: ', numpy.unique(pred_sc.asnumpy())
        #print 'pred_tp: ', numpy.unique(pred_tp.asnumpy())
        #print 'pred_bt: ', numpy.unique(pred_bt.asnumpy())
        #print 'pred_lf: ', numpy.unique(pred_lf.asnumpy())
        #print 'pred_rt: ', numpy.unique(pred_rt.asnumpy())
        self.assign(out_data[0][0][0], req[0], pred_sc)
        self.assign(out_data[0][0][1], req[0], pred_tp)
        self.assign(out_data[0][0][2], req[0], pred_bt)
        self.assign(out_data[0][0][3], req[0], pred_lf)
        self.assign(out_data[0][0][4], req[0], pred_rt)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        eps = 1e-06
        data_in = mx.nd.slice_axis(in_data[0], axis=1, begin=0, end=5)
        data = mx.nd.slice_axis(data_in[0], axis=0, begin=0, end=5)
        data_sc = data[0]
        data_tp = data[1]
        data_bt = data[2]
        data_lf = data[3]
        data_rt = data[4]
        #print 'IOULoss--------------------------data: ', numpy.unique(data_sc.asnumpy())
        data_label = mx.nd.slice_axis(in_data[1], axis=1, begin=0, end=5)
        label = mx.nd.slice_axis(data_label[0], axis=0, begin=0, end=5)
        label_sc = label[0]
        label_tp = label[1]
        label_bt = label[2]
        label_lf = label[3]
        label_rt = label[4]
        #print 'IOULoss-------------------------label_sc: ', numpy.unique(label_sc.asnumpy())
        #print 'IOULoss-------------------------label_tp: ', numpy.unique(label_tp.asnumpy())
        #print 'IOULoss-------------------------label_bt: ', numpy.unique(label_bt.asnumpy())
        #print 'IOULoss-------------------------label_lf: ', numpy.unique(label_lf.asnumpy())
        #print 'IOULoss-------------------------label_rt: ', numpy.unique(label_rt.asnumpy())
        pred_sc = mx.nd.divide(mx.nd.ones(data_sc.shape, data_sc.context), mx.nd.ones(data_sc.shape, data_sc.context) + mx.nd.exp(-data_sc))
        grad_sc = ((pred_sc - 1) * label_sc + pred_sc * (1 - label_sc)) / (pred_sc.shape[1] + eps)
        self.assign(in_grad[0][0][0], req[0], grad_sc)
        predBox = (data_tp + data_bt) * (data_lf + data_rt)
        labelBox = (label_tp + label_bt) * (label_lf + label_rt)
        ih = mx.nd.minimum(data_tp, label_tp) + mx.nd.minimum(data_bt, label_bt)
        iw = mx.nd.minimum(data_lf, label_lf) + mx.nd.minimum(data_rt, label_rt)
        I = ih * iw
        U = predBox + labelBox - I
        iou = I * 1.0 / (U + eps)
        loss = -mx.nd.log(iou) * label_sc
        grad_pred_tp = grad_pred_bt = data_lf + data_rt
        grad_pred_lf = grad_pred_rt = data_tp + data_bt
        pred_I_tp = data_tp < label_tp
        grad_I_tp = iw * pred_I_tp
        pred_I_bt = data_bt < label_bt
        grad_I_bt = iw * pred_I_bt
        pred_I_lf = data_lf < label_lf
        grad_I_lf = ih * pred_I_lf
        pred_I_rt = data_rt < label_rt
        grad_I_rt = ih * pred_I_rt
        grad_loss_tp = grad_pred_tp / (mx.nd.full(label_sc.shape, U, label_sc.context) + eps) - mx.nd.full(label_sc.shape, (U + I) / (U * I + eps), label_sc.context) * grad_I_tp
        grad_loss_bt = grad_pred_bt / (mx.nd.full(label_sc.shape, U, label_sc.context) + eps) - mx.nd.full(label_sc.shape, (U + I) / (U * I + eps), label_sc.context) * grad_I_bt
        grad_loss_lf = grad_pred_lf / (mx.nd.full(label_sc.shape, U, label_sc.context) + eps) - mx.nd.full(label_sc.shape, (U + I) / (U * I + eps), label_sc.context) * grad_I_lf
        grad_loss_rt = grad_pred_rt / (mx.nd.full(label_sc.shape, U, label_sc.context) + eps) - mx.nd.full(label_sc.shape, (U + I) / (U * I + eps), label_sc.context) * grad_I_rt
        self.assign(in_grad[0][0][1], req[0], grad_loss_tp * label_sc)
        self.assign(in_grad[0][0][2], req[0], grad_loss_bt * label_sc)
        self.assign(in_grad[0][0][3], req[0], grad_loss_lf * label_sc)
        self.assign(in_grad[0][0][4], req[0], grad_loss_rt * label_sc)

@mx.operator.register('IOULoss')
class IOULossProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(IOULossProp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['data', 'label']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        label_shape = in_shape[0]
        output_shape = in_shape[0]
        return ([data_shape, label_shape], [output_shape], [])

    def infer_type(self, in_type):
        return (in_type, [in_type[0]], [])

    def create_operator(self, ctx, shapes, dtypes):
        return IOULoss()
