import logging
import math
import sys
import time
import pdb
sys.path.append('/home/wcj/mxnet/python')
import mxnet as mx
import numpy

class HuberLoss(mx.operator.CustomOp):
    def forward(self, is_train, req, in_data, out_data, aux):
        #pdb.set_trace()
        #print '\n\n\n\n\n--------------------------------------------------------------------------------'
        #print 'HuberLoss----forward'
        #print 'in_data: ', in_data
        #print 'in_data[0].shape: ', in_data[0].shape
        #print 'in_data[1].shape: ', in_data[1].shape
        pred_bx = in_data[0][0][0].copy()
        #print 'pred_bx: ', pred_bx
        self.assign(out_data[0][0][0], req[0], pred_bx)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        eps = 1e-06
        sigma = 0.0011
        data_in = in_data[0][0][0]
        label_sc = in_data[1][0][0]
        label_bx = in_data[1][0][1]
        #print '\n\n\n\n----------------------------------------------------------------------------------'
        #print 'HuberLoss-----------------backward'
        #print '--------------------------data '
        #print 'data_in:', numpy.unique(data_in.asnumpy())
        #print 'label_sc:', numpy.unique(label_sc.asnumpy())
        #print 'label_bx:', numpy.unique(label_bx.asnumpy())
        pred_ = data_in.copy()
        res_err = label_bx-data_in
        res_err_more_pos_sigma = res_err>sigma
        res_err_less_neg_sigma = res_err<(-1*sigma)
        res_err_between_posneg_sigma = mx.nd.abs(res_err)<=sigma
        grad_loss = -1*res_err_more_pos_sigma*sigma+res_err_less_neg_sigma*sigma+(-1)*res_err*res_err_between_posneg_sigma
        #print 'grad_loss: ', grad_loss
        #print 'grad_loss: ', numpy.unique(grad_loss.asnumpy())
        #print 'in_grad: ', in_grad
        self.assign(in_grad[0][0][0], req[0], grad_loss * label_sc)

@mx.operator.register('HuberLoss')
class HuberLossProp(mx.operator.CustomOpProp):

    def __init__(self):
        super(HuberLossProp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['data', 'label']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        label_shape = in_shape[1]
        output_shape = in_shape[0]
        return ([data_shape, label_shape], [output_shape], [])

    def infer_type(self, in_type):
        return (in_type, [in_type[0]], [in_type[0]])

    def create_operator(self, ctx, shapes, dtypes):
        return HuberLoss()
