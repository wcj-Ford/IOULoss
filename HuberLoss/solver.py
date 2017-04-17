# pylint: skip-file
import numpy as np
import mxnet as mx
import time
import pdb
import logging
import HuberLoss
import MultiAcc
from collections import namedtuple
from mxnet import optimizer as opt
from mxnet.optimizer import get_updater
from mxnet import metric

# Parameter to pass to batch_end_callback
BatchEndParam = namedtuple('BatchEndParams', ['epoch', 'nbatch', 'eval_metric'])
class Solver(object):
    def __init__(self, symbol, ctx=None,
                 begin_epoch=0, num_epoch=None,
                 arg_params=None, aux_params=None,
                 optimizer='sgd', **kwargs):
        self.symbol = symbol
        if ctx is None:
            ctx = mx.cpu()
        self.ctx = ctx
        self.begin_epoch = begin_epoch
        self.num_epoch = num_epoch
        self.arg_params = arg_params
        self.aux_params = aux_params
        self.optimizer = optimizer
        self.kwargs = kwargs.copy()

    def fit(self, train_data, eval_data=None,
            eval_metric='acc',
            grad_req='write',
            epoch_end_callback=None,
            batch_end_callback=None,
            kvstore='local',
            logger=None):
        if logger is None:
            logger = logging
        logging.info('Start training with %s', str(self.ctx))
        d_sh = train_data.provide_data[0][1]
        arg_shapes, out_shapes, aux_shapes = self.symbol.infer_shape(data=train_data.provide_data[0][1], score_label=(1,1,d_sh[2],d_sh[3]), top_label=(1,1,d_sh[2],d_sh[3]), bottom_label=(1,1,d_sh[2],d_sh[3]), left_label=(1,1,d_sh[2],d_sh[3]), right_label=(1,1,d_sh[2],d_sh[3]))
        arg_names = self.symbol.list_arguments()
        if grad_req != 'null':
            self.grad_params = {}
            for name, shape in zip(arg_names, arg_shapes):
                if not (name.endswith('data') or name.endswith('label')):
                    self.grad_params[name] = mx.nd.zeros(shape, self.ctx)
        else:
            self.grad_params = None
        aux_names = self.symbol.list_auxiliary_states()
        self.aux_params = {k : mx.nd.zeros(s) for k, s in zip(aux_names, aux_shapes)}
        data_name = train_data.data_name
        label_sc = train_data.label_score
        label_tp = train_data.label_top
        label_bt = train_data.label_bottom
        label_lf = train_data.label_left
        label_rt = train_data.label_right
        self.optimizer = opt.create(self.optimizer, rescale_grad=(1.0/train_data.get_batch_size()), **(self.kwargs))
        self.updater = get_updater(self.optimizer)
        eval_metric_MuLb = MultiAcc.Multi_Accuracy()
        # begin training
        auxparams = self.aux_params.copy()
        for k, v in self.aux_params.items():
            auxparams[k] = mx.nd.zeros(v.shape, self.ctx)
            v.copyto(auxparams[k])
        for epoch in range(self.begin_epoch, self.num_epoch):
            if epoch%15==0 and epoch>0:
                self.optimizer.lr = self.optimizer.lr*0.1
            nbatch = 0
            train_data.reset()
            for data in train_data:
                #break
                #pdb.set_trace()
                nbatch += 1
                label_shape = data[label_sc].shape
                self.arg_params[data_name] = mx.nd.array(data[data_name], self.ctx)
                self.arg_params[label_sc] = mx.nd.array(data[label_sc].reshape(label_shape[0], label_shape[1], label_shape[2], label_shape[3]), self.ctx)
                self.arg_params[label_tp] = mx.nd.array(data[label_tp].reshape(label_shape[0], label_shape[1], label_shape[2], label_shape[3]), self.ctx)
                self.arg_params[label_bt] = mx.nd.array(data[label_bt].reshape(label_shape[0], label_shape[1], label_shape[2], label_shape[3]), self.ctx)
                self.arg_params[label_lf] = mx.nd.array(data[label_lf].reshape(label_shape[0], label_shape[1], label_shape[2], label_shape[3]), self.ctx)
                self.arg_params[label_rt] = mx.nd.array(data[label_rt].reshape(label_shape[0], label_shape[1], label_shape[2], label_shape[3]), self.ctx)
                output_names = self.symbol.list_outputs()
                self.exector = self.symbol.bind(self.ctx, self.arg_params,
                                args_grad=self.grad_params,
                                grad_req=grad_req,
                                aux_states=auxparams)
                                #aux_states=self.aux_params)
                assert len(self.symbol.list_arguments()) == len(self.exector.grad_arrays)
                update_dict = {name: nd for name, nd in zip(self.symbol.list_arguments(), self.exector.grad_arrays) if nd}
                output_dict = {}
                output_buff = {}
                for key, arr in zip(self.symbol.list_outputs(), self.exector.outputs):
                    output_dict[key] = arr
                    output_buff[key] = mx.nd.empty(arr.shape, ctx=mx.cpu())
                self.exector.forward(is_train=True, data=self.arg_params[data_name], score_label=self.arg_params[label_sc], top_label=self.arg_params[label_tp], bottom_label=self.arg_params[label_bt], left_label=self.arg_params[label_lf], right_label=self.arg_params[label_rt])
                for key in output_dict:
                    output_dict[key].copyto(output_buff[key])
                self.exector.backward()
                for key, arr in update_dict.items():
                    if key != "bigscore_weight":
                        self.updater(key, arr, self.arg_params[key])
                pred_shape = self.exector.outputs[0].shape
                #print self.exector.outputs
                #pdb.set_trace()
                #print 'self.exector.outputs: '
                #print '--------------------------------------------------------------------self.exector.outputs: '
                #print 'outputs[0]: ', np.unique(self.exector.outputs[0].asnumpy())
                #print 'outputs[1]: ', np.unique(self.exector.outputs[1].asnumpy())
                #print 'outputs[2]: ', np.unique(self.exector.outputs[2].asnumpy())
                #print 'outputs[3]: ', np.unique(self.exector.outputs[3].asnumpy())
                #print 'outputs[4]: ', np.unique(self.exector.outputs[4].asnumpy())
                output_sc = output_buff['loss_sc_output']
                output_tp = output_buff['loss_tp_output']
                output_bt = output_buff['loss_bt_output']
                output_lf = output_buff['loss_lf_output']
                output_rt = output_buff['loss_rt_output']
                sc_label = mx.nd.array(data[label_sc].reshape(label_shape[2],label_shape[3]))
                pred_sc = mx.nd.array(output_sc[0][0].asnumpy().reshape(pred_shape[2],pred_shape[3]))
                tp_label = mx.nd.array(data[label_tp].reshape(label_shape[2],label_shape[3]))
                pred_tp = mx.nd.array(output_tp[0][0].asnumpy().reshape(pred_shape[2],pred_shape[3]))
                bt_label = mx.nd.array(data[label_bt].reshape(label_shape[2],label_shape[3]))
                pred_bt = mx.nd.array(output_bt[0][0].asnumpy().reshape(pred_shape[2],pred_shape[3]))
                lf_label = mx.nd.array(data[label_lf].reshape(label_shape[2],label_shape[3]))
                pred_lf = mx.nd.array(output_lf[0][0].asnumpy().reshape(pred_shape[2],pred_shape[3]))
                rt_label = mx.nd.array(data[label_rt].reshape(label_shape[2],label_shape[3]))
                pred_rt = mx.nd.array(output_rt[0][0].asnumpy().reshape(pred_shape[2],pred_shape[3]))
                #print '----------------------------------------------------network output:'
                #print 'sc_label: ', np.unique(sc_label.asnumpy())
                #print 'pred_sc: ',pred_sc.shape,  np.unique(pred_sc.asnumpy())
                #print 'tp_label: ', np.unique(255*tp_label.asnumpy())
                #print 'pred_tp: ', pred_tp.shape, np.unique(pred_tp.asnumpy())
                #print 'bt_label: ', np.unique(255*bt_label.asnumpy())
                #print 'pred_bt: ', pred_bt.shape, np.unique(pred_bt.asnumpy())
                #print 'lf_label: ', np.unique(255*lf_label.asnumpy())
                #print 'pred_lf: ', pred_lf.shape, np.unique(pred_lf.asnumpy())
                #print 'rt_label: ', np.unique(255*rt_label.asnumpy())
                #print 'pred_rt: ', pred_rt.shape, np.unique(pred_rt.asnumpy())
                if not nbatch%10:
                    logging.info('Epoch[%d]-batch[%d]---------------------------------------', epoch, nbatch)
                eval_metric_MuLb.update([sc_label, tp_label, bt_label, lf_label, rt_label], [pred_sc, pred_tp, pred_bt, pred_lf, pred_rt])
                self.exector.outputs[0].wait_to_read()
                batch_end_params = BatchEndParam(epoch=epoch, nbatch=nbatch, eval_metric=eval_metric_MuLb)
                batch_end_callback(batch_end_params)
                #lr_rate = self.optimizer.lr
            if epoch_end_callback != None:
                epoch_end_callback(epoch, self.symbol, self.arg_params, auxparams)
            # evaluation
            if eval_data:
                logger.info(" in eval process...")
                nbatch = 0
                eval_data.reset()
                vl = [0.0,0.0,0.0,0.0,0.0]
                for data in eval_data:
                    nbatch += 1
                    label_shape = data[label_sc].shape
                    self.arg_params[data_name] = mx.nd.array(data[data_name], self.ctx)
                    self.arg_params[label_sc] = mx.nd.array(data[label_sc].reshape(label_shape[0], label_shape[1], label_shape[2], label_shape[3]), self.ctx)
                    self.arg_params[label_tp] = mx.nd.array(data[label_tp].reshape(label_shape[0], label_shape[1], label_shape[2], label_shape[3]), self.ctx)
                    self.arg_params[label_bt] = mx.nd.array(data[label_bt].reshape(label_shape[0], label_shape[1], label_shape[2], label_shape[3]), self.ctx)
                    self.arg_params[label_lf] = mx.nd.array(data[label_lf].reshape(label_shape[0], label_shape[1], label_shape[2], label_shape[3]), self.ctx)
                    self.arg_params[label_rt] = mx.nd.array(data[label_rt].reshape(label_shape[0], label_shape[1], label_shape[2], label_shape[3]), self.ctx)
                    exector = self.symbol.bind(self.ctx, self.arg_params,
                                    args_grad=self.grad_params,
                                    grad_req=grad_req,
                                    aux_states=auxparams)
                    #output_sc = output_buff['loss_sc_output']
                    #output_tp = output_buff['loss_tp_output']
                    #output_bt = output_buff['loss_bt_output']
                    #output_lf = output_buff['loss_lf_output']
                    #output_rt = output_buff['loss_rt_output']
                    cpu_output_sc = mx.nd.zeros(exector.outputs[0][0][0].shape)
                    cpu_output_tp = mx.nd.zeros(exector.outputs[1][0][0].shape)
                    cpu_output_bt = mx.nd.zeros(exector.outputs[2][0][0].shape)
                    cpu_output_lf = mx.nd.zeros(exector.outputs[3][0][0].shape)
                    cpu_output_rt = mx.nd.zeros(exector.outputs[4][0][0].shape)
                    exector.forward(is_train=False)
                    exector.outputs[0][0][0].copyto(cpu_output_sc)
                    exector.outputs[1][0][0].copyto(cpu_output_tp)
                    exector.outputs[2][0][0].copyto(cpu_output_bt)
                    exector.outputs[3][0][0].copyto(cpu_output_lf)
                    exector.outputs[4][0][0].copyto(cpu_output_rt)
                    pred_shape = cpu_output_sc.shape
                    sc_label = mx.nd.array(data[label_sc].reshape(label_shape[2], label_shape[3]))
                    pred_sc = mx.nd.array(cpu_output_sc.asnumpy().reshape(pred_shape[0], pred_shape[1]))
                    tp_label = mx.nd.array(data[label_tp].reshape(label_shape[2], label_shape[3]))
                    pred_tp = mx.nd.array(cpu_output_tp.asnumpy().reshape(pred_shape[0], pred_shape[1]))
                    bt_label = mx.nd.array(data[label_bt].reshape(label_shape[2], label_shape[3]))
                    pred_bt = mx.nd.array(cpu_output_bt.asnumpy().reshape(pred_shape[0], pred_shape[1]))
                    lf_label = mx.nd.array(data[label_lf].reshape(label_shape[2], label_shape[3]))
                    pred_lf = mx.nd.array(cpu_output_lf.asnumpy().reshape(pred_shape[0], pred_shape[1]))
                    rt_label = mx.nd.array(data[label_rt].reshape(label_shape[2], label_shape[3]))
                    pred_rt = mx.nd.array(cpu_output_rt.asnumpy().reshape(pred_shape[0], pred_shape[1]))
                    eval_metric_MuLb.update([sc_label, tp_label, bt_label, lf_label, rt_label], [pred_sc, pred_tp, pred_bt, pred_lf, pred_rt])
                    #eval_metric_sc.update([sc_label], [pred_sc])
                    #eval_metric_tp.update([tp_label], [pred_tp])
                    #eval_metric_bt.update([bt_label], [pred_bt])
                    #eval_metric_lf.update([lf_label], [pred_lf])
                    #eval_metric_rt.update([rt_label], [pred_rt])
                    #pdb.set_trace()
                    name, value = eval_metric_MuLb.get() 
                    vl = np.add(vl,value)
                sc = vl/nbatch
                logger.info('batch[%d] Validation-%s=%f', nbatch, name[0], sc[0])
                logger.info('batch[%d] Validation-%s=%f', nbatch, name[1], sc[1])
                logger.info('batch[%d] Validation-%s=%f', nbatch, name[2], sc[2])
                logger.info('batch[%d] Validation-%s=%f', nbatch, name[3], sc[3])
                logger.info('batch[%d] Validation-%s=%f', nbatch, name[4], sc[4])
