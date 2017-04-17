import mxnet as mx
import numpy as np
import sys
import logging
import pdb
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def upsample_filt(size):
    factor = (size+1)//2
    if size%2==1:
        center = factor-1.0
    else:
        center = factor-0.5
    og = np.ogrid[:size, :size]
    return (1-abs(og[0]-center)/factor)*(1-abs(og[1]-center)/factor)

def init(ctx, syb):
    arg_names = syb.list_arguments()
    arg_shapes, _, _ = syb.infer_shape(data=(1,3,500,500), score_label=(1,1,500,500), top_label=(1,1,500,500),bottom_label=(1,1,500,500),left_label=(1,1,500,500),right_label=(1,1,500,500))
    arg_dict = dict([(x[0], mx.nd.zeros(x[1], ctx)) for x in zip(arg_names, arg_shapes)])
    #pdb.set_trace()
    for k, v in arg_dict.items():
        if not (k.endswith('data') or k.endswith('label')):
            mx.initializer.Normal(0.01)(k, arg_dict[k])
        if k.startswith('upsample'):
            #pdb.set_trace()
            v_s = v.shape
            filt = upsample_filt(v_s[3])
            initw = np.zeros(v_s)
            initw[range(v_s[0]), range(v_s[1]), :, :] = filt
            arg_dict[k] = mx.nd.array(initw, ctx)
    #arg_dict = dict([(x[0], mx.nd.zeros(x[1], ctx)) for x in zip(arg_names, arg_shapes)])
    return arg_dict
