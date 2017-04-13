# pylint: skip-file
import mxnet as mx
import numpy as np
import sys
import logging
import pdb
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# make a bilinear interpolation kernel, return a numpy.ndarray
def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1.0
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)

def init_retrain(ctx, fcnxs_symbol, vgg16fc_args, vgg16fc_auxs):
    fcnxs_args = vgg16fc_args.copy()
    fcnxs_auxs = vgg16fc_auxs.copy()
    for k,v in fcnxs_args.items():
        if(v.context != ctx):
            fcnxs_args[k] = mx.nd.zeros(v.shape, ctx)
            v.copyto(fcnxs_args[k])
    for k,v in fcnxs_auxs.items():
        if(v.context != ctx):
            fcnxs_auxs[k] = mx.nd.zeros(v.shape, ctx)
            v.copyto(fcnxs_auxs[k])
    return fcnxs_args, fcnxs_auxs

def init(ctx, syb, fc_args, fc_auxs):
    fcnxs_args = fc_args.copy()
    #fcnxs_auxs = fc_auxs.copy()
    #pdb.set_trace()
    arg_names = syb.list_arguments()
    arg_shapes, _, _ = syb.infer_shape(data=(1,3,500,500),score_label=(1,1,500,500),top_label=(1,1,500,500),bottom_label=(1,1,500,500),left_label=(1,1,500,500),right_label=(1,1,500,500))
    #arg_dict = dict(zip(arg_names, arg_shapes))
    rest_params = dict([(x[0], mx.nd.zeros(x[1], ctx)) for x in zip(arg_names, arg_shapes)])
    for k, v in rest_params.items():
        if not (k.endswith('data') or k.endswith('label')):
            mx.initializer.Normal(0.01)(k, rest_params[k])
    fcnxs_args.update(rest_params)
    #for k, v in arg_dict.items():
    #    if not (k.endswith('data') or k.endswith('label')):
    #        mx.initializer.Normal(0.01)(k, fcnxs_args[k])
    return fcnxs_args, fcnxs_args

def init_from_vgg16(ctx, fcnxs_symbol, vgg16fc_args, vgg16fc_auxs):
    fcnxs_args = vgg16fc_args.copy()
    fcnxs_auxs = vgg16fc_auxs.copy()
    #pdb.set_trace()
    for k,v in fcnxs_args.items():
        if(v.context != ctx):
            fcnxs_args[k] = mx.nd.zeros(v.shape, ctx)
            v.copyto(fcnxs_args[k])
    for k,v in fcnxs_auxs.items():
        if(v.context != ctx):
            fcnxs_auxs[k] = mx.nd.zeros(v.shape, ctx)
            v.copyto(fcnxs_auxs[k])
    arg_names = fcnxs_symbol.list_arguments()
    arg_shapes, _, _ = fcnxs_symbol.infer_shape(data=(1,3,500,500),score_label=(1,1,500,500),top_label=(1,1,500,500),bottom_label=(1,1,500,500),left_label=(1,1,500,500),right_label=(1,1,500,500))
    rest_params = dict([(x[0], mx.nd.ones(x[1], ctx)) for x in zip(arg_names, arg_shapes) if x[0] in ['fcc1_weight', 'fcc1_bias', 'score_pool4_weight', 'score_pool4_bias', 'score_pool3_weight', 'score_pool3_bias']])
    for k, v in rest_params.items():
        mx.initializer.Normal(0.01)(k, rest_params[k])
    fcnxs_args.update(rest_params)
    deconv_params = dict([(x[0], x[1]) for x in zip(arg_names, arg_shapes)
            if x[0] in ["bigscore_weight", 'score2_weight', 'score4_weight']])
    for k, v in deconv_params.items():
        filt = upsample_filt(v[3])
        initw = np.zeros(v)
        for wi in range(v[0]):
            for hi in range(v[1]):
                initw[wi, hi, :, :] = filt
        #initw[range(v[0]), range(v[1]), :, :] = filt  # becareful here is the slice assing
        fcnxs_args[k] = mx.nd.array(initw, ctx)
    return fcnxs_args, fcnxs_auxs

def init_from_fcnxs(ctx, fcnxs_symbol, fcnxs_args_from, fcnxs_auxs_from):
    """ use zero initialization for better convergence, because it tends to oputut 0,
    and the label 0 stands for background, which may occupy most size of one image.
    """
    fcnxs_args = fcnxs_args_from.copy()
    fcnxs_auxs = fcnxs_auxs_from.copy()
    for k,v in fcnxs_args.items():
        if(v.context != ctx):
            fcnxs_args[k] = mx.nd.zeros(v.shape, ctx)
            v.copyto(fcnxs_args[k])
    for k,v in fcnxs_auxs.items():
        if(v.context != ctx):
            fcnxs_auxs[k] = mx.nd.zeros(v.shape, ctx)
            v.copyto(fcnxs_auxs[k])
    data_shape=(1,3,500,500)
    arg_names = fcnxs_symbol.list_arguments()
    #arg_shapes, _, _ = fcnxs_symbol.infer_shape(data=data_shape)
    arg_shapes, _, _ = fcnxs_symbol.infer_shape(data=(1,3,500,500),score_label=(1,1,500,500),top_label=(1,1,500,500),bottom_label=(1,1,500,500),left_label=(1,1,500,500),right_label=(1,1,500,500))
    #pdb.set_trace()
    #fcnxs_args['iouLoss_label']=mx.nd.array(arg_shapes)
    rest_params = {}
    deconv_params = {}
    # this is fcn8s init from fcn16s
    if 'score_pool3_weight' in arg_names:
        rest_params = dict([(x[0], mx.nd.zeros(x[1], ctx)) for x in zip(arg_names, arg_shapes)
            if x[0] in ['score_pool3_bias', 'score_pool3_weight', 'fcc1_weight', 'fcc1_bias', 'score_pool4_weight', 'score_pool4_bias', 'iouLoss_label']])
        deconv_params = dict([(x[0], x[1]) for x in zip(arg_names, arg_shapes) if x[0] \
            in ["bigscore_weight", 'score2_weight', 'score4_weight']])
    # this is fcn16s init from fcn32s
    elif 'score_pool4_weight' in arg_names:
        rest_params = dict([(x[0], mx.nd.zeros(x[1], ctx)) for x in zip(arg_names, arg_shapes)
            if x[0] in ['score_pool4_weight', 'score_pool4_bias']])
        deconv_params = dict([(x[0], x[1]) for x in zip(arg_names, arg_shapes) if x[0] \
            in ["bigscore_weight", 'score2_weight']])
    # this is fcn32s init
    else:
        logging.error("you are init the fcn32s model, so you should use init_from_vgg16()")
        sys.exit()
    fcnxs_args.update(rest_params)
    for k, v in deconv_params.items():
        #pdb.set_trace()
        filt = upsample_filt(v[3])
        initw = np.zeros(v)
        initw[range(v[0]), range(v[1]), :, :] = filt  # becareful here is the slice assing
        fcnxs_args[k] = mx.nd.array(initw, ctx)
    return fcnxs_args, fcnxs_auxs