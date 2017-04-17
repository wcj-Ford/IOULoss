# pylint: skip-file
import mxnet as mx
import IOULoss

def residual_unit(data, num_filter, stride, dim_match, name, bottle_neck=True, bn_mom=0.9, workspace=256, memonger=False):
    """Return ResNet Unit symbol for building ResNet
    Parameters
    ----------
    data : str
        Input data
    num_filter : int
        Number of output channels
    bnf : int
        Bottle neck channels factor with regard to num_filter
    stride : tupe
        Stride used in convolution
    dim_match : Boolen
        True means channel number between input and output is the same, otherwise means differ
    name : str
        Base name of the operators
    workspace : int
        Workspace used in convolution operator
    """
    if bottle_neck:
        # the same as https://github.com/facebook/fb.resnet.torch#notes, a bit difference with origin paper
        bn1 = mx.sym.BatchNorm(data=data)
        act1 = mx.sym.Activation(data=bn1, act_type='relu')
        conv1 = mx.sym.Convolution(data=act1, num_filter=int(num_filter*0.25), kernel=(1,1), stride=(1,1), pad=(0,0),
                                   no_bias=True, workspace=workspace, name=name + '_conv1')
        bn2 = mx.sym.BatchNorm(data=conv1)
        act2 = mx.sym.Activation(data=bn2, act_type='relu')
        conv2 = mx.sym.Convolution(data=act2, num_filter=int(num_filter*0.25), kernel=(3,3), stride=stride, pad=(1,1),
                                   no_bias=True, workspace=workspace, name=name + '_conv2')
        bn3 = mx.sym.BatchNorm(data=conv2)
        act3 = mx.sym.Activation(data=bn3, act_type='relu')
        conv3 = mx.sym.Convolution(data=act3, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0), no_bias=True,
                                   workspace=workspace, name=name + '_conv3')
        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_sc')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return conv3 + shortcut
    else:
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1)
        conv1 = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(3,3), stride=stride, pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv1')
        bn2 = mx.sym.BatchNorm(data=conv1)
        act2 = mx.sym.Activation(data=bn2)
        conv2 = mx.sym.Convolution(data=act2, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv2')
        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_sc')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return conv2 + shortcut

def residual_unit_nobn(data, num_filter, stride, dim_match, name, bottle_neck=True, bn_mom=0.9, workspace=256, memonger=False):
    """Return ResNet Unit symbol for building ResNet
    Parameters
    ----------
    data : str
        Input data
    num_filter : int
        Number of output channels
    bnf : int
        Bottle neck channels factor with regard to num_filter
    stride : tupe
        Stride used in convolution
    dim_match : Boolen
        True means channel number between input and output is the same, otherwise means differ
    name : str
        Base name of the operators
    workspace : int
        Workspace used in convolution operator
    """
    if bottle_neck:
        # the same as https://github.com/facebook/fb.resnet.torch#notes, a bit difference with origin paper
        #bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
        act1 = mx.sym.Activation(data=data, act_type='relu')
        conv1 = mx.sym.Convolution(data=act1, num_filter=int(num_filter*0.25), kernel=(1,1), stride=(1,1), pad=(0,0),
                                   no_bias=True, workspace=workspace, name=name + '_conv1')
        #bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
        act2 = mx.sym.Activation(data=conv1, act_type='relu')
        conv2 = mx.sym.Convolution(data=act2, num_filter=int(num_filter*0.25), kernel=(3,3), stride=stride, pad=(1,1),
                                   no_bias=True, workspace=workspace, name=name + '_conv2')
        #bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')
        act3 = mx.sym.Activation(data=conv2, act_type='relu')
        conv3 = mx.sym.Convolution(data=act3, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0), no_bias=True,
                                   workspace=workspace, name=name + '_conv3')
        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_sc')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return conv3 + shortcut
    else:
        #bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn1')
        act1 = mx.sym.Activation(data=data, act_type='relu')
        conv1 = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(3,3), stride=stride, pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv1')
        #bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn2')
        act2 = mx.sym.Activation(data=conv1, act_type='relu')
        conv2 = mx.sym.Convolution(data=act2, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv2')
        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_sc')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return conv2 + shortcut

def resnet(input_data, units, num_stages, filter_list, num_classes, bottle_neck=True, bn_mom=0.9, workspace=1024, memonger=False):
    """Return ResNet symbol of
    Parameters
    ----------
    units : list
        Number of units in each stage
    num_stages : int
        Number of stage
    filter_list : list
        Channel size of each stage
    num_classes : int
        Ouput size of symbol
    dataset : str
        Dataset type, only cifar10 and imagenet supports
    workspace : int
        Workspace used in convolution operator
    """
    num_unit = len(units)
    assert(num_unit == num_stages)
    data = mx.sym.Variable(name='data')
    res3 = mx.sym.Variable(name='data3')
    res4 = mx.sym.Variable(name='data4')
    data = input_data
    data = mx.sym.BatchNorm(data=data)
    body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(7, 7), stride=(2,2), pad=(64, 64),
                              no_bias=True, name="conv0", workspace=workspace)
    body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom)
    body = mx.sym.Activation(data=body, act_type='relu')
    body = mx.symbol.Pooling(data=body, kernel=(3, 3), stride=(2,2), pad=(1,1), pool_type='max')
    for i in range(num_stages):
        #print i
        body = residual_unit_nobn(body, filter_list[i+1], (1 if i==0 else 2, 1 if i==0 else 2), False,
                             name='stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck, workspace=workspace,
                             memonger=memonger)
        for j in range(units[i]-1):
            body = residual_unit(body, filter_list[i+1], (1,1), True, name='stage%d_unit%d' % (i + 1, j + 2),
                                 bottle_neck=bottle_neck, workspace=workspace, memonger=memonger)
        if i==0:
            #print '-------------------------------------------i ', i
            #arg_shape, out_shape, aux_shape = body.infer_shape(data=(1,1, 224, 224))
            #print '-------------------------------------------res3 body shape', out_shape
            res3 = body
            #print body.infer_shape()
        elif i==1:
            res4 = body
            #print '------------------------------------------i ', i
            #arg_shape, out_shape, aux_shape = body.infer_shape(data=(1,1, 224, 224))
            #print '-------------------------------------------res4 body shape', out_shape
    bn1 = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom)
    relu1 = mx.sym.Activation(data=bn1, act_type='relu')
    #arg_shape, out_shape, aux_shape = relu1.infer_shape(data=(1,1, 224, 224))
    #print '-------------------------------------------relu1 shape', out_shape
    fcc1 = mx.symbol.Convolution(data=relu1, num_filter=512, kernel=(1, 1), stride=(1,1), pad=(0, 0), name='fcc1')
    #arg_shape, out_shape, aux_shape = fcc1.infer_shape(data=(1,1, 224, 224))
    #print '-------------------------------------------fcc1 shape', out_shape
    return fcc1, res3, res4

def filter_map(kernel=1, stride=1, pad=0):
    return (stride, (kernel-stride)/2-pad)

def compose_fp(fp_first, fp_second):
    return (fp_first[0]*fp_second[0], fp_first[0]*fp_second[1]+fp_first[1])

def compose_fp_list(fp_list):
    fp_out = (1.0, 0.0)
    for fp in fp_list:
        fp_out = compose_fp(fp_out, fp)
    return fp_out

def inv_fp(fp_in):
    return (1.0/fp_in[0], -1.0*fp_in[1]/fp_in[0])

def offset():
    conv1_1_fp = filter_map(kernel=3, pad=100)
    conv1_2_fp = conv2_1_fp = conv2_2_fp = conv3_1_fp = conv3_2_fp = conv3_3_fp \
               = conv4_1_fp = conv4_2_fp = conv4_3_fp = conv5_1_fp = conv5_2_fp \
               = conv5_3_fp = filter_map(kernel=3, pad=1)
    pool1_fp = pool2_fp = pool3_fp = pool4_fp = pool5_fp = filter_map(kernel=2, stride=2)
    fc6_fp = filter_map(kernel=7)
    fc7_fp = score_fp = score_pool4_fp = score_pool3_fp = filter_map()
    # for fcn-32s
    fcn32s_upscore_fp = inv_fp(filter_map(kernel=64, stride=32))
    fcn32s_upscore_list = [conv1_1_fp, conv1_2_fp, pool1_fp, conv2_1_fp, conv2_2_fp,
                           pool2_fp, conv3_1_fp, conv3_2_fp, conv3_3_fp, pool3_fp,
                           conv4_1_fp, conv4_2_fp, conv4_3_fp, pool4_fp, conv5_1_fp,
                           conv5_2_fp, conv5_3_fp, pool5_fp, fc6_fp, fc7_fp, score_fp,
                           fcn32s_upscore_fp]
    crop = {}
    crop["fcn32s_upscore"] = (-int(round(compose_fp_list(fcn32s_upscore_list)[1])),
                              -int(round(compose_fp_list(fcn32s_upscore_list)[1])))
    # for fcn-16s
    score2_fp = inv_fp(filter_map(kernel=4, stride=2))
    fcn16s_upscore_fp = inv_fp(filter_map(kernel=32, stride=16))
    score_pool4c_fp_list = [inv_fp(score2_fp), inv_fp(score_fp), inv_fp(fc7_fp), inv_fp(fc6_fp),
                            inv_fp(pool5_fp), inv_fp(conv5_3_fp), inv_fp(conv5_2_fp),
                            inv_fp(conv5_1_fp), score_pool4_fp]
    crop["score_pool4c"] = (-int(round(compose_fp_list(score_pool4c_fp_list)[1])),
                            -int(round(compose_fp_list(score_pool4c_fp_list)[1])))
    fcn16s_upscore_list =  [conv1_1_fp, conv1_2_fp, pool1_fp, conv2_1_fp, conv2_2_fp,
                            pool2_fp, conv3_1_fp, conv3_2_fp, conv3_3_fp, pool3_fp,
                            conv4_1_fp, conv4_2_fp, conv4_3_fp, pool4_fp, score_pool4_fp,
                            inv_fp((1, -crop["score_pool4c"][0])), fcn16s_upscore_fp]
    crop["fcn16s_upscore"] = (-int(round(compose_fp_list(fcn16s_upscore_list)[1])),
                              -int(round(compose_fp_list(fcn16s_upscore_list)[1])))
    # for fcn-8s
    score4_fp = inv_fp(filter_map(kernel=4, stride=2))
    fcn8s_upscore_fp = inv_fp(filter_map(kernel=16, stride=8))
    score_pool3c_fp_list = [inv_fp(score4_fp), (1, -crop["score_pool4c"][0]), inv_fp(score_pool4_fp),
                            inv_fp(pool4_fp), inv_fp(conv4_3_fp), inv_fp(conv4_2_fp),
                            inv_fp(conv4_1_fp), score_pool3_fp, score_pool3_fp]
    crop["score_pool3c"] = (-int(round(compose_fp_list(score_pool3c_fp_list)[1])),
                            -int(round(compose_fp_list(score_pool3c_fp_list)[1])))
    fcn8s_upscore_list =  [conv1_1_fp, conv1_2_fp, pool1_fp, conv2_1_fp, conv2_2_fp, pool2_fp,
                           conv3_1_fp, conv3_2_fp, conv3_3_fp, pool3_fp, score_pool3_fp,
                           inv_fp((1, -crop["score_pool3c"][0])), fcn8s_upscore_fp]
    crop["fcn8s_upscore"] = (-int(round(compose_fp_list(fcn8s_upscore_list)[1])),
                             -int(round(compose_fp_list(fcn8s_upscore_list)[1])))
    return crop

def fcnxs_score(input, label_sc, label_tp, label_bt, label_lf, label_rt, crop, offset, kernel=(64,64), stride=(32,32), numclass=5, workspace_default=1024):
    bigscore = mx.symbol.Deconvolution(data=input, kernel=kernel, stride=stride, adj=(stride[0]-1, stride[1]-1), num_filter=256, workspace=workspace_default, name="bigscore")
    upscore = mx.symbol.Crop(*[bigscore, crop], offset=offset, name='upscore')
    fcn_sc = mx.symbol.Convolution(data=upscore, num_filter=128, kernel=(3, 3), stride=(1,1), pad=(1, 1), name='fcn_sc', workspace=workspace_default)
    fcn_bx = mx.symbol.Convolution(data=upscore, num_filter=128, kernel=(3, 3), stride=(1,1), pad=(1, 1), name='fcn_bx', workspace=workspace_default)
    fcn1_sc = mx.symbol.Convolution(data=fcn_sc, num_filter=1, kernel=(1, 1), stride=(1,1), pad=(0, 0), name='fcn1_sc', workspace=workspace_default)
    bx_tp = mx.symbol.Convolution(data=fcn_bx, num_filter=1, kernel=(1, 1), stride=(1,1), pad=(0, 0), name='bx_tp', workspace=workspace_default)
    bx_bt = mx.symbol.Convolution(data=fcn_bx, num_filter=1, kernel=(1, 1), stride=(1,1), pad=(0, 0), name='bx_bt', workspace=workspace_default)
    bx_lf = mx.symbol.Convolution(data=fcn_bx, num_filter=1, kernel=(1, 1), stride=(1,1), pad=(0, 0), name='bx_lf', workspace=workspace_default)
    bx_rt = mx.symbol.Convolution(data=fcn_bx, num_filter=1, kernel=(1, 1), stride=(1,1), pad=(0, 0), name='bx_rt', workspace=workspace_default)
    #fcn1_sc = mx.sym.Activation(data=fcn1_sc, act_type='sigmoid')
    bx_tp = mx.sym.Activation(data=bx_tp, act_type='relu')
    bx_bt = mx.sym.Activation(data=bx_bt, act_type='relu')
    bx_lf = mx.sym.Activation(data=bx_lf, act_type='relu')
    bx_rt = mx.sym.Activation(data=bx_rt, act_type='relu')
    fcn_data = mx.symbol.Concat(fcn1_sc, bx_tp, bx_bt, bx_lf, bx_rt)
    label = mx.symbol.Concat(label_sc, label_tp, label_bt, label_lf, label_rt)
    #ioudata = mx.symbol.Concat(upscore, label)
    iouloss = mx.symbol.Custom(data=fcn_data, label=label, name='iouLoss', op_type='IOULoss')
    #arg_shape, out_shape, aux_shape = iouloss.infer_shape(data=(1, 3, 500, 500), score_label=(1,500,500), top_label=(1,500,500), bottom_label=(1,500,500), left_label=(1,500,500), right_label=(1,500,500))
    #arg_shape, out_shape, aux_shape = iouloss.infer_shape(data=(1,3, 500, 500))
    #print '-----------------------------------------------iouloss', out_shape
    return iouloss

def get_fcn32s_symbol(numclass=21, workspace_default=1024):
    data = mx.symbol.Variable(name="data")
    #label_sc=label_tp=label_bt=label_lf=label_rt=data
    label_sc = mx.symbol.Variable("score_label")
    label_tp = mx.symbol.Variable("top_label")
    label_bt = mx.symbol.Variable('bottom_label')
    label_lf = mx.symbol.Variable('left_label')
    label_rt = mx.symbol.Variable('right_label')
    score, res3, res4 = resnet(data, [3, 4, 6, 3], 4, [64, 256, 512, 1024, 1024], 5, True) 
    #arg_shape, out_shape, aux_shape = score.infer_shape(data=(1,3, 500, 500))
    #print arg_shape
    #print out_shape
    #print aux_shape
    iouloss = fcnxs_score(score, label_sc, label_tp, label_bt, label_lf, label_rt, data, offset()["fcn32s_upscore"], (64, 64), (32, 32), 5, workspace_default)
    #iouloss = fcnxs_score(score, data, offset()["fcn32s_upscore"], (64,64), (32,32), 5, workspace_default)
    return iouloss

def get_fcn16s_symbol(numclass=21, workspace_default=1024):
    data = mx.symbol.Variable(name="data")
    label_sc = mx.symbol.Variable("score_label")
    label_tp = mx.symbol.Variable("top_label")
    label_bt = mx.symbol.Variable(name='bottom_label')
    label_lf = mx.symbol.Variable(name='left_label')
    label_rt = mx.symbol.Variable(name='right_label')
    score, res3, res4 = resnet(data, [3, 4, 6, 3], 4, [64, 256, 512, 1024, 512], 5, True)
    score2 = mx.symbol.Deconvolution(data=score, kernel=(4, 4), stride=(2, 2), num_filter=numclass,
                     adj=(1, 1), workspace=workspace_default, name="score2")  # 2X
    score_pool4 = mx.symbol.Convolution(data=res4, kernel=(1, 1), num_filter=numclass,
                     workspace=workspace_default, name="score_pool4")
    score_pool4c = mx.symbol.Crop(*[score_pool4, score2], offset=(3,3), name="score_pool4c")
    score_fused = score2 + score_pool4c
    softmax = fcnxs_score(score_fused, label_sc, label_tp, label_bt, label_lf, label_rt, data, offset()["fcn16s_upscore"], (32, 32), (16, 16), 5, workspace_default)
    #softmax = fcnxs_score(score_fused, data, offset()["fcn16s_upscore"], (32, 32), (16, 16), numclass, workspace_default)
    return softmax

def get_fcn8s_symbol(numclass=5, workspace_default=1024):
    data = mx.symbol.Variable("data")
    label_sc = mx.symbol.Variable("score_label")
    label_tp = mx.symbol.Variable("top_label")
    label_bt = mx.symbol.Variable(name='bottom_label')
    label_lf = mx.symbol.Variable(name='left_label')
    label_rt = mx.symbol.Variable(name='right_label')
    score, res3, res4 = resnet(data, [3, 4, 6, 3], 4, [64, 256, 512, 1024, 2048], 5, True)
    #print '\n\n--------------------------------------get_fcn8s_symbol-> '
    #arg_shape, out_shape, aux_shape = score.infer_shape(data=(1,1, 587, 587))
    #print '--------------------------------------score out shape ', out_shape
    #arg_shape, out_shape, aux_shape = res3.infer_shape(data=(1,1, 587, 587))
    #print '--------------------------------------res3 out shape ', out_shape
    #arg_shape, out_shape, aux_shape = res4.infer_shape(data=(1,1, 587, 587))
    #print '--------------------------------------res4 out shape ', out_shape
    score2 = mx.symbol.Deconvolution(data=score, kernel=(4, 4), stride=(2, 2),num_filter=5,
    adj=(1, 1), workspace=workspace_default, name="score2")  # 2X
    score_pool4 = mx.symbol.Convolution(data=res4, kernel=(1, 1), num_filter=5,
                     workspace=workspace_default, name="score_pool4")
    #score_pool4c = mx.symbol.Crop(*[score_pool4, score2], offset=offset()["score_pool4c"], name="score_pool4c")
    score_pool4c = mx.symbol.Crop(*[score_pool4, score2], offset=(3,3), name="score_pool4c")
    #arg_shape, out_shape, aux_shape = score2.infer_shape(data=(1,1, 587, 587))
    #print '--------------------------------------score2 out shape ', out_shape
    #arg_shape, out_shape, aux_shape = score_pool4c.infer_shape(data=(1,1, 587, 587))
    #print '--------------------------------------score_pool4c out shape ', out_shape
    score_fused = score2 + score_pool4c
    # score 4X
    score4 = mx.symbol.Deconvolution(data=score_fused, kernel=(4, 4), stride=(2, 2),num_filter=5,
                     adj=(1, 1), workspace=workspace_default, name="score4") # 4X
    score_pool3 = mx.symbol.Convolution(data=res3, kernel=(1, 1), num_filter=numclass,
                     workspace=workspace_default, name="score_pool3")
    #score_pool3c = mx.symbol.Crop(*[score_pool3, score4], offset=offset()["score_pool3c"], name="score_pool3c")
    score_pool3c = mx.symbol.Crop(*[score_pool3, score4], offset=(3,3), name="score_pool3c")
    #arg_shape, out_shape, aux_shape = score4.infer_shape(data=(1,1, 587, 587))
    #print '--------------------------------------score4 out shape ', out_shape
    #arg_shape, out_shape, aux_shape = score_pool3c.infer_shape(data=(1,1, 587, 587))
    #print '--------------------------------------score_pool3c out shape ', out_shape
    score_final = score4 + score_pool3c
    softmax = fcnxs_score(score_final, label_sc, label_tp, label_bt, label_lf, label_rt, data, offset()["fcn8s_upscore"], (16, 16), (8, 8), numclass, workspace_default)
    return softmax

