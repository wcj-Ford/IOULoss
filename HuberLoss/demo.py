import sys
import pdb
import IOULoss
import numpy as np
import init_fcnxs_resnet50
from PIL import Image

sys.path.append('/home/wcj/mxnet/python')
import mxnet as mx

model_previx = sys.argv[1]
epoch = int(sys.argv[2])
img = sys.argv[3]
seg = 'test/'+img[-10:-4]
ctx = mx.cpu(0)

def get_data(img_path):
    mean = np.array([123.68, 116.779, 103.939])
    img = Image.open(img_path)
    img = np.array(img, dtype=np.float32)
    reshaped_mean = mean.reshape(1,1,3)
    img = img-reshaped_mean
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = np.expand_dims(img, axis=0)
    #img = np.expand_dims(img, axis=0)
    print '-----------------------img.shape: ', img.shape
    return img

def main():
    pdb.set_trace()
    fcnxs, fcnxs_args, fcnxs_auxs = mx.model.load_checkpoint(model_previx, epoch)
    all_layes = fcnxs.get_internals()
    #mx.viz.plot_network(fcnxs).view()
    data = mx.nd.array(get_data(img), ctx)
    fcnxs_args['data'] = mx.nd.array(get_data(img), ctx)
    d_sh = get_data(img).shape
    label_shape = (1,1,d_sh[2], d_sh[3])
    fcnxs_args['score_label'] = mx.nd.empty(label_shape, ctx)
    fcnxs_args['top_label'] = mx.nd.empty(label_shape, ctx)
    fcnxs_args['bottom_label'] = mx.nd.empty(label_shape, ctx)
    fcnxs_args['left_label'] = mx.nd.empty(label_shape, ctx)
    fcnxs_args['right_label'] = mx.nd.empty(label_shape, ctx)
    fcnxs_args, fcnxs_auxs = init_fcnxs_resnet50.init_retrain(ctx, fcnxs, fcnxs_args, fcnxs_auxs)
    exector = fcnxs.bind(ctx, fcnxs_args, args_grad=None, grad_req='null', aux_states = fcnxs_auxs)
    exector.forward(is_train=False, data=data)
    output = exector.outputs[0]
    sc = output[0][0]
    tp = output[0][1]
    bt = output[0][2]
    lf = output[0][3]
    rt = output[0][4]
    pdb.set_trace()
    img_sc = Image.fromarray(int(sc.asnumpy()*255))
    img_tp = Image.fromarray(int(tp.asnumpy()*255))
    img_bt = Image.fromarray(int(bt.asnumpy()*255))
    img_lf = Image.fromarray(int(lf.asnumpy()*255))
    img_rt = Image.fromarray(int(rt.asnumpy()*255))
    #img_sc.save(seq+'_sc.png')
    #img_tp.save(seq+'_tp.png')
    #img_bt.save(seq+'_bt.png')
    #img_lf.save(seq+'_lf.png')
    #img_rt.save(seq+'_rt.png')

if __name__=='__main__':
    main()
