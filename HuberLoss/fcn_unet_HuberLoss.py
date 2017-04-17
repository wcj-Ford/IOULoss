# pylint: skip-file
import sys, os
sys.path.append('/home/wcj/mxnet/python')
import sys, os
import argparse
import mxnet as mx
import numpy as np
import logging
import pdb
import symbol_unet_HuberLoss
import init_unet
from data import FileIter
from solver import Solver

logger = logging.getLogger()
logger.setLevel(logging.INFO)
ctx = mx.gpu(0)
#ctx = mx.cpu()

def main():
    #pdb.set_trace()
    fcnxs = symbol_unet_HuberLoss.get_unet(workspace=8192)
    fcnxs_model_prefix = "model_pascal/unet-huberloss"
    #arg_names = fcnxs.list_arguments()
    fcnxs_args = init_unet.init(ctx, fcnxs)
    train_dataiter = FileIter(
        root_dir             = "/home/wcj/fcn-xs/data/faceMultiLabel",
        flist_name           = "train.lst",
        cut_off_size         = 600,
        rgb_mean             = (123.68, 116.779, 103.939),
        )
    val_dataiter = FileIter(
        root_dir             = "/home/wcj/fcn-xs/data/faceMultiLabel",
        flist_name           = "val.lst",
        cut_off_size         = 600,
        rgb_mean             = (123.68, 116.779, 103.939),
        )
    model = Solver(
        ctx                 = ctx,
        symbol              = fcnxs,
        begin_epoch         = 0,
        num_epoch           = 50,
        arg_params          = fcnxs_args,
        aux_params          = fcnxs_args,
        learning_rate       = 1e-3,
        momentum            = 0.9,
        wd                  = 0.0005)
#    pdb.set_trace()
    model.fit(
        train_data          = train_dataiter,
        eval_data           = val_dataiter,
        batch_end_callback  = mx.callback.Speedometer(1, 10),
        epoch_end_callback  = mx.callback.do_checkpoint(fcnxs_model_prefix))

if __name__ == "__main__":
    #parser = argparse.ArgumentParser(description='Convert vgg16 model to vgg16fc model.')
    #parser.add_argument('--model', default='fcnxs',
    #    help='The type of fcn-xs model, e.g. fcnxs, fcn16s, fcn8s.')
    #parser.add_argument('--prefix', default='VGG_FC_ILSVRC_16_layers',
    #    help='The prefix(include path) of vgg16 model with mxnet format.')
    #parser.add_argument('--epoch', type=int, default=74,
    #    help='The epoch number of vgg16 model.')
    #parser.add_argument('--init-type', default="vgg16",
    #    help='the init type of fcn-xs model, e.g. vgg16, fcnxs')
    #parser.add_argument('--retrain', default=False,
    #    help='true means continue training.')
    #args = parser.parse_args()
    #logging.info(args)
    main()
