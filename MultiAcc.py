import sys
import pdb
sys.path.append('/home/wcj/mxnet/python')
import numpy as np
import mxnet as mx

class Multi_Accuracy(mx.metric.EvalMetric):
    """Calculate accuracies of multi label"""

    def __init__(self):
        super(Multi_Accuracy, self).__init__('multi_accuracy')
        self.acc_score = self.acc_tp = self.acc_bt = self.acc_lf = self.acc_rt = 0.0

    def update(self, labels, pres):
        assert len(labels) == len(pres)
        self.label_sc = labels[0].asnumpy()
        self.pred_sc = pres[0].asnumpy()
        #pdb.set_trace()
        self.label_tp = labels[1].asnumpy()
        self.pred_tp = pres[1].asnumpy()
        self.label_bt = labels[2].asnumpy()
        self.pred_bt = pres[2].asnumpy()
        self.label_lf = labels[3].asnumpy()
        self.pred_lf = pres[3].asnumpy()
        self.label_rt = labels[4].asnumpy()
        self.pred_rt = pres[4].asnumpy()
        #pdb.set_trace()
        snum = np.sum(self.label_sc)
        snum = snum*1.0+0.000001
        num = self.pred_sc.size*1.0
        sc=1.0*(self.pred_sc>0.5)
        self.acc_score = np.sum(1*np.equal(sc,self.label_sc))*1.0/num
        #print 'acc_score--------: ', self.acc_score
        self.acc_top = np.sum(1.0*np.equal(self.pred_tp,self.label_tp)*self.label_sc)*1.0/snum
        #print 'acc_top----------: ', self.acc_top
        self.acc_bottom = np.sum(1.0*np.equal(self.pred_bt,self.label_bt)*self.label_sc)*1.0/snum
        #print 'acc_bottom-------: ', self.acc_bottom
        self.acc_left = np.sum(1.0*np.equal(self.pred_lf, self.label_lf)*self.label_sc)*1.0/snum
        #print 'acc_left---------: ', self.acc_left
        self.acc_right = np.sum(1.0*np.equal(self.pred_rt,self.label_rt)*self.label_sc)*1.0/snum
        #print 'acc_right--------: ', self.acc_right

    def get(self):
        names = ['score',
         'top',
         'bottom',
         'left',
         'right']
        values = [self.acc_score,
         self.acc_top,
         self.acc_bottom,
         self.acc_left,
         self.acc_right]
        return (names, values)

    def get_name_value(self):
        name, value = self.get()
        return zip(name, value)
