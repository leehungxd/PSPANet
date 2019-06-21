# -*- coding: utf-8 -*-

import sys

sys.path.append('/home/lihang/projects/PSPNet/python')
# sys.path.append('/home/lihang/projects/fcn')
import caffe
import surgery, score

import numpy as np
import os
import sys

try:
    import setproctitle
    setproctitle.setproctitle(os.path.basename(os.getcwd()))
except:
    pass

# vgg_weights = 'fcn8s-heavy-pascal.caffemodel'
# vgg_proto = 'train.prototxt'
weights = 'jobs/cityscape/pspnet50/snapshots/pspnet50_cityscape_iter_19040.caffemodel'

# init
caffe.set_mode_gpu()
# caffe.set_device(int(sys.argv[0]))
caffe.set_device(0)

solver = caffe.SGDSolver('solver.prototxt')
solver.net.copy_from(weights)
solver.net
# solver = caffe.SGDSolver('solver.prototxt')
# vgg_net = caffe.Net(vgg_proto, vgg_weights, caffe.TRAIN)
# surgery.transplant(solver.net, vgg_net)
# del vgg_net

# surgeries
# interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
# surgery.interp(solver.net, interp_layers)

# scoring
test = np.loadtxt('/home/lihang/data/cityscape/valSeg.txt', dtype=str)

for _ in range(100):
    solver.step(2975)
    # N.B. metrics on the semantic labels are off b.c. of missing classes;
    # score manually from the histogram instead for proper evaluation
    score.seg_tests(solver, False, test, layer='conv6_interp')
    # score.seg_tests(solver, False, test, layer='score_geo', gt='geo')