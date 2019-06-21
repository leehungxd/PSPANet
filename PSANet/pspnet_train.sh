export PYTHONPATH=/home/lihang/projects/PSPNet/python:$PYTHONPATH
cd /home/lihang/projects/PSPNet
./build/tools/caffe train \
--solver="/home/lihang/projects/PSPNet/PSANet/solver.prototxt" \
--gpu=0 2>&1 | tee /home/lihang/projects/PSPNet/PSANet/psanet50.log
# --snapshot="/caffe/caffe-ssd/models/MobileNet/BDD100K/SSD_300x300/Mobile_BDD100K_SSD_300x300_iter_4000.solverstate" \
# --weights="/home/lihang/projects/PSPNet/pspnet/models/pspnet101_cityscapes.caffemodel" \