import os
import sys
import argparse
import time
from numpy import *
from numpy.random import *
root_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(root_dir, 'caffe', 'python'))
import caffe

begin = 1
end = 2049
data_num = 222628 + 355539
data_dim = end - begin
dataset = zeros([data_num, data_dim], dtype=float32)
img_names = []
img_labels = []

#load data
print 'Loading siyang dataset'
count = 1
with open("/home/xqt/features/v0_siyang_fc7.fea", 'r') as f:
    for line in f:
        ln = line.split()
        img_name = ln[0]
        img_names.append('/media/mmr6-home/lhy/Documents/Data/Vehicles/vehicles_with_plate/cropped_uncovered/' + img_name)
        with open(os.path.join('/media/megatron-home/dwliang/search_eval/query/cropped_label/', img_name.split('.')[0]+'.txt'), 'r') as f_label:
            img_labels.append(int(f_label.readlines()[0].strip()))
        #for i in range(1024):
        #    dataset[count][i] = ln[i+1]
        dataset[count] = ln[begin:end]
        count = count+1

print 'Loading wendeng dataset'
with open("/home/xqt/features/v0_wendeng_fc7.fea", 'r') as f:
    for line in f:
        ln = line.split()
        img_names.append('/media/megatron-home/dwliang/data/wendeng_res/' + ln[0])
        img_labels.append(int(-1))
        #for i in range(1024):
        #    dataset[count][i] = ln[i+1]
        dataset[count] = ln[begin:end]
        count = count+1
        if count == data_num:
            break
#print('Done in %.2f s' % (time.time()-start))

print 'computing dot product of dataset rows'
dataset_dot_product = array([dataset[i].dot(dataset[i]) for i in range(data_num)])

# load net
print "load net"
caffe.set_device(0)
caffe.set_mode_gpu()
net = caffe.Net("/home/xqt/exp/deploy_multi_no_sigm.prototxt", 
				"/home/xqt/essence/no_sigm_iter_75000.caffemodel", caffe.TEST)

in_ = net.inputs[0]
in_shape = net.blobs[in_].data.shape
transformer = caffe.io.Transformer({in_: in_shape})
transformer.set_transpose(in_, (2, 0, 1))
transformer.set_raw_scale(in_, 255)
transformer.set_channel_swap(in_, (2, 1, 0))


query_path = "/media/mmr6-home/lhy/Documents/Data/Vehicles/vehicles_with_plate/cropped_uncovered/0294204.jpg"
query_img = caffe.io.load_image(query_path)

print('extracting features')
input_ = transformer.preprocess(in_, caffe.io.resize_image(query_img, (in_shape[2], in_shape[3])))
out = net.forward_all(**{in_: input_.reshape((1, 3, in_shape[2], in_shape[3]))})
feat = net.blobs["fc7"].data[0].flatten()


dist = dataset_dot_product-2*dataset.dot(feat)
result = dist.argsort()

for i in range(10):
	print img_names[result[i]]






