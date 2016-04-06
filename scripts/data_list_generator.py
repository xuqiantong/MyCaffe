import h5py
#import caffe
import numpy as np
from random import randint
import random

dic = {}
dic_test = {}
dic_train = {}
test_samples = [0]*250

#read & seperate data
src = open('label.txt')

lines = src.readlines()
for i,l in enumerate(lines):
    l = l[:-1]
    sp = l.split(' ')
    dic[sp[0]] = [sp[1], sp[2], sp[3]]
    test_samples[int(sp[2])] += 1

test_samples = [int(x/4.5) for x in test_samples]
print sum(x>0 for x in test_samples)

for k,v in dic.items():
    if test_samples[int(v[1])] > 0:
        test_samples[int(v[1])] -= 1
        dic_test[k] = v
    else:
        dic_train[k] = v


# for triplet net
# test data 
des1 = open('exp/v4/test_a_class.txt', "w")
des2 = open('exp/v4/test_p_color.txt', "w")
des3 = open('exp/v4/test_n_weight.txt', "w")

test_list = [k for k, v in dic_test.items()]
test_nb = {}

for k, v in dic_test.items():
    id = v[0]
    if not test_nb.has_key(id):
        test_nb[id] = []
    test_nb[id].append(k)

l = len(test_list)
w1 = []
w2 = []
w3 = []

for anchor in test_list:
    model = dic_test[anchor][1]
    color = dic_test[anchor][2]

    anchor_neighbors = test_nb[dic_test[anchor][0]]
    loc = anchor_neighbors.index(anchor)
    p = anchor_neighbors[(loc+1)%len(anchor_neighbors)]

    k = randint(0,l-1)
    while test_list[k] in anchor_neighbors:
        k = randint(0,l-1)
            
        
    w1.append(anchor + '.jpg ' + model + '\n')
    w2.append(p + '.jpg ' + color + '\n')
    w3.append(test_list[k] + '.jpg ' + '1\n')
    

order = [i for i in range(len(w1))]
random.shuffle(order)
for i in range(len(w1)):
    j = order[i]
    des1.write(w1[j])
    des2.write(w2[j])
    des3.write(w3[j])


#training data
des1 = open('exp/v4/test_a_class.txt', "w")
des2 = open('exp/v4/test_p_color.txt', "w")
des3 = open('exp/v4/test_n_weight.txt', "w")

train_list = [k for k, v in dic_train.items()]
train_nb = {}

for k, v in dic_train.items():
    id = v[0]
    if not train_nb.has_key(id):
        train_nb[id] = []
    train_nb[id].append(k)

## 9543 different ids
## 522 id only appear once
## 977 id appear twice
## 1384 id appear three times

l = len(train_list)
w1 = []
w2 = []
w3 = []


i = 0
for anchor in train_list:
    i += 1
    print i

    model = dic_train[anchor][1]
    color = dic_train[anchor][2]

    positive = []
    anchor_neighbors = train_nb[dic_train[anchor][0]]
    loc = anchor_neighbors.index(anchor)
    for j in range(1, min(5, len(anchor_neighbors))):
        positive.append(anchor_neighbors[(loc+j)%len(anchor_neighbors)])

    for p in positive:
        negative = []
        for j in range(10):
            k = randint(0,l-1)
            while train_list[k] in anchor_neighbors or train_list[k] in negative:
                k = randint(0,l-1)

            w1.append(anchor + '.jpg ' + model + '\n')
            w2.append(p + '.jpg ' + color + '\n')
            w3.append(train_list[k] + '.jpg ' + '1\n')
    

order = [i for i in range(len(w1))]
random.shuffle(order)
for i in range(len(w1)):
    j = order[i]
    des1.write(w1[j])
    des2.write(w2[j])
    des3.write(w3[j])


'''
#hdf5
SIZE = 224 # fixed size to all images
dir = '/media/megatron-home/lhy/Documents/Data/Vehicles/VehicleID/cropped/'
with open( 'label.txt', 'r' ) as T :
    lines = T.readlines()

#print lines
# If you do not have enough memory split data into
# multiple batches and generate multiple separate h5 files
X = np.zeros( (len(lines), 3, SIZE, SIZE), dtype='f4' )
y = np.zeros( (len(lines),2), dtype='f4' )
for i,l in enumerate(lines):
    l = l[:-1]
    print l
    sp = l.split(' ')
    print sp[1]
    #img = caffe.io.load_image( dir+sp[0]+'.jpg' )
    #img = caffe.io.resize( img, (3, SIZE, SIZE) ) # resize to fixed size

    #you may apply other input transformations here...
    #X[i] = img
    y[i][0] = int(sp[2])
    y[i][1] = int(sp[3])
    print X[i].size
    print y[i]
    break

with h5py.File('train.h5','w') as H:
        #print "hdf5 writing"
    H.create_dataset( 'X', data=X ) # note the name X given to the dataset!
    H.create_dataset( 'y', data=y ) # note the name y given to the dataset!
with open('train_h5_list.txt','w') as L:
    L.write( 'train.h5' ) # list all h5 files you are going to use
'''