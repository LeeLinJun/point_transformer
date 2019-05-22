#!/usr/bin/env python
# coding: utf-8

# In[11]:


import argparse
import math
from datetime import datetime
import h5py
import numpy as np
#import tensorflow as tf
import socket
import importlib
import os
import sys
import modelnet_dataset
import modelnet_h5_dataset
import os
from visdom import Visdom

vis=Visdom(env='3d_trans')

NUM_CLASSES = 40
normal = False 
NUM_POINT= 1024
BATCH_SIZE = 32
if normal:
    assert(NUM_POINT<=10000)
    DATA_PATH = os.path.join(ROOT_DIR, 'data/modelnet40_normal_resampled')
    TRAIN_DATASET = modelnet_dataset.ModelNetDataset(root=DATA_PATH, npoints=NUM_POINT, split='train', normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)
    TEST_DATASET = modelnet_dataset.ModelNetDataset(root=DATA_PATH, npoints=NUM_POINT, split='test', normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)
else:
    assert(NUM_POINT<=2048)
    TRAIN_DATASET = modelnet_h5_dataset.ModelNetH5Dataset('data/modelnet40_ply_hdf5_2048/train_files.txt', batch_size=BATCH_SIZE, npoints=NUM_POINT, shuffle=True)
    TEST_DATASET = modelnet_h5_dataset.ModelNetH5Dataset('data/modelnet40_ply_hdf5_2048/test_files.txt', batch_size=BATCH_SIZE, npoints=NUM_POINT, shuffle=False)


# In[12]:


#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from modules import MultiHeadAttention, PositionwiseFeedForward


# In[13]:


import torch, torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
# cudnn.enabled = False
# cudnn.benchmark = True

class GlobalAveragePooling(nn.Module):
    def __init__(self, dim=-1):
        super(self.__class__, self).__init__()
        self.dim = dim
        
    def forward(self, x):
        return x.mean(dim=self.dim)
    
class GlobalPooling(nn.Module):
    def __init__(self, dim=-1):
        super(self.__class__, self).__init__()
        self.dim = dim
        
    def forward(self, x):
        
        avg = x.mean(dim=self.dim)
        max = x.max(dim=self.dim)[0]
        min = x.min(dim=self.dim)[0]
        
        return torch.cat([min, avg, max], dim=-1)


# In[14]:


class Discriminator(nn.Module):
    def __init__(self, 
                 hidden_dim=128,
                 ffn_dim =256,
                 n_head=8,
                 normalize_loc=True,      #True
                 normalize_scale=True):  #False
        super(Discriminator, self).__init__()
        self.normalize_loc = normalize_loc
        self.normalize_scale = normalize_scale
        self.dropout1  = nn.Dropout(p=0.2) #0.2

        self.conv1=nn.Conv1d(3,64,1)
        self.conv2 = nn.Conv1d(64,128,1)
        self.bn1=nn.BatchNorm1d(64)
        self.bn2=nn.BatchNorm1d(128)
        
        
        self.fc1 = nn.Linear(3, hidden_dim)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0.0)
        self.dropout2  = nn.Dropout(p=0.1)#0.1
        self.dropout3  = nn.Dropout(p=0.1)#0.1
        self.dropout4  = nn.Dropout(p=0.1)#0.1
        self.mha_1 = MultiHeadAttention(n_head=n_head,d_model = hidden_dim)
        self.ffn_1 = PositionwiseFeedForward(hidden_dim, ffn_dim, use_residual=True)
        self.mha_2 = MultiHeadAttention(n_head=n_head,d_model = hidden_dim)
        self.ffn_2 = PositionwiseFeedForward(hidden_dim, ffn_dim, use_residual=True)
        
        self.gl_1 =  GlobalPooling(dim = 1)
        
        self.fc2 = nn.Linear(hidden_dim * 3, 40)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0.0)
        
    def forward(self, x):
        if self.normalize_loc:
            x = x - x.mean(dim=1, keepdim=True)
        if self.normalize_scale:
            x = x / x.std(dim=1, keepdim=True)
        
        #c1 = F.relu(self.bn1(self.conv1(x.transpose(1,2))))
        #c2 = F.relu(self.bn2(self.conv2(c1)))

        #h1 = F.relu(self.fc1(c2.transpose(1,2)))
        h1 = F.relu(self.fc1(x))
        h1 = self.dropout1(h1) 
        h2 = self.mha_1(h1)
        h2 = self.dropout2(h2)
        h3 = self.ffn_1(h2)
        h3 = self.dropout3(h3)
        h4 = self.mha_2(h3)
        h4 = self.dropout4(h4)
        h5 = self.ffn_2(h4)
        score = self.fc2(self.gl_1(h5))
        return score
        


# In[15]:


model = Discriminator(hidden_dim=128,ffn_dim=128,n_head=8)
#model=nn.DataParallel(model,device_ids=[0,1]).cuda()
model = model.cuda(0)


# In[16]:


def compute_loss(X_batch, y_batch):
    X_batch = Variable(torch.FloatTensor(X_batch)).cuda()
    y_batch = Variable(torch.LongTensor(y_batch)).cuda()
    logits = model(X_batch)
    return F.cross_entropy(logits, y_batch).mean()

def iterate_minibatches(X, y, batchsize):
    indices = np.random.permutation(np.arange(len(X)))
    for start in range(0, len(indices), batchsize):
        ix = indices[start: start + batchsize]
        yield X[ix], y[ix]
opt = torch.optim.Adam(model.parameters(),lr=5e-4)


# In[ ]:


import time
from  tqdm import tqdm
num_epochs = 150 # total amount of full passes over training data
batch_size = 32
train_loss = []
val_accuracy = []
for epoch in tqdm(range(num_epochs)):
    start_time = time.time()
    model.train(True) 
    TRAIN_DATASET.reset()
    while TRAIN_DATASET.has_next_batch():
        batch_data, batch_label = TRAIN_DATASET.next_batch(augment=True)
        loss = compute_loss(batch_data, batch_label)
        loss.backward()
        opt.step()
        opt.zero_grad()
        train_loss.append(loss.cpu().detach().numpy())
        del loss
    # And a full pass over the validation data:
    model.train(False) # disable dropout / use averages for batch_norm
    TEST_DATASET.reset()
    while TEST_DATASET.has_next_batch():
        batch_data, batch_label = TEST_DATASET.next_batch(augment=False)
        logits = model(Variable(torch.FloatTensor(batch_data)).cuda())
        #logits = model(Variable(torch.FloatTensor(batch_data)))
        y_pred = logits.max(1)[1].cpu().detach().numpy()
        val_accuracy.append(np.mean(batch_label == y_pred))
        del logits
    
    # Then we print the results for this epoch:
    print("Epoch {} of {} took {:.3f}s".format(
        epoch + 1, num_epochs, time.time() - start_time))
    print("  training loss (in-iteration): \t{:.6f}".format(
        np.mean(train_loss[-9840 // batch_size :])))
    print("  validation accuracy: \t\t\t{:.2f} %".format(
        np.mean(val_accuracy[-2468 // batch_size :]) * 100))
    vis.text("training loss:\t{:.6f}".format( np.mean(train_loss[-9840 // batch_size :])),win='loss_log')
    vis.text("  validation accuracy: \t\t\t{:.2f} %".format(np.mean(val_accuracy[-2468 // batch_size :]) * 100),win='accuracy_log')
    vis.line(X=[epoch],Y=[np.mean(train_loss[-9840 // batch_size :])],win='loss',update='append')
    vis.line(X=[epoch],Y=[np.mean(val_accuracy[-2468 // batch_size :]) * 100],win='accuracy',update='append')
# In[ ]:
    f = open("log.txt", "a")
    f.write("Epoch {} of {} took {:.3f}s /n".format(                                                 epoch + 1, num_epochs, time.time() - start_time))
    f.write("  training loss (in-iteration): \t{:.6f}/n".format(                                    np.mean(train_loss[-9840 // batch_size :])))
    f.write("  validation accuracy: \t\t\t{:.2f} % /n".format(                                       np.mean(val_accuracy[-2468 // batch_size :]) * 100))
    f.close()




