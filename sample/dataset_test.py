
# coding: utf-8

# In[1]:


from junno.j_utils import log
log


# ### Import

# In[2]:


import junno.datasets as D
import time
import multiprocessing as mp
import pickle
import numpy as np


# In[3]:


def f_wait(x):
    time.sleep(0.1)
    return x


# ## Image Dataset

# In[ ]:


PATH = '/home/gaby/Ecole/EPM/MaitriseGit/db/Gabriel/train/'


# In[ ]:


raw = D.images(PATH+'raw_full/', name='raw')
gt = D.images(PATH+'avgt_full/', name='gt')


# In[ ]:


join = D.join('name', name=raw.col.name, raw=raw.col.data, gt=gt.col.data)


# In[ ]:


labelled = join.as_label('gt', {1:'red', 2: 'blue'}, sampling=1/2)
labelled


# In[ ]:


RD = D.RandomDistribution
da = D.DataAugment().flip_horizontal().rotate().hue()
augmented = labelled.augment(da, N=5, original=True)
augmented


# In[ ]:


augmented.get_augmented(labelled.at(0))


# In[ ]:


stop


# ## Shuffle

# In[35]:


d = D.NumPyDataSet({'id':np.arange(0,100)})
s = d.shuffle()


# In[36]:


t = time.time()
gen = s.generator(n=5, determinist=False, intime=True) 
for i, r in enumerate(gen):
    print(i, r['id'])
print('%.2fs'%(time.time()-t))


# In[ ]:


s.columns.id.format = {'default': '', 0: '0', 1: '1', 2:'2', 3:'3', 4:'4', 5:'5', 6:'test', 7:'7', 8:'8', 9:'9', 10:'10'}


# In[ ]:


s.parent_datasets


# In[ ]:


s.size


# ## Standard Datasets

# In[ ]:


from junno.datasets import cifar10_datasets
c = cifar10_datasets()['test'].map(x1='x', y='y', x2='x')


# In[ ]:


c_dupli = c.apply('x_copy', lambda x1: x1[0], n_factor=None)


# In[ ]:


cache = c_dupli.cache(ondisk='test.hd5', overwrite=False)


# In[ ]:


cache.select('(y==1) | (y==2) | (y==3)').sort('y')


# In[ ]:


cache.mean(ncore=1, end=10)


# In[ ]:


cache.std('x_copy', end=2).shape


# In[ ]:


cache.std()


# ## Generator Trace

# In[ ]:


gen = out_long.generator(n=7)


# In[ ]:


r = gen.next()
r.trace.print()


# In[ ]:


out


# In[ ]:


c = out_long.as_cache(ncore=4)


# In[ ]:


pgen = out_long.generator(parallel_exec=True)


# In[ ]:


r.trace.print()


# In[ ]:


r = pgen.poll()

