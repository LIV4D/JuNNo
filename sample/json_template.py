
# coding: utf-8

# In[1]:


from junno.j_utils import log
log


# ### Import

# In[2]:


from junno.j_utils.json_template import JSONClass, JSONAttr


# ## Image Dataset

# In[20]:


class A(JSONClass):
    __template__ = '{"constante": 1, "groupa": {"suba": "{{a}}", "subconst": 0}, "b": "{{b}}"}'
    a = JSONAttr.String()
    b = JSONAttr.Float(list=True)
    
class C(JSONClass):
    __template__ = '{"pi": 3.14159, "a": "{{a}}"}'
    a = JSONAttr.List((A, str))


# In[4]:


a = A()


# In[5]:


a.a = 54
a


# In[6]:


a['groupa']['suba'] = 7


# In[7]:


a.b = [1, 5.0, '8']
a


# In[21]:


c = C()
c


# In[22]:


c.a.append(a)
c

