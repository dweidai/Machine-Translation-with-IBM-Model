#!/usr/bin/env python
# coding: utf-8

# In[87]:


import os
import numpy as np
english_words = []
spanish_words = []
english_index={}
spanish_index={}
count = {}
base_t=[]

# # Data Preprocess

# In[88]:


def read_english(filename):
    file = open(filename, encoding='utf-8')
    toReturn = []
    for phrase in file:
        toAdd = phrase.split()
        toAdd.insert(0, "*")
        toReturn.append(toAdd)
    return toReturn

def read_spanish(filename):
    file = open(filename, encoding='utf-8')
    toReturn = []
    toRemove = []
    for phrase in file:
        toReturn.append(phrase.split())
    i = 0
    for phrase in toReturn:
        if len(phrase) == 0:
            toRemove.append(i)
        i += 1
    return toReturn, toRemove

print("\nReading in corpus data...\n")
english = read_english("corpus.en")
spanish, remove = read_spanish("corpus.es")


# In[89]:


print("Removing null phrases...\n")
for none in sorted(remove, reverse=True):
    del english[none]
    del spanish[none]
print("English dataset is " + str(len(english)))
print("Spanish dataset is " + str(len(spanish)))


# In[101]:


print("Collecting words and indices...\n")
for i in range(len(english)):
    for eword in english[i]:
        if eword not in english_words:
            english_words.append(eword)
    for sword in spanish[i]:
        if sword not in spanish_words:
            spanish_words.append(sword)
    for k in range(len(english_words)):
        english_index[english_words[i]] = k
    for k in range(len(spanish_words)):
        spanish_index[spanish_words[i]] = k
print("Spanish words are " + str(len(spanish_words)))
print("English words are " + str(len(english_words)))


# # IBM Model 1

# In[102]:


print("Generating first generation t numbers...\n")
parallel = zip(english, spanish)
visited = set()
for f, s in parallel:
    for f_j in f:
        for s_i in s:
            pair = (f_j, s_i)
            if pair not in visited:
                visited.add(pair)
                if not f_j in count:
                    count[f_j] = 0
                count[f_j] += 1
                
for word in english_words:
    base_t.append(1/ count[word])


# In[103]:


import scipy.sparse as sp
t_params = sp.dok_matrix((len(english_words), len(spanish_words)), dtype=np.float32)

def delta(i, k):
    spanish_word = spanish[k][i];
    num = []
    summation=0
    for word in range(len(english[k])):
        english_word= english[k][word]
        p_1 = english_index[english_word]
        p_2 = spanish_index[spanish_word]
        temp = t_params[p_1, p_2]
        if temp == 0:
            temp_index = english_index[english_word]
            temp= base_t[temp_index]
        summation += temp
        num.append(temp)
    return np.array(num)/summation

def t(counts):
    r = counts.sum(1)
    #print(r)
    return counts/r


# In[104]:


import scipy.sparse as sp
print("Generating t parameters...\n")
t_params = sp.dok_matrix((len(english_words), len(spanish_words)), dtype=np.float32)
n = 5
for iteration in range(n):
    print(str(iteration) + " out of " + str(n) + " iterations")
    temp = sp.dok_matrix((len(english_words), len(spanish_words)), dtype=np.float32)
    for k in range(len(english)): 
        for i in range(len(spanish[k])):
            d = delta(i, k)
            f_w = spanish[k][i]
            f_i = spanish_index[f_w]
            for j in range(len(english[k])):
                e_w = english[k][j]
                e_i = english_index[e_w]
                temp[e_i, f_i] += d[j]
    t_params=t(temp)


# In[105]:


print("Saving the t parameters...\n")
#print(type(t_params))
#print(t_params.shape)
np.savetxt('t_params.txt', t_params, delimiter=',')


# # Evaluation

# In[106]:


print("Loading dev data corpus...\n")
english_file = open('dev.en', encoding='utf-8')
spanish_file = open('dev.es', encoding='utf-8')
english_dev = []
spanish_dev = []
for phrase in english_file:
    english_dev.append(phrase.split())
for phrase in spanish_file:
    spanish_dev.append(phrase.split())


# In[107]:


outfile = open("out.key", "w", encoding='utf-8')


# In[110]:


print("Start generating key with dev data...\n")
for i in range(len(s_dev)):
    for s_i_dev,s_w_dev in enumerate(s_dev[i],1):
        if s_w_dev in spanish_words:
            max = 0
            s_index = spanish_index[s_w_dev]
            predicted_index=0
            for e_i_dev,e_w_dev in enumerate(e_dev[i],1):
                e_index= english_index[e_w_dev]
                if t_params[e_index,s_index]==0:
                    value = base_t[e_index]
                else:
                    value = t_params[e_index,s_index]
                if value > max:
                    max= value;
                    predicted_index = e_i_dev
            outfile.write(str(i+1)+' '+str(predicted_index)+' '+str(s_i_dev)+'\n')


# In[111]:


outfile.close()
print("Evaluating...\n")
os.system("python eval_alignment.py dev.key out.key")


# In[112]:


#!python eval_alignment.py 'dev.key' 'out.key'


# In[ ]:




