#!/usr/bin/env python
# coding: utf-8

# In[52]:


import os
import numpy as np
import scipy.sparse as sp
import copy
english_words = []
spanish_words = []
english_indices={}
spanish_indices={}
count = {}
base_t=[]
visited = {}
visited_e = {}
visited_s = {}

# In[53]:


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

print("Reading in corpus data...\n")
english = read_english("corpus.en")
spanish, remove = read_spanish("corpus.es")


# In[54]:


print("Removing empty phrases...\n")
for none in sorted(remove, reverse=True):
    del english[none]
    del spanish[none]
print("English dataset is " + str(len(english)))
print("Spanish dataset is " + str(len(spanish)))


# In[55]:


print("Collecting words and indices...\n")
for i in range(len(english)):
    for eword in english[i]:
        if eword not in english_words:
            english_words.append(eword)
    for sword in spanish[i]:
        if sword not in spanish_words:
            spanish_words.append(sword)
    for i in range(len(english_words)):
        english_indices[english_words[i]]=i
    for i in range(len(spanish_words)):
        spanish_indices[spanish_words[i]]=i    
print("Spanish words are " + str(len(spanish_words)))
print("English words are " + str(len(english_words)))


# In[56]:


print("Reading t numbers from IBM Model 1...\n")
filename = "t_params.txt"
t_params = np.loadtxt(filename, delimiter=',')


# In[57]:


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
                count[f_j]+=1
                
for i in english_words:
    base_t.append(1/ count[i] )


# In[78]:


print("Generating q parameters...\n")
for j in range(len(spanish)):
    l=len(english[j])
    m=len(spanish[j])
    if (l,m) not in visited.keys():
        visited_e[(l,m)]= {}
        visited_s[(l,m)]= {}
        s_x = len(set(english[j]))
        s_y = len(set(spanish[j]))
        visited[(l,m)] = sp.dok_matrix((s_x, s_y), dtype=np.float32)
        for i in range(len(set(english[j]))):
            visited_e[(l,m)][list(set(english[j]))[i]] = i
        for i in range(len(set(spanish[j]))):
            visited_s[(l,m)][list(set(spanish[j]))[i]] = i
    else:
        l_e=list(set(english[j]))
        l_s=list(set(spanish[j]))
        for i in range(len(l_e)):
            assigned = visited_e[(l,m)]
            if l_e[i] not in assigned.keys():
                assigned[l_e[i]]=len(assigned.keys())
        for i in range(len(l_s)):
            if l_s[i] not in visited_s[(l,m)].keys():
                visited_s[(l,m)][l_s[i]]= len(visited_s[(l,m)].keys())
        x = len(visited_e[(l,m)].keys())
        y = len(visited_s[(l,m)].keys())
        visited[(l,m)]=sp.dok_matrix((x, y), dtype=np.float32)
base_q=copy.deepcopy(visited)


# In[79]:


def delta(i, k):
    s_word = spanish[k][i]
    count=[]
    l = len(english[k]) 
    sum = 0
    m = len(spanish[k])
    for x in range(l):
        e_word=english[k][x]
        t = t_params[english_indices[e_word],spanish_indices[s_word]]
        e = q_e[(l,m)][e_word]
        s = q_s[(l,m)][s_word]
        t_q= q[(l,m)][e,s]
        if t_q==0:
            t_q= 1/(l)          
        if t==0:
            t= base_t[english_indices[e_word]]
        sum += (t*t_q)
        count.append(t*t_q)
    return np.array(count)/sum


# In[80]:


def q_f(counts_for_q, q):
    for x in range(len(list(counts_for_q.keys()))):
        (l, m)=list(counts_for_q.keys())[x]
        q[(l, m)]=counts_for_q[(l, m)]/(counts_for_q[(l, m)].sum(1))
    return q

def t(counts):
    r = counts.sum(1)
    #print(r)
    return counts/r


# In[81]:


print("Generating t and q parameters...\n")
n = 5
for s in range(n):
    print(str(s) + " out of " + str(n) + " iterations")
    counts = sp.dok_matrix((len(english_words), len(spanish_words)), dtype=np.float32)
    c_q = copy.deepcopy(base_q)
    for k in range(len(english)): 
        print("\t" + str(k) + " out of " + str(len(english)))
        l = len(english[k])
        m = len(spanish[k])
        for i in range(len(spanish[k])):
            delta = delta(i, k)
            s_w = spanish[k][i]
            s_i = spanish_indices[s_w]
            for j in range(len(english[k])):
                e_w = english[k][j]
                e_i = english_indices[e_w]
                counts[e_i, s_i] += delta[j]
                x = q_e[(l,m)][e_w]
                y = q_s[(l,m)][s_w]
                c_q[(l, m)][x,y] += delta[j]
    t_params=t(counts)
    q = q_f(c_q, q)


# In[82]:


print("Loading dev data corpus...\n")
english_file = open('dev.en', encoding='utf-8')
spanish_file = open('dev.es', encoding='utf-8')
e_dev = []
s_dev = []
for phrase in english_file:
    e_dev.append(phrase.split())
for phrase in spanish_file:
    s_dev.append(phrase.split())


# In[83]:


outfile=open("dev.out", "w", encoding='utf-8')



# In[84]:


outfile.close()
print("Evaluating...\n")
os.system("python eval_alignment.py dev.key dev.out")
#!python eval_alignment.py 'dev.key' 'dev.out'


# In[ ]:




