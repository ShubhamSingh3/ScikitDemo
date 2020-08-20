#!/usr/bin/env python
# coding: utf-8

# # Lab Session 1: Machine Learning

# ### Define a task for classification

# # Load the dataset and Preprocess

# In[1]:


f=open('Dataset/sentiment.txt','r')
sentences=f.readlines()
f.close()


# In[7]:


print(sentences[500])


# # Feature Extraction
# Features can be list Unigrams or bigrams or trigrams ....  or combine
# 
# We are considering unigram

# In[3]:


import re
import numpy as np

stopwords = ["a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the"]

def text_cleaner(text): 
    text=remove_link(text.lower())
    long_words=[]
    for i in text.split():
        if i not in stopwords:                  
            long_words.append(i)
    return long_words

def remove_link(text):
    regex = r'https?://[^\s<>)"‘’]+'
    match = re.sub(regex,' ', text)
    regex = r'https?:|urls?|[/\:,-."\'?!;…]+'
    tweet = re.sub(regex,' ', match)
    tweet = re.sub("[^a-zA-Z_]", " ", tweet)
    tweet = re.sub("[ ]+", " ", tweet) 
    return tweet


# In[36]:


vocabulary=[]     # Features                  
samples=[]        # List of samples
y=[]              # Class Label
for line in sentences:
	texts=line.strip().lower().split('\t')
	tokens=text_cleaner(texts[0])
	y.append(int(texts[1]))
	samples.append(tokens)
	for word in tokens:
		vocabulary.append(word)


# In[5]:


print("Number of features is :"+str(len(vocabulary)))


# # Represent samples using vector space model
# 
# Each sample will look like
#  
#     f1, f2, f3,      fk
# 1. [w11, w12, w13m ....., w1k] c1
# 2. [w21, w22, w23m ....., w2k] c2
# ....
# 
# In this example, we have considered boolean vector space model

# In[6]:


def t2v(tokens,attributes):
    vect=[]
    for feature in attributes:
        if feature in tokens:
            vect.append(1)         #Presence
        else:
            vect.append(0)         #Absence
    return vect


# In[37]:


vector=[]
for i,tokens in enumerate(samples):
    vect=t2v(tokens,vocabulary)
    vector.append((vect,y[i],i)) #vector representation, label, sentence no.
    
i=2
print("Sentence No. \t Vector Representation \t label")
print(str(vector[i][2])+" "+str(vector[i][0])+" "+str(vector[i][1]))


# # Splitting the dataset for training and testing

# In[8]:


import random
random.shuffle(vector)

X=[]
y=[]
s_idx=[]
for i in vector:
    X.append(i[0]) #vector representation
    y.append(i[1]) #label
    s_idx.append(i[2]) #sentence_number
X=np.asarray(X)
y=np.asarray(y)
s_idx=np.asarray(s_idx)
print("Number of samples and number of Features")
X.shape


# In[9]:


del vector,samples

import gc
gc.collect()


# In[10]:


from sklearn.model_selection import KFold
kf = KFold(n_splits=10)
folds=[]
for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]   #sentence vector
    y_train, y_test = y[train_index], y[test_index]   #label
    s_train, s_test = s_idx[train_index], s_idx[test_index]   #sentence index
    folds.append((X_train, X_test, y_train, y_test,s_train, s_test))


# ## 1st fold

# In[11]:


(X_train, X_test, y_train, y_test,s_train, s_test)=folds[0]
print("Number of training samples :"+str(len(X_train)))
print("number of test data :"+str(len(X_test)))


# ## Building a classifier

# In[12]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import NearestCentroid
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression


# # Naive bayes Classifier with Multinomial Kernal

# In[13]:


mnb=MultinomialNB()
model = mnb.fit(X_train,y_train)

y = model.predict(X_test)
i=0
print("Input Data => Predicted => Actual")
for p in y:
    print(X_test[i]," =>" , p," =>"  ,y_test[i])
    i+=1


# In[51]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


# In[38]:


print(y_test)
pred0=[]
true0=[]
pred1=[]
true1=[]

for i,ii in enumerate(y_test):
    if ii==1:
        true1.append(ii)
        pred1.append(y[i])
    else:
        true0.append(ii)
        pred0.append(y[i])

pred=pred0+pred1
true=true0+true1
print(true)
del true0,true1,pred0,pred1


# In[56]:


print("Accuracy ",accuracy_score(y_test, y, normalize = True))
print("Precision ",precision_score(y_test, y, average = None))
print("Recall ",recall_score(y_test, y, average = None))
print("F1 Score ",f1_score(y_test, y, average = None))
print("AUC Score ", roc_auc_score(true, pred))


# In[40]:


test_ip="I love watching netflix original movies"
tokens=text_cleaner(test_ip)
vect_pos=t2v(tokens,vocabulary)

test_ip="I hate watching netflix original movies" #You have won 1 crore rupees #India ka dusra nam he Bharat
tokens=text_cleaner(test_ip)
vect_neg=t2v(tokens,vocabulary)


# In[41]:


y = model.predict([vect_pos,vect_neg])
y


# # Naive Bayes with Gaussian kernal

# In[21]:


gnb= GaussianNB()
model = gnb.fit(X_train,y_train)

y = model.predict(X_test)

print("Accuracy ",accuracy_score(y_test, y, normalize = True))


# In[22]:


cc=NearestCentroid()
model = cc.fit(X_train,y_train)

y = model.predict(X_test)

print("Accuracy ",accuracy_score(y_test, y, normalize = True))


# In[23]:


dtc =DecisionTreeClassifier()
model = dtc.fit(X_train,y_train)

y = model.predict(X_test)

print("Accuracy ",accuracy_score(y_test, y, normalize = True))


# In[24]:


knnc=KNeighborsClassifier(n_neighbors=15)
model = knnc.fit(X_train,y_train)

y = model.predict(X_test)

print("Accuracy ",accuracy_score(y_test, y, normalize = True))


# In[25]:


svml=svm.LinearSVC()
model = svml.fit(X_train,y_train)

y = model.predict(X_test)

print("Accuracy ",accuracy_score(y_test, y, normalize = True))


# In[26]:


clf = MLPClassifier(solver='sgd', hidden_layer_sizes=(10,), activation="relu")
model = clf.fit(X_train,y_train)

y = model.predict(X_test)

print("Accuracy ",accuracy_score(y_test, y, normalize = True))


# In[27]:


cLR = LogisticRegression()
model = cLR.fit(X_train,y_train)

y = model.predict(X_test)

print("Accuracy ",accuracy_score(y_test, y, normalize = True))


# In[28]:


clf = RandomForestClassifier()
model = clf.fit(X_train,y_train)

y = model.predict(X_test)

print("Accuracy ",accuracy_score(y_test, y, normalize = True))


# In[29]:


cLR = LinearRegression()
model = cLR.fit(X_train,y_train)

y = model.predict(X_test)

for i,p in enumerate(y):
    print(p, y_test[i])



# In[44]:


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=0).fit(X_train)
kmeans.labels_


# In[45]:


y=kmeans.predict(X_test)


# In[46]:


kmeans.predict([vect_neg])


# In[47]:


i=0
test_ip=sentences[s_idx[i]].strip().split('\t')[0]
tokens=text_cleaner(test_ip)
vect_sem16=t2v(tokens,vocabulary)
kmeans.predict([vect_sem16])


# In[49]:


from sklearn.metrics.cluster import adjusted_rand_score, homogeneity_score, completeness_score, v_measure_score


# In[50]:


print("Rand Index score ",adjusted_rand_score(y_test, y))
print("Homogeneity score ",homogeneity_score(y_test, y))
print("Completeness score ",completeness_score(y_test, y))
print("V measure score ",v_measure_score(y_test, y))


# In[ ]:




