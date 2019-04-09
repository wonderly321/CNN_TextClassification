
# coding: utf-8

# In[1]:


with open('test.txt', 'r') as f:
    data = f.readlines()
testlabels=[dat.split('\t')[0] for dat in data]
testdata=[dat.split('\t')[1] for dat in data]
with open('train.txt', 'r') as f:
    data = f.readlines()
trainlabels=[dat.split('\t')[0] for dat in data]
traindata=[dat.split('\t')[1] for dat in data]
labels=testlabels+trainlabels
data=testdata+traindata


# ### extract feature

# In[2]:


MAX_SEQUENCE_LENGTH = 50 # 每条新闻最大长度
EMBEDDING_DIM = 200 # 词向量空间维度
VALIDATION_SPLIT = 0.16 # 验证集比例
TEST_SPLIT = 0.2 # 测试集比例

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import tensorflow as tf

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np

tokenizer = Tokenizer()
tokenizer.fit_on_texts(data)
sequences = tokenizer.texts_to_sequences(data)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)


# In[3]:


p1 = int(len(data)*(1-VALIDATION_SPLIT-TEST_SPLIT))
p2 = int(len(data)*(1-TEST_SPLIT))
x_train = data[:p1]
y_train = labels[:p1]
x_val = data[p1:p2]
y_val = labels[p1:p2]
x_test = data[p2:]
y_test = labels[p2:]
print('train docs: '+str(len(x_train)))
print('val docs: '+str(len(x_val)))
print('test docs: '+str(len(x_test)))


# In[4]:


from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Sequential


model = Sequential()
model.add(Embedding(len(word_index) + 1, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
model.add(Dropout(0.2))
model.add(Conv1D(250, 3, padding='valid', activation='relu', strides=1))
model.add(MaxPooling1D(3))
model.add(Flatten())
model.add(Dense(EMBEDDING_DIM, activation='relu'))
model.add(Dense(labels.shape[1], activation='softmax'))
#model.summary()

model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['acc'])
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=2, batch_size=128)
y_test_predict=model.predict(x_test)
y_predict=[np.argmax(y_test_predict[i,:]) for i in range(4385)]
#print(y_predict)
y_test_1d=[np.argmax(y_test[i,:]) for i in range(4385)]
#print(y_test_1d)
from sklearn.metrics import classification_report
print(classification_report(y_test_1d, y_predict, target_names=['0','1','2','3','4','5']))
model.save('cnn_only.h5')
#print(model.evaluate(x_test, y_test))

