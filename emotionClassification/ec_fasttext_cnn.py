
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


# In[2]:


MAX_SEQUENCE_LENGTH = 50 # 每条新闻最大长度
EMBEDDING_DIM = 300 # 词向量空间维度
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


# In[3]:


VECTOR_DIR = '../fasttext/wiki.zh.vec' # 词向量模型文件

from keras.utils import plot_model
from keras.layers import Embedding
from gensim.models import FastText
import io
def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data
    
fasttext_model = load_vectors(VECTOR_DIR)
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))

for word, i in word_index.items(): 
    if word in fasttext_model:
        l = [i for i in fasttext_model[word]]
        embedding_matrix[i] = np.asarray(l,dtype='float32')
embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)


# In[4]:


from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Sequential

model = Sequential()
model.add(embedding_layer)
model.add(Dropout(0.2))
model.add(Conv1D(250, 3, padding='valid', activation='relu', strides=1))
model.add(MaxPooling1D(3))
model.add(Flatten())
model.add(Dense(EMBEDDING_DIM, activation='relu'))
model.add(Dense(labels.shape[1], activation='softmax'))
#model.summary()
#plot_model(model, to_file='model.png',show_shapes=True)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])
model.optimizer.lr.assign(0.01)
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=5, batch_size=128)
y_test_predict=model.predict(x_test)
y_predict=[np.argmax(y_test_predict[i,:]) for i in range(4385)]
#print(y_predict)
y_test_1d=[np.argmax(y_test[i,:]) for i in range(4385)]
#print(y_test_1d)
from sklearn.metrics import classification_report
print(classification_report(y_test_1d, y_predict, target_names=['0','1','2','3','4','5']))
model.save('ec_fasttext_cnn.h5')
#print(model.evaluate(x_test, y_test))

