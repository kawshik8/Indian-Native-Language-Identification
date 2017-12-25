# #import tensorflow
# # import keras

def get_bow():
    f = open("/Users/apple/temp/nltk/features.txt",'r')
    sentences = f.read().splitlines()
    
    bow = []
    l = []
    for sentence in sentences:
        #print(sentence)
        label = sentence.split("\t")[0]
        if label=='BEN':
            l.append(0)
        if label=='HIN':
            l.append(1)
        if label=='KAN':
            l.append(2)
        if label=='MAL':
            l.append(3)
        if label=='TAM':
            l.append(4)
        if label=='TEL':
            l.append(5)
        #print(sentence.split("\t")[1])
        line = sentence.split("\t")[1]
        tokens = line.split(" ")
        for word in tokens:
            if(len(word)>1):
                bow.append(word)
    bow = list(set(bow))
    #print(len(bow))
    f = open("finalbow.txt","w+")
    for word in bow:
        f.write("%s\n" % word)
    f.close()
    return bow,l

# #bow,_ = get_bow()
# #get_bow()

def fea_vect_ini(fea_vect,bow):
    for i in bow:
        fea_vect.append(0)
    return fea_vect


def fea_vect():
    f = open("/Users/apple/temp/nltk/features.txt",'r')
    sentences = f.read().splitlines()
    bow,_ = get_bow()
    
    fea_set = []
    l = []
    i = 0
    for sentence in sentences:
        bows = []
        #print(str(i))
        i+=1
        line = sentence.split("\t")[1]
        fea_vect = []
        fea_vect = fea_vect_ini(fea_vect,bow)
        #print(fea_vect)
        #print("")
        words = line.split(" ")
        for word in words:
            if(len(word)>1):
                bows.append(word)
        for word in bows:
            pos = bow.index(word)
            fea_vect[pos] = 1
        #print(fea_vect)
        fea_set.append(fea_vect)
    return fea_set,l
    fea_set.append(fea_vect)

# fea_vect()
y_train = []
bb,y_train = get_bow()
#print(len(bb),len(y_train))

# import pickle
#print("\n\nY:\n\n")
print("Y_train:",len(y_train))

x_train = []
x_train,_ = fea_vect()

print("X_TRAIN:",len(x_train),len(x_train[0]))

xy = [(c,b) for (c,b) in zip(x_train,y_train)]
xy = sorted(xy, key=lambda x: x[1])
#cntbb = sorted(cntbb, key=lambda x: x[1])
x_train = [x for x,y in xy]
y_train = [y for x,y in xy]
#print("XY:",xy[0],xy[1])
print("X_TRAIN:",len(x_train),len(x_train[0]))
print("Y_train:",len(y_train))
###############################################

import nltk
import pickle

def get_bow():
    f = open("/Users/apple/temp/nltk/features.txt",'r')
    sentences = f.read().splitlines()
    
    bow = []
    i=0
    for sentence in sentences:
        i+=1
        #print(sentence)
        #print(sentence.split("\t")[1])
        line = sentence.split("\t")[1]
        tokens = line.split(" ")
        for word in tokens:
            if(len(word)>1):
                bow.append(word)
    bow = list(set(bow))
    #print(len(bow))
    f = open("finalbow.txt","w+")
    for word in bow:
        f.write("%s\n" % word)
    f.close()
    return bow



def get_fea_test():
    f = open("/Users/apple/temp/nltk/featurestest.txt",'r')
    bow = get_bow()
    lines = f.read().splitlines()
    fea_set = []
    for line in lines:
        words = line.split(" ")
        bowtmp = []
        for word in words:
            bowtmp.append(word)
        fea_vect = []
        for word in bow:
            fea_vect.append(0)
        for word in bowtmp:
            if word in bow:
                pos = bow.index(word)
                fea_vect[pos] = 1
        fea_set.append(fea_vect)
    return fea_set

x_test = get_fea_test()
print("X_TEST:", len(x_test),len(x_test[0]))

##############################################

import keras
import pickle
from keras.models import Sequential
from keras.layers.core import Dense,Activation,regularizers,Dropout, Reshape
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.layers import Embedding, LSTM, Dense, Conv1D, MaxPooling1D, Dropout, Activation
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import keras.backend as K
import numpy as np
from numpy import zeros, newaxis
from keras import optimizers



# # print(len(x_test),len(x_test[0]))
# x_train = np.array(x_train)
# x_train = x_train.reshape(x_train.shape + (1,))

# model = Sequential()
########################################
# x_train = x_train[:, :, newaxis]
# print(x_train.shape)
# model = Sequential()
# model.add(Conv1D(64, 3, activation='relu', input_shape=x_train.shape[1:]))
# model.add(Conv1D(64, 3, activation='relu'))
# model.add(MaxPooling1D(3))
# model.add(Conv1D(128, 3, activation='relu'))
# model.add(Conv1D(128, 3, activation='relu'))
# model.add(GlobalAveragePooling1D())
# model.add(Dropout(0.5))
# model.add(Dense(6, activation='softmax'))
    
# model.compile(loss='categorical_crossentropy',
#               optimizer='rmsprop',
#               metrics=['accuracy'])
# model.fit(x_train, y_train, batch_size=16, epochs=30)
# #####################################
# import tflearn
# from tflearn.layers.core import input_data, dropout, fully_connected
# from tflearn.layers.conv import conv_2d, max_pool_2d
# from tflearn.layers.estimator import regression
# from tflearn.metrics import Accuracy
# acc = Accuracy()
# network = input_data(shape=[None, x_train[0], 1])
# # Conv layers
# network = conv_2d(network, 64, 3, strides=1, activation='relu', name = 'conv1_3_3_1')
# network = max_pool_2d(network, 2, strides=2)
# network = conv_2d(network, 64, 3, strides=1, activation='relu', name = 'conv1_3_3_2')
# network = conv_2d(network, 64, 3, strides=1, activation='relu', name = 'conv1_3_3_3')
# network = max_pool_2d(network, 2, strides=2)
# # Fully Connected Layer 
# network = fully_connected(network, 1024, activation='tanh')
# # Dropout layer
# network = dropout(network, 0.5)
# # Fully Connected Layer
# network = fully_connected(network, 10, activation='softmax')
# # Final network
# network = regression(network, optimizer='momentum',
#                      loss='categorical_crossentropy',
#                      learning_rate=0.001, metric=acc)
###################################################
# from keras.layers import Input, Dense, Embedding, merge, Convolution2D, MaxPooling2D, Dropout
# from sklearn.cross_validation import train_test_split
# from keras.layers.core import Reshape, Flatten
# from keras.callbacks import ModelCheckpoint
# from data_helpers import load_data
# from keras.optimizers import Adam
# from keras.models import Model



# #print 'Loading data'
# # x, y, vocabulary, vocabulary_inv = load_data()

# # X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=42)

# sequence_length = x.shape[1]
# vocabulary_size = len(x_train[0])
# embedding_dim = 256
# filter_sizes = [3,4,5]
# num_filters = 512
# drop = 0.5

# nb_epoch = 100
# batch_size = 30

# # this returns a tensor
# inputs = Input(shape=(sequence_length,), dtype='int32')
# embedding = Embedding(output_dim=embedding_dim, input_dim=len(x_train[0]), input_length=sequence_length)(inputs)
# reshape = Reshape((sequence_length,embedding_dim,1))(embedding)

# conv_0 = Convolution2D(num_filters, filter_sizes[0], embedding_dim, border_mode='valid', init='normal', activation='relu', dim_ordering='tf')(reshape)
# conv_1 = Convolution2D(num_filters, filter_sizes[1], embedding_dim, border_mode='valid', init='normal', activation='relu', dim_ordering='tf')(reshape)
# conv_2 = Convolution2D(num_filters, filter_sizes[2], embedding_dim, border_mode='valid', init='normal', activation='relu', dim_ordering='tf')(reshape)

# maxpool_0 = MaxPooling2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1), strides=(1,1), border_mode='valid', dim_ordering='tf')(conv_0)
# maxpool_1 = MaxPooling2D(pool_size=(sequence_length - filter_sizes[1] + 1, 1), strides=(1,1), border_mode='valid', dim_ordering='tf')(conv_1)
# maxpool_2 = MaxPooling2D(pool_size=(sequence_length - filter_sizes[2] + 1, 1), strides=(1,1), border_mode='valid', dim_ordering='tf')(conv_2)

# merged_tensor = merge([maxpool_0, maxpool_1, maxpool_2], mode='concat', concat_axis=1)
# flatten = Flatten()(merged_tensor)
# # reshape = Reshape((3*num_filters,))(merged_tensor)
# dropout = Dropout(drop)(flatten)
# output = Dense(output_dim=6, activation='softmax')(dropout)

# # this creates a model that includes
# model = Model(input=inputs, output=output)

# checkpoint = ModelCheckpoint('weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
# adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

# model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
# model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, callbacks=[checkpoint])  # starts training
##########################################
# # model = Sequential()
# # model.add(Embedding(max_features, 128, input_length=maxlen))
# # model.add(Bidirectional(LSTM(64)))
# # model.add(Dropout(0.5))
# # model.add(Dense(1, activation='sigmoid'))

# # # # try using different optimizers and different optimizer configs
# # # model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

# # # print('Train...')
# # # model.fit(x_train, y_train,
# # #           batch_size=batch_size,
# # #           epochs=4)
###################################
# model = Sequential()
# model.add(Embedding(12067, 128, input_length=300))
# model.add(LSTM(len(x_train[0]), dropout=0.2, recurrent_dropout=0.2))
# model.add(Dense(1, activation='relu'))
# model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#########################################
# model = Sequential()
# model.add(Embedding(len(x_train[0]), 128, input_length=len(x_train[0]))
# model.add(Dropout(0.2))
# model.add(Conv1D(64, 5, activation='relu'))
# model.add(MaxPooling1D(pool_size=4))
# model.add(LSTM(128))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.fit(data, np.array(balanced_labels), validation_split=0.5, epochs=3)
#####################################
# # # sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
# # # embedded_sequences = embedding_layer(sequence_input)
# # # l_lstm = Bidirectional(LSTM(100))(embedded_sequences)
# # # preds = Dense(2, activation='softmax')(l_lstm)
# # # model = Model(sequence_input, preds)
# # # model.compile(loss='categorical_crossentropy',
# # #               optimizer='rmsprop',
# # #               metrics=['acc'])

# # # print("model fitting - Bidirectional LSTM")
# # # model.summary()
# # # model.fit(x_train, y_train, validation_data=(x_val, y_val),
# # #           nb_epoch=10, batch_size=50)

# # ###################################
model = Sequential()
model.add(Dense(units=64, input_dim=len(x_train[0]), activation='relu'))
model.add(Dropout(0.2))
# model.add(Activation('relu'))
# model.add(Activation(''))
# model.add(Dense(units=32, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros',kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01)))
model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.2))
# model.add(Dense(units=64, activation='relu'))
# model.add(Activation('tanh'))
model.add(Dense(6, activation='softmax'))




model.compile(loss='sparse_categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
#model.compile(loss=keras.losses.binary_crossentropy,optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))

model.fit(x_train, y_train, epochs=100, batch_size=32)
model.save('/Users/apple/temp/nltk/my_modelnn.h5')
# del model
# # #model.train_on_batch(x_batch, y_batch)
# # # loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
# # # classes = model.predict(x_test, batch_size=128)

# # #print(classes)

from keras.models import load_model

model = load_model('/Users/apple/temp/nltk/my_modelnn.h5')

# x_test = np.array(x_test)
# x_test = x_test.reshape(x_test.shape + (1,))
#print(x_test.shape)

y = model.predict(x_test,batch_size=128)
for line in y:
    line = list(line)[0]

y = y.tolist()
labels = []
for row in y:
    maxe = max(row)
    pos = row.index(maxe)
    if(pos==0):
        labels.append("BE")
    elif(pos==1):
        labels.append("HI")
    elif(pos==2):
        labels.append("KA")
    elif(pos==3):
        labels.append("MA")
    elif(pos==4):
        labels.append("TA")
    elif(pos==5):
        labels.append("TE")

print(labels)

#########################################

import pickle
import os

def read_files():
    print(len(labels))
    path =  "/Users/apple/temp/nltk/INLI2017-TEST"
    files = os.listdir("/Users/apple/temp/nltk/INLI2017-TEST")
    
    #print(directories[0])
    i = 0
    f = open("/Users/apple/temp/nltk/output1.txt",'w+')
    for file in files:
        print(file,labels[i])
        f.write("%s\t" %file)
        f.write("%s\n" %labels[i])
        
        # print(file,labels[i])
        i+=1

read_files()

