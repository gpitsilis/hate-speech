from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from keras.optimizers import Adam

from keras.layers import Dropout
from keras.layers import LSTM
from keras.preprocessing import sequence
from keras.models import model_from_json
from sklearn import cross_validation

from sklearn.model_selection import train_test_split

from keras.regularizers import l2, l1,l1l2

from keras import regularizers

from sklearn.model_selection import StratifiedKFold

from keras.models import load_model

import sys
import os
import random
from sklearn.feature_extraction.text import CountVectorizer

import numpy as np

from subprocess import Popen, PIPE

from keras import callbacks

import ht_lib
import tok_lib

tokenizer_cmd = tok_lib.init_tokenizer()


load_none_train         =  10889  # no limitation. used the complete dataset
load_racism_train       =  1943
load_sexism_train       =  3166

corpus = []  

ids = []

uIDs = []

users_lkp = {}

# ------------ users tweeting behaviour as a feature ----------------

def add_user_features_3(X_corpus):

    for i in range(0, len(X_corpus)):
        content = a_uIDs[i]

        ntrl  = lookup_user_prof_data(content[0], 'none')
        racsm = lookup_user_prof_data(content[0], 'racism')
        sexsm = lookup_user_prof_data(content[0], 'sexism')

        X_corpus[i].append(int(ntrl * 1000))  # neutral
        X_corpus[i].append(int(racsm * 1000))  # racism
        X_corpus[i].append(int(sexsm * 1000))  # sexism

    return X_corpus

# --------------- user ID s dictionary ---------------

# returns the short ID of a user giver the original user id from a tweet
def lookup_user(id):

    try:
        uid = users_lkp[id]
        return (uid)
    except KeyError:
        # create a new one if not exist
        newID = len(users_lkp) + 1
        users_lkp[id] = newID

        return newID


print "Loading userIDs into the dictionary ....."

uID_dict = {}

input_file = 'hate_speech_tweets_users.csv'

# read values and save after rescale
leg_inpt = open(input_file, "r+")

line = leg_inpt.readline()

while (line != ''):
    if (line == '\n'):
        print "Ran out of userIDs !"
        sys.exit(1)

    # print line
    fields = line.split(",")
    tid = fields[0]
    uid = fields[1]
    uid = uid.rstrip('\n')

    luid = lookup_user(uid)
    if (luid < 0):
        sys.exit(2) # Oops something is going wrong here

    uID_dict[tid] = luid

    line = leg_inpt.readline()

# ----------------- load user prof data --------------

user_prof_data = {}


input_file = 'user_class_ratio.csv'
leg_inpt = open(input_file, "r+")

line = leg_inpt.readline()

while (line != ''):
    if (line == '\n'):
        print "Ran out of user feature profiles !"
        sys.exit(1)

    # print line
    fields = line.split(",")
    uid     = fields[0]
    # lookup in user_dict to get the short ID
    uid = lookup_user(uid)

    classID = fields[1]
    value   = fields[2].rstrip('\n')

    user_prof_data[uid,classID] = value

    line = leg_inpt.readline()

# -------------------------------------------------------

def lookup_user_prof_data(userID,classN):

    try:
        value = float(user_prof_data[userID,classN])
        return (value)
    except KeyError:
        # create a new one if does not exist
        return float(0)

# ----------------- tokenize function ----------------
def tokenize(sentences):
    print ('Tokenizing..')

    text = "\n".join(sentences)

    tokenizer = Popen(tokenizer_cmd, stdin=PIPE, stdout=PIPE)

    tok_text, _ = tokenizer.communicate(text)

    toks = tok_text.split('\n')[:-1]

    print ('Done..')

    return toks

# ----------------- build dictionary ------------------

def build_dict(corp):
    tok_corpus = tokenize(corp)
    
    print ('Building dictionary..')
    wordcount = dict()
    for ss in tok_corpus:   # for all sentences
            words = ss.strip().lower().split()
            
            for w in words:  # for all words in a sentence
                if w not in wordcount:
                    wordcount[w] = 1
                else:
                    wordcount[w] += 1

    counts = wordcount.values()
    keys = wordcount.keys()

    # sorting the wordcount array by frequency

    sorted_idx = np.argsort(counts)[::-1]
    worddict = dict()

    for idx, ss in enumerate(sorted_idx):
            worddict[keys[ss]] = idx+2

    print (np.sum(counts), ' total words ', len(keys), ' unique words')

    return(worddict)



# This is for making use of the neutral and racism feature of users
def add_user_features_2b(X_corpus):

    for i in range(0, len(X_corpus)):
        content = a_uIDs[i]

        ntrl   = lookup_user_prof_data(content[0], 'none')
        sexism = lookup_user_prof_data(content[0], 'sexism')

        X_corpus[i].append(int(ntrl * 1000))  # neutral
        X_corpus[i].append(int(sexism * 1000))  # sexism
        
    return X_corpus


# This is for making use of the neutral and racism feature of users
def add_user_features_2a(X_corpus):

    for i in range(0, len(X_corpus)):
        content = a_uIDs[i]

        ntrl  = lookup_user_prof_data(content[0], 'none')
        racsm = lookup_user_prof_data(content[0], 'racism')

        X_corpus[i].append(int(ntrl * 1000))  # neutral
        X_corpus[i].append(int(racsm * 1000))  # racism
        

    return X_corpus

# This is for making use of the racism and sexism feature of users
def add_user_features_2(X_corpus):

    for i in range(0, len(X_corpus)):
        content = a_uIDs[i]

        racsm = lookup_user_prof_data(content[0], 'racism')
        sexsm = lookup_user_prof_data(content[0], 'sexism')

        X_corpus[i].append(int(racsm * 1000))  # racism
        X_corpus[i].append(int(sexsm * 1000))  # sexism

    return X_corpus

# -----------------------------------------------------------

def grab_data(corp, dictionary):

# input: data, dictionary
# output:sentences in the form of word frequency

    sentences = []

    for ff in corp:
        sentences.append(ff.strip())

    sentences = tokenize(sentences)

    seqs = [None] * len(sentences)
    for idx, ss in enumerate(sentences):
        words = ss.strip().lower().split()
        # words: is a list containing the words in the sentence.
        seqs[idx] = [dictionary[w] if w in dictionary else 1 for w in words]

    return seqs

# ------------------------------------------------------------


cmd_args = len(sys.argv)


if (cmd_args<2):
   print "Not enough arguments."
   sys.exit(0) 

else:
  if (cmd_args<3):
    file_id = "0"
     
  else:
     file_id = sys.argv[2]   

  if (sys.argv[1]=='O'):
     print "No extra features will be included"
     # no features
     output_filename = 'unigrams_relu.csv'
     # limit sentences to 30 words.
     max_words = 30
     FTRS = "NO"

  if (sys.argv[1]=='RS'):
     print "Including Racism and Sexism features"
     # limit sentences to 30+2 words.
     max_words = 32
     output_filename = 'unigrams_RS_relu.csv'
     FTRS = "RS"

  if (sys.argv[1]=='NR'):
     print "Including Neutral and Racism features"
     max_words = 32
     output_filename = 'unigrams_NR_relu.csv'
     FTRS = "NR"

  if (sys.argv[1]=='NRS'):
     print "Including Neutral, Racism and Sexism features"
     # limit sentences to 30+3 words.
     max_words = 33
     output_filename = 'unigrams_NRS_relu.csv'
     FTRS = "NRS"

  if (sys.argv[1]=='NS'):
     print "Including Neutral and Sexism features"
     # limit sentences to 30+2 words.
     max_words = 32
     output_filename = 'unigrams_NS_relu.csv'
     FTRS = "NS"


output_filename = file_id + "_" + output_filename
open(output_filename, 'w').close()

ht_lib.load_neutral(load_none_train,corpus,ids,uID_dict, uIDs)
ht_lib.load_racism(load_racism_train,corpus,ids,uID_dict, uIDs)
ht_lib.load_sexism(load_sexism_train,corpus,ids,uID_dict, uIDs)

a_uIDs = np.array(uIDs).reshape(-1,1)

word_dict = build_dict(corpus)

# word_dict contains the frequency table for each word in the corpus in the form of tuples ( word, freq )

X_corpus = grab_data(corpus, word_dict)

# loading the labels to Y
sizeY = load_none_train + load_sexism_train + load_racism_train

# storing the labels into the profiles
y_train = np.zeros(((sizeY),1), dtype=int)
for i in range(0,load_none_train-1):
    y_train[i] = int(0)
for i in range(load_none_train,(load_none_train + load_racism_train)):
    y_train[i] = int(1)
for i in range((load_none_train + load_racism_train),sizeY):
    y_train[i] = int(2)

# this is for converting a single column of 3 distinct values into 3 columns of binary values.
from keras.utils import np_utils
y_corpus = np_utils.to_categorical(y_train, nb_classes=3)


if (FTRS=='RS'):
    # incorporating features for racism and sexism only
    print "Incorporating Racism and Sexism features..."
    X_corpus = add_user_features_2(X_corpus)

if (FTRS=='NR'):
    # incorporating features for racism and sexism only
    print "Incorporating Neutral and Racism features..."
    X_corpus = add_user_features_2a(X_corpus)

if (FTRS=='NRS'):
    # incorporating features for racism and sexism only
    print "Incorporating Neutral,Sexism and Racism features..."
    X_corpus = add_user_features_3(X_corpus)

if (FTRS=='NS'):
    # incorporating features for racism and sexism only
    print "Incorporating Neutral and Sexism features..."
    X_corpus = add_user_features_2b(X_corpus)

# padding 
X_corpus = sequence.pad_sequences(X_corpus, maxlen=(max_words))

dataset = np.append(X_corpus,y_corpus, axis=1)

print "dataset shape:",(dataset.shape)

# initialize the random number generator
seed = 71
np.random.seed(seed)

# the vocabulary size
top_words = 25000


# the vector dimension
vector_dimension = 30
lstm_size = 200

print "Train_X:", X_corpus

colID = dataset.shape[1]

print "colID:", colID

colID = dataset.shape[1]

# spliting the training data into X and Y
X = dataset[:,0:colID-3]
Y = dataset[:,colID-3:colID]

print "dataset:",dataset.shape
print "X:",X.shape
print "Y:",Y.shape

dm = Y.shape[0]
Y_1 = np.zeros(dm)

# convert the 3 classes into 1 column.
for i in range(0,Y.shape[0]):
    k = Y[i]
    if (k[0] == 1):
        Y_1[i] = 0
    else:
        if (k[1] == 1):
            Y_1[i] = 1
        else:
            Y_1[i] = 2


# shuffling data for 10-fold cross validation
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []

print "max_words:", max_words

fld = 0

############################################################
# this is for limiting the mem requirements when running 
# in a GPU environment. It can be totally removed if using CPUs
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.08
set_session(tf.Session(config=config))
############################################################

for train_validation_idx, train_test_idx in kfold.split(X, Y_1):

    hist_fname = "history_" + str(fld) + "_" + output_filename
    open(hist_fname, 'w').close()

    fld += 1
    X_train_valid, X_valid =   X[train_validation_idx],   X[train_test_idx]  
    Y_train_valid, Y_valid =   Y[train_validation_idx],   Y[train_test_idx]

    X_train, X_test, Y_train, Y_test = train_test_split(X_train_valid, Y_train_valid, test_size=0.13, random_state=seed)

    model = Sequential()

    model = Sequential()
    model.add(Embedding(top_words, vector_dimension, input_length=max_words))
    model.add(LSTM(lstm_size))
    model.add(Dense(max_words, activation='relu', W_regularizer=l2(0.90)))
    model.add(Dense(3, activation='softmax', W_regularizer=l2(0.1)))
    adam_1 = Adam(lr=0.008)
    model.compile(loss='categorical_crossentropy', optimizer=adam_1,metrics=['accuracy'])

    print (model.summary())

    # loop through all epocs

    best_acc = 0
    best_err = 10
    best_epoch = 0

    max_counter = 100       # max number of epochs
    counter = max_counter
    i = 0
    while counter > 0:

        counter -=1
        i += 1

        history = model.fit(X_train,Y_train, nb_epoch=1, batch_size=50, validation_data=(X_test, Y_test), verbose=1, shuffle=False)

        acc = history.history['val_acc'][0]
        train_acc = history.history['acc'][0]
        err = history.history['val_loss'][0]
        print "Fold:", fld, " Epoch:",i, " best accuracy:", best_acc, " best loss:", best_err, " ErrDiff:", abs(err - best_err), " since epoch:", best_epoch

        peak = 0
        if ((acc > best_acc) and ((err < best_err) or ((err - best_err) < 0.1))):

            print "Saving model"
            model.save("model_" + output_filename + ".h5")
            best_err = err
            best_acc = acc
            best_epoch = i

            # save a marking to history for a new peak
            peak = 1

            # increase the counter depending when a new peak is found
            counter = counter + int(30 * (max_counter - counter) / max_counter)
            if (counter > max_counter):
                max_counter = counter


        f = open(hist_fname, 'a')
        f.write((str(i) + "," + str(history.history['acc'][0]) + "," + str(history.history['loss'][0]) + "," + str(history.history['val_acc'][0]) + "," + str(history.history['val_loss'][0])))
        if (peak==1):
           f.write(",1 \n")
        else:
           f.write(",0 \n")
        f.close()

    # end of epocs loop

    scores = model.evaluate(X_test,Y_test, verbose=1)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    cvscores.append(scores[1] * 100)

    print(history.history.keys())

    # ----------------------------------------------------------------------
    # load the optimum model saved on disk and use it for making predictions.

    del model
    model = load_model("model_" + output_filename + ".h5")

    predictions = model.predict(X_valid, verbose=1)
    print predictions

    scores = model.evaluate(X_valid, Y_valid, verbose=1)
    print scores

    # array with all ids of test_set ( validation )
    train_test_idx = train_test_idx.reshape(train_test_idx.shape[0], 1)

    # build the output array
    tIDs = []
    Corpus = []
    labels = []
    for i in range(0, train_test_idx.shape[0]):
	    tIDs.append(ids[int(train_test_idx[i])])
	    Corpus.append(corpus[int(train_test_idx[i])])
	    labels.append(Y_1[int(train_test_idx[i])])

    pred1 = [item[0] for item in predictions]
    pred2 = [item[1] for item in predictions]
    pred3 = [item[2] for item in predictions]

    labels_A = np.asarray(labels)
    tIDs_A = np.asarray(tIDs)
    pred1_A = np.asarray(pred1,dtype=float)
    pred2_A = np.asarray(pred2,dtype=float)
    pred3_A = np.asarray(pred3,dtype=float)

    tIDs_A = tIDs_A.reshape(tIDs_A.shape[0],1)
    labels_A = labels_A.reshape(labels_A.shape[0],1)
    pred1_A = pred1_A.reshape(pred1_A.shape[0],1)
    pred2_A = pred2_A.reshape(pred2_A.shape[0],1)
    pred3_A = pred3_A.reshape(pred3_A.shape[0],1)

    output = np.hstack((tIDs_A,labels_A))
    output = np.hstack((output,pred1_A))
    output = np.hstack((output,pred2_A))
    output = np.hstack((output,pred3_A))
 
    # save report to disk file.

    f_handle = file(output_filename, 'a')
    np.savetxt(f_handle, (output),delimiter=',',fmt="%s")
    f_handle.close()
