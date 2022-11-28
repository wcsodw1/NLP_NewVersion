# python word2vec_TF_Keras.py

import tensorflow as tf
from tensorflow.keras import layers
from numpy import array
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1.define documents : Input 10個詞
docs = ['Well done!',
        'Good work',
        'Great effort',
        'nice work',
        'Excellent!',
        'Weak',
        'Poor effort!',
        'not good',
        'poor work',
        'Could have done better.']

# 2.Labeling : define class labels
labels = array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])  # 給標準答案(正面負面情緒)
vocab_size = 50
maxlen = 4
encoded_docs = [one_hot(d, vocab_size) for d in docs]
padded_docs = pad_sequences(encoded_docs, maxlen=maxlen, padding='post')


# 3.Modeling :
model = tf.keras.Sequential()
model.add(layers.Embedding(vocab_size, 8, input_length=maxlen))
model.add(layers.Flatten())

# 3.A Fully-Connect : 加上一般的完全連接層(Dense)
print("=============================== Fully-Connect ===============================")
model.add(layers.Dense(1, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])
# summarize the model
print(model.summary())
# fit the model
model.fit(padded_docs, labels, epochs=50, verbose=0)
# evaluate the model
loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
print('Accuracy: %f' % (accuracy*100))

# Result : Accuracy: 89.999998

# 3.B .Add RNN 測試效果(效果更佳) :
print("")
print("=============================== RNN ===============================")
model = tf.keras.Sequential()
model.add(layers.Embedding(vocab_size, 8, input_length=maxlen))
# Add a RNN layer with 128 internal units.
model.add(layers.SimpleRNN(128))
model.add(layers.Dense(1, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])
# summarize the model
print(model.summary())
# fit the model
model.fit(padded_docs, labels, epochs=50, verbose=0)
# evaluate the model
loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
print('Accuracy: %f' % (accuracy*100))

# Result : Accuracy: 100.000000
