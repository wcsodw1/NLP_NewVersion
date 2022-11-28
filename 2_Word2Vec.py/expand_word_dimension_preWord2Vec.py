# python expand_word_dimension_preWord2Vec.py
# cd C:/Users/user/Desktop/Github_NEW/AI/NLP/Text_Classfication/Sentiment_Analysis/Tensorflow_Keras

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers  # keras命名空間(Namespace)
# 模型結構的第一層必須為嵌入層(Embedding layer)，它將文字轉為緊密的實數空間，使輸入變為向量，才能進行後續的運算。

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
model = tf.keras.Sequential()

# 字彙表最大為1000，輸出維度為 64，輸入的字數為 10
model.add(layers.Embedding(input_dim=1000, output_dim=64))

# 產生亂數資料，32筆資料，每筆 10 個數字
input_array = np.random.randint(1000, size=(32, 10))
print("input_array", input_array)
# 指定損失函數
model.compile('rmsprop', 'mse')

# 預測
output_array = model.predict(input_array)
print(output_array.shape)
print(output_array[0])

