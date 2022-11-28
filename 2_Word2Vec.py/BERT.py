# python BERT.py

# coding: utf-8

# 1.Preprocessing :
# 1.1 import libraries
import os
import torch
from transformers import AutoTokenizer, AutoModel
from keras.preprocessing.sequence import pad_sequences
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# 1.2 Initial Model(BERT'S Chinese) and Tokenization
tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
embedding = AutoModel.from_pretrained('bert-base-chinese')

'''
要使用 BERT 轉換文字成向量
1.首先我們需要把我們的文字轉換成 BERT 模型當中單個 Token 的編號
2.並把我們的輸入都 Padding 成一樣的長度
3.然後提出一個句子的 Mask
4.使用 Hugging Face 事先訓練好的 Pre-trained 模型了
'''

# 1.3 Preprocess
sent = '今天天氣真 Good。'
# 使用 tokenizer.encode() 將我的句子編碼成 BERT 中所需要的編號，每個編號對應著一個『字』(Character)，最前方為 [CLS]，最後方為 [SEP]，這是 BERT 所需要的輸入格式
sent_token = tokenizer.encode(sent)
sent_token_padding = pad_sequences(
    [sent_token], maxlen=10, padding='post', dtype='int')
masks = [[float(value > 0) for value in values]
         for values in sent_token_padding]
print('sent:', sent)
print('sent_token:', sent_token)
print('sent_token_padding:', sent_token_padding)
print('masks:', masks)
print('\n')


# 2.Convert
'''都轉為 torch.tensor 的型態後，輸入建立好的 "embedding" (在這裡即為 BERT 模型)，就可以得到最後的輸出。'''
inputs = torch.tensor(sent_token_padding)
masks = torch.tensor(masks)
embedded, _ = embedding(inputs, attention_mask=masks)
print(embedded)

# ========================= #
# Preprocess
sent = '今天天氣真 Good。'
sent_token = ['[CLS]'] + tokenizer.tokenize(sent) + ['[SEP]']
sent_token_encode = tokenizer.convert_tokens_to_ids(sent_token)
sent_token_decode = tokenizer.convert_ids_to_tokens(sent_token_encode)

print('sent:', sent)
print('sent_token:', sent_token)
print('encode:', sent_token_encode)
print('decode:', sent_token_decode)
