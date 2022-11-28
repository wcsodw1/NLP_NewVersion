# python BERT_Example.py
# coding: utf-8
# Reference : https://leemeng.tw/attack_on_bert_transfer_learning_in_nlp.html

import torch
import random
from transformers import BertTokenizer
from IPython.display import clear_output
from transformers import BertForMaskedLM


PRETRAINED_MODEL_NAME = "bert-base-chinese"  # 指定繁簡中文 BERT-BASE 預訓練模型

# 取得此預訓練模型所使用的 tokenizer
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

# clear_output()
print("PyTorch 版本：", torch.__version__)

vocab = tokenizer.vocab
print("字典大小：", len(vocab))

random_tokens = random.sample(list(vocab), 10)
random_ids = [vocab[t] for t in random_tokens]

print("{0:20}{1:15}".format("token", "index"))
print("-" * 25)
for t, id in zip(random_tokens, random_ids):
    print("{0:15}{1:10}".format(t, id))

indices = list(range(647, 657))
some_pairs = [(t, idx) for t, idx in vocab.items() if idx in indices]
for pair in some_pairs:
    print(pair)


text = "[CLS] 等到潮水 [MASK] 了，就知道誰沒穿褲子。"
tokens = tokenizer.tokenize(text)
ids = tokenizer.convert_tokens_to_ids(tokens)

print(text)
print(tokens[:10], '...')
print(ids[:10], '...')


# 除了 tokens 以外我們還需要辨別句子的 segment ids
tokens_tensor = torch.tensor([ids])  # (1, seq_len)
segments_tensors = torch.zeros_like(tokens_tensor)  # (1, seq_len)
maskedLM_model = BertForMaskedLM.from_pretrained(PRETRAINED_MODEL_NAME)
clear_output()

# 使用 masked LM 估計 [MASK] 位置所代表的實際 token
maskedLM_model.eval()
with torch.no_grad():
    outputs = maskedLM_model(tokens_tensor, segments_tensors)
    predictions = outputs[0]
    # (1, seq_len, num_hidden_units)
del maskedLM_model

# 將 [MASK] 位置的機率分佈取 top k 最有可能的 tokens 出來
masked_index = 5
k = 10
probs, indices = torch.topk(torch.softmax(predictions[0, masked_index], -1), k)
predicted_tokens = tokenizer.convert_ids_to_tokens(indices.tolist())

# 顯示 top k 可能的字。一般我們就是取 top 1 當作預測值
print("輸入 tokens ：", tokens[:10], '...')
print('-' * 50)
for i, (t, p) in enumerate(zip(predicted_tokens, probs), 1):
    tokens[masked_index] = t
    print("Top {} ({:2}%)：{}".format(
        i, int(p.item() * 100), tokens[:10]), '...')

'''
Result : 
輸入 tokens ： ['[CLS]', '等', '到', '潮', '水', '[MASK]', '了', '，', '就', '知'] ...
--------------------------------------------------
Top 1 (67%)：['[CLS]', '等', '到', '潮', '水', '來', '了', '，', '就', '知'] ...
Top 2 (25%)：['[CLS]', '等', '到', '潮', '水', '濕', '了', '，', '就', '知'] ...
Top 3 ( 2%)：['[CLS]', '等', '到', '潮', '水', '過', '了', '，', '就', '知'] ...
Top 4 ( 0%)：['[CLS]', '等', '到', '潮', '水', '流', '了', '，', '就', '知'] ...
Top 5 ( 0%)：['[CLS]', '等', '到', '潮', '水', '走', '了', '，', '就', '知'] ...
Top 6 ( 0%)：['[CLS]', '等', '到', '潮', '水', '停', '了', '，', '就', '知'] ...
Top 7 ( 0%)：['[CLS]', '等', '到', '潮', '水', '乾', '了', '，', '就', '知'] ...
Top 8 ( 0%)：['[CLS]', '等', '到', '潮', '水', '到', '了', '，', '就', '知'] ...
Top 9 ( 0%)：['[CLS]', '等', '到', '潮', '水', '溼', '了', '，', '就', '知'] ...
Top 10 ( 0%)：['[CLS]', '等', '到', '潮', '水', '干', '了', '，', '就', '知'] ...
'''
