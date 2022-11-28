# python BertTokenizer.py

# from transformers import BertTokenizer, BertModel
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained("bert-base-uncased")
# text = "Hello, my dog is cute"
# encoded_input = tokenizer(text, max_length=100,
#                           add_special_tokens=True, truncation=True,
#                           padding=True, return_tensors="pt")
# output = model(**encoded_input)
# last_hidden_state, pooler_output = output[0], output[1]
# print("last_hidden_state", last_hidden_state.shape)
# print("pooler_output", pooler_output.shape)


# 1. BertConfig
# transformers.BertConfig 可以自定義 Bert 模型的架構，參數都是可選的
from transformers import BertTokenizer
# import argparse
# from transformers import BertModel, BertConfig
# configuration = BertConfig()  # 进行模型的配置，变量为空即使用默认参数
# model = BertModel(configuration)  # 使用自定义配置实例化 Bert 模型
# configuration = model.config  # 查看模型参数


# 2. BertTokenizer
data_path = "C:/Users/user/Desktop/Project/AI/data/NLP/bert_base_vocab.txt"

# [CLS]：在做分類任務時其最後一層的 repr. 會被視為整個輸入序列的 repr.
# [SEP]：有兩個句子的文本會被串接成一個輸入序列，並在兩句之間插入這個 token 以做區隔
# [UNK]：沒出現在 BERT 字典裡頭的字會被這個 token 取代
# [PAD]：zero padding 遮罩，將長度不一的輸入序列補齊方便做 batch 運算
# [MASK]：未知遮罩，僅在預訓練階段會用到

tokenizer = BertTokenizer(vocab_file=data_path)
print(tokenizer)
