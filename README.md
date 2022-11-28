# NLP Skill Tree : 

- NLP STEP : Document --> 0.Language Detection --> 1.Preprocessing --> 2.Modeling --> 3.TasK/Output --> 4.Evaluation 
![image](./data/img/nlp_3Step.png)

- 底下提供一張我個人覺得畫得很好的圖，完美詮釋NLP與其他資訊科學領域的交集關係。<br>
![image](./data/img/nlp.png)

## 1.Preprocessing(預/前處理) : 
- 以NLP來說，preprocessing通常包含兩個重要工作：爬蟲(crawling)及斷詞(tokenization)。
    - 1.A Crawling : 
        - 常用的爬蟲套件有requests和Beautiful Soup, Selenium等等
        - Selensium Project 

    - 1.B Tokenization( = word segmentation) : 
        - Data cleaning
        - Word Cut(斷詞), tokenization
        - stemming?!!
        - NLTK(Natural Language Tool Kit)
        - StopWords
        - vectorization (apply CountVectorizer) -> from sklearn.feature_extraction.text import CountVectorizer
        - !中文的tokenization特別會有粒度(granularity)的問題-> 相較英文，中文多了好幾種不同的斷詞方式。這是因為在中文裡，並沒有像英文裡空白的機制可以用來區分字與字之間的間隔。
        <br>[Reference](https://medium.com/@derekliao_62575/nlp%E7%9A%84%E5%9F%BA%E6%9C%AC%E5%9F%B7%E8%A1%8C%E6%AD%A5%E9%A9%9F-i-%E8%AA%9E%E6%96%99%E7%9A%84%E9%A0%90%E8%99%95%E7%90%86-preprocessing-8538f0b763d6)

## 2.Modeling(Language Models 建立語言模型) : 其實語言模型指的就是一種將文字轉為數字表達的方法。

- 2.A Word to Vector
    - Tradition : BoW(詞袋模型-關鍵次數出現次數) : By using "One-hot encoding"
        - 傳統BOW的缺點非常多 :
            - A.容易造成維度災難(curse of dimensionality)
            - B.向量表達過於稀疏(sparse)
            - C.無法表達語意

    - Advance : 
    - 因此Word2vec有了升級版BOW Model-> TF-IDF/ CBoW
        - TF-IDF :  分成兩個部份，TF和IDF。分別表示詞頻（term frequency，tf）和逆向檔案頻率（inverse document frequency，idf）。和Word2Vec一樣，是種 "將文字轉換為向量的方式"。
        - CBoW(Continuous Bag of Words)
        - skip-gram
        - CBOW
        - Clustering
        - K-means
        - average word vec
        - doc2vec
        - XGBoost train model
        - Plot result

- 2.B Feature_engineering : 
    - Base on the domain knowledge to create the Feature<br>
    [Reference](https://github.com/mohdahmad242/Feature-Engineering-in-NLP/blob/main/Feature_engineering_NLP.ipynb)

- 2.C Model : 
    - 其餘 Model 架構 : 
        - CNN
        - RNN
        - LSTM
        - GloVe

## 3.Apply - Run and take the Result(執行任務/產出結果) : 
- Apply(應用方向)
    - A.文字分類(Text Classification)：例如情緒分析(Sentiment Analysis)、主題的分類、垃圾信(Spam)的辨識、...等，乃至於聊天機器人(ChatBot)。
    - B.文字生成(Text Generation)：例如文本摘要(Text Summary)、作詞、作曲、製造假新聞(Fake News)、影像標題(Image captioning)...等。
    - C.翻譯(Text Translation)：多國語言互轉。
    - D.其他：克漏字、錯字更正、命名實體識別（NER）、著作風格的比對，例如紅樓夢最後幾個章節是不是曹雪芹寫的。
    - E.PTT gossip like/dislike binary classification
    - F.POS [詞性分析](/)

## 4.Evaluation (評估方法) :
    - BLEU
    - ROUGE : Rouge-N/ Rouge-L/ Rouge-W
    - METEOR
    - CIDEr

## 5.Others Tool : Attention is all you need 
   - theorem and program 
   - wmt19 中英文對照資料集

<hr\>

## Sklearn Tool : 
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import GaussianNB # 貝氏分類器

#### TSNE : 降維工具 : 
- from sklearn.manifold import TSNE


#### k-means 設置分群數量 : 
from sklearn import cluster
- ex : kmeans = cluster.KMeans(n_clusters=NUM_CLUSTERS)
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn import metrics

本文覆盖的NLP方法有:
TF-IDF
Count Features
Logistic Regression
Naive Bayes
SVM
Xgboost
Grid Search
Word Vectors
Dense Network
LSTM
GRU
Ensembling
