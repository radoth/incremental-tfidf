from tokenizers.pre_tokenizers import Whitespace
import os
import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import math
import operator
from transformers import BertTokenizer
from scipy.sparse import csr_matrix, save_npz
import pickle
import time
import datetime
import logging
import shutil

#输出结果路径
result_pth = r"C:\Users\tom\Desktop\tfidftry"
#IDF字典路径
idf_path = r"C:\Users\tom\Desktop\tfidftry\dicts"
#输入新闻路径
input_news_path = r"C:\Users\tom\Desktop\allnews210615\newsWithQuotes"

# 初始化logger
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler(os.path.join(result_pth, "log.txt"))
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
console.setFormatter(formatter)

logger.addHandler(handler)
logger.addHandler(console)


# 输出文件夹用Unix时间戳命名
current_time = str(int(time.mktime(datetime.datetime.now().timetuple())))
os.mkdir(os.path.join(result_pth, current_time))
output_path = os.path.join(result_pth, current_time)

# 空格分词器和BERT分词器
logger.info("加载分词器")
pre_tokenizer = Whitespace()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 加载IDF字典
logger.info("加载IDF字典")
with open(os.path.join(idf_path, "word_doc.pkl"), "rb") as fp:  # Pickling
    word_doc = pickle.load(fp)
with open(os.path.join(idf_path, "word_doc_wordpiece.pkl"), "rb") as fp:  # Pickling
    word_doc_wordpiece = pickle.load(fp)


# 预处理文本
def preprocess(text):
    st = pre_tokenizer.pre_tokenize_str(text)
    st = [i[0].lower() for i in st if len(i[0]) > 1 and not i[0].isdigit()]
    return st

# 读取文本


def readText(file_pth):
    files = sorted(os.listdir(file_pth))
    all_file = [str(i) for i in files]
    np.save(os.path.join(output_path, "fileNum.npy"), np.array(all_file))

    whitespace_tokenized = []
    wordpiece_tokenized = []
    for i in tqdm(files):
        with open(os.path.join(file_pth, i), encoding='utf-8') as f:
            inp = json.loads(f.read())
            data = inp['content']
            title = inp['title']
            data = (title+' ')*5+data
            whitespace_tokenized.append(preprocess(data))
            wordpiece_tokenized.append(tokenizer(data)['input_ids'])
    return whitespace_tokenized, wordpiece_tokenized

# 空格分词法计算TF-IDF


def feature_select_whitespace(text_list):
    global word_doc
    results = []
    for text in tqdm(text_list):
        # 某个文档的词频统计
        doc_frequency = defaultdict(int)
        for i in text:
            doc_frequency[i] += 1

        # 计算TF-IDF值
        word_tf = {}
        word_idf = {}
        text_length = len(text)
        text_num = len(text_list)+word_doc["NEWS_TXTS_COUNT"]
        word_tf_idf = {}
        for i in doc_frequency:
            word_tf[i] = doc_frequency[i]/text_length
            word_doc[i] += 1
            word_idf[i] = math.log(text_num/(word_doc[i]+1))
            word_tf_idf[i] = word_tf[i]*word_idf[i]
        results.append(sorted(word_tf_idf.items(),
                       key=operator.itemgetter(1), reverse=True))
    return results

# BERT分词计算TF-idf


def feature_select_wordpiece(text_list):
    global word_doc_wordpiece
    results = []
    for text in tqdm(text_list):
        # 某个文档的词频统计
        doc_frequency = defaultdict(int)
        for i in text:
            doc_frequency[i] += 1

        # 计算TF值
        word_tf = {}
        word_idf = {}
        text_length = len(text)
        text_num = len(text_list)+word_doc_wordpiece["NEWS_TXTS_COUNT"]
        word_tf_idf = {}
        for i in doc_frequency:
            word_tf[i] = doc_frequency[i]/text_length
            word_doc_wordpiece[i] += 1
            word_idf[i] = math.log(text_num/(word_doc_wordpiece[i]+1))
            word_tf_idf[i] = word_tf[i]*word_idf[i]
        results.append(word_tf_idf)
    return results

# 去除top-k保存


def saveTopk(k, results):
    tfidf = [[j[0] for j in i[:k]] if len(
        i) >= k else [j[0] for j in i[:k]]+["null"]*(k-len(i)) for i in results]
    tfidffreq = [[j[1] for j in i[:k]] if len(
        i) >= k else [j[0] for j in i[:k]]+["null"]*(k-len(i)) for i in results]
    np.save(os.path.join(output_path, "tfidf.npy"), np.array(tfidf))
    np.save(os.path.join(output_path, "tfidffreq.npy"), np.array(tfidffreq))

# 保存特征向量


def saveSparseMatrix(results):
    row = []  # 行指标
    col = []  # 列指标
    data = []  # 在行指标列指标下的数字
    for i in tqdm(range(len(results))):
        for j in results[i]:
            row.append(i)
            col.append(j)
            data.append(results[i][j])
    team = csr_matrix((data, (row, col)), shape=(len(results), 30522))
    save_npz(os.path.join(output_path, "tvfit.npz"), team)

# 保存IDF字典，覆盖到原位置


def saveIDFDict(whitespace_tokenized, wordpiece_tokenized):
    word_doc["NEWS_TXTS_COUNT"] += len(whitespace_tokenized)
    word_doc_wordpiece["NEWS_TXTS_COUNT"] += len(wordpiece_tokenized)
    with open(os.path.join(idf_path, "word_doc.pkl"), "wb") as fp:
        pickle.dump(word_doc, fp, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(idf_path, "word_doc_wordpiece.pkl"), "wb") as fp:
        pickle.dump(word_doc_wordpiece, fp, protocol=pickle.HIGHEST_PROTOCOL)


logger.info("开始读取文本并分词")
whitespace_tokenized, wordpiece_tokenized = readText(input_news_path)
logger.info("开始使用空格分词法增量计算TF-IDF")
whitespace_result = feature_select_whitespace(whitespace_tokenized)
logger.info("开始使用WordPiece分词法增量计算TF-IDF")
wordpiece_result = feature_select_wordpiece(wordpiece_tokenized)
logger.info("开始保存TF-IDF结果")
saveTopk(50, whitespace_result)
logger.info("开始保存特征向量稀疏矩阵")
saveSparseMatrix(wordpiece_result)
logger.info("生成词表")
shutil.copy(os.path.join(idf_path, "words.npy"),
            os.path.join(output_path, "fe.npy"))
logger.info("正在更新IDF值字典，原文章数：{}    新增文章数：{}".format(
    word_doc["NEWS_TXTS_COUNT"], len(whitespace_tokenized)))
saveIDFDict(whitespace_tokenized, wordpiece_tokenized)
