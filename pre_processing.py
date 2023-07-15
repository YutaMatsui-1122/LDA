import pandas as pd
import glob 
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

def pre_processing():
    text_list = []
    document_dir_list =  glob.glob("dataset/paper/*")
    document_file_list = []
    for document_dir in document_dir_list:
        document_file_list += glob.glob(document_dir+"/*")
    for i, file_name in enumerate(document_file_list):
        f = open(file_name, 'r', encoding='UTF-8')
        text = f.read().lower()
        text_list.append([text])
    dataset = pd.DataFrame(data=text_list,columns=['text'])
    my_stopwords = stopwords.words('english') + ["'", '"', ':', ';', '.', ',', '-', '!', '?', "'s", ")", "(", "}", "{","$","^d", "\\"]
    # テキストデータのトークン化
    dataset['text'] = dataset['text'].apply(lambda x: word_tokenize(x)).apply(lambda x: [word for word in x if not word.lower() in my_stopwords])
    full_list = []
    documents = []
    max_length = 0
    for text in dataset.values:
        full_list = full_list + text[0]
        max_length = max(max_length, len(text[0]))
        documents.append(text[0])
    full_set = set(full_list)
    dict_list = list(full_set)
    index2word_dict = {i:dict_list[i] for i in range(len(dict_list))}
    word2index_dict = {dict_list[i]:i for i in range(len(dict_list))}
    w = []
    for doc in documents:
        
        w.append([word2index_dict[word] for word in doc]+[-1 for i in range(max_length-len(doc))])
    
    w = np.array(w)
    print(w.shape)
    return w, index2word_dict
    
