import pandas as pd
import re
import string
import numpy as np
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from itertools import chain
from collections import Counter
from typing import List, Dict
from torch import Generator, tensor, zeros, hstack
from torch.utils.data import TensorDataset, DataLoader, random_split

class MovieTokenizer():
    def __init__(self, vocab2idx: Dict[str, int], padding_len: int):
        self.vocab2idx = vocab2idx
        self.padding_len = padding_len

    def tokenize_inputs(self, review_frame: pd.DataFrame):
        word_list = review_frame['review_parsed'].apply(lambda words: [self.vocab2idx[word] for word in words]).to_list()
        return word_list

    def pad_reviews(self, tokenized_data):
        padded_reviews = zeros((len(tokenized_data), self.padding_len), dtype = int)
        for i, review_tokens in enumerate(tokenized_data):
            if len(review_tokens) > self.padding_len:
                padded_reviews[i,:] = tensor(review_tokens[:self.padding_len])
            else:
                zeroes = zeros(self.padding_len-len(review_tokens))
                padded_reviews[i,:] = hstack((zeroes, tensor(review_tokens)))
        return padded_reviews


def train_test_split(
        padded_tokens: List[List[int]], labels: List[int],
        seed: int, batch_size: int, train_prop: float, 
        test_prop: float, val_prop: float, shuffle: bool
    ):
    dataset = TensorDataset(tensor(padded_tokens), tensor(labels))

    N = len(dataset)
    train_size = int(N * train_prop)
    dev_size = int(N * val_prop)
    test_size = int(N * test_prop)

    train_dataset, dev_dataset, test_dataset = random_split(
        dataset, [train_size, dev_size, test_size],
        generator = Generator().manual_seed(seed)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=shuffle)   
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
    return train_loader, dev_loader, test_loader


def read_data(fname: str) -> pd.DataFrame:
    assert fname.suffix == '.csv'
    return pd.read_csv(fname)



def clean_data(review_frame: pd.DataFrame, stop_words) -> pd.DataFrame:
    def text_prepoc(txt: str, stop_words: set) -> str:
        # lowercasing
        txt = txt.lower()
        # removing HTML tags
        soup = BeautifulSoup(txt, 'html.parser')
        txt = soup.get_text()
        # removing URLs
        txt = re.sub(r'http[s]?://\S+|www\.\S+', '', txt)
        # removing punctuations
        txt = txt.translate(str.maketrans('', '', string.punctuation))   # str.maketrans(x, y, z)  x:要被替换的字符集合  y:替换后的字符集合 z:要删除的字符集合
        # removing stopwords
        txt = ' '.join([word for word in word_tokenize(txt) if word.lower() not in stop_words])    # word_tokenize(txt) or txt.split()
        # delete numbers
        txt = re.sub(r'\d+', '', txt)
        # word tokenization
        txt = word_tokenize(txt)
        return txt

    review_frame['review_parsed'] = review_frame['review'].apply(lambda x: text_prepoc(x, stop_words)) 
    return review_frame


def get_word_count(review_frame: pd.DataFrame):
    sent_lists = review_frame['review_parsed'].to_list()
    words = list(chain(*sent_lists))
    most_freq_words = Counter(words).most_common(len(words))
    vocab2idx = {word: idx for idx, (word, freq) in enumerate(most_freq_words)}
    return vocab2idx

