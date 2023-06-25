# -*- coding: UTF-8 -*-
import torch
import torch.utils.data as Data
from collections import Counter
import pickle
import numpy as np

class Dataset():
    def load(self, obj_file):
        with open(obj_file, 'rb') as f:
            data_loaded = pickle.loads(f.read())
            self.vocab_size = data_loaded.vocab_size
            self.n = data_loaded.n
            self.batch_size = data_loaded.batch_size
            self.top_words = data_loaded.top_words
            self.train_dataset = data_loaded.train_dataset
            self.test_dataset = data_loaded.test_dataset
            self.train_dataloader = data_loaded.train_dataloader
            self.test_dataloader = data_loaded.test_dataloader

    def save(self, obj_file):
        with open(obj_file, 'wb') as f:
            str = pickle.dumps(self)
            f.write(str)

    def gen(self, vocab_size, n, batch_size=512, filename='ChineseCorpus199801.txt', encoding="gbk", type="n_gram"):
        self.vocab_size = vocab_size
        self.n = n
        self.batch_size = batch_size

        with open(filename, encoding=encoding) as f:
            # used to select the top common words
            words = f.read().split()
        with open(filename, encoding=encoding) as f:
            # used to construct n-gram
            lines = f.readlines()

        # count the number of occurrences of each word
        word_counts = Counter(words)

        # print word_counts to file according to the frequency of occurrence
        with open('word_counts.txt', 'w') as f:
            for word, count in word_counts.most_common():
                f.write(word + ' ' + str(count) + '\n')

        # top words
        self.top_words = {word[0]:idx+1 for idx, word in enumerate(word_counts.most_common(vocab_size-1))}
        print("top words:", self.top_words)
        
        x = []
        y = []
            
        if type=="n_gram":
            # n-gram
            for line in lines:
                words = line.split()
                if len(words) >= n:
                    for i in range(len(words)-n+1):
                        x.append([self.top_words.get(word, 0) for word in words[i:i+n-1]])
                        y.append(self.top_words.get(words[i+n-1], 0))
        elif type=="rnn":
            # for rnn
            for line in lines:
                words = line.split()
                if len(words) >= n:
                    for i in range(len(words)-n+1):
                        x.append([self.top_words.get(word, 0) for word in words[i:i+n-1]])
                        y.append([self.top_words.get(word, 0) for word in words[i+1:i+n]])
        else:
            assert False, "Unknown Type"
        
        print("x shape:", np.array(x).shape)
        print("y shape:", np.array(y).shape)

        # split the data set into training set and test set
        dataset = Data.TensorDataset(torch.tensor(x), torch.tensor(y))
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        self.train_dataset, self.test_dataset = Data.random_split(dataset, [train_size, test_size])
        self.train_dataloader = Data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.test_dataloader = Data.DataLoader(self.test_dataset, batch_size=batch_size, shuffle=True)
        print("train batch count:", len(self.train_dataloader))
        print("test batch count:", len(self.test_dataloader))
