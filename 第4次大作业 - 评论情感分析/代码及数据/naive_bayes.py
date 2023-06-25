import os
import json
import math
from feature import jieba_cut

class NaiveBayesPredictor:
    def __init__(self, ig_c_t, words_all, words_times, p_c, n_grams, output_path="output", feature_words_num=10000):
        self.ig_c_t = ig_c_t
        self.words_all = words_all
        self.words_times = words_times
        self.p_c = p_c
        self.n_grams = n_grams

        # count feature words times
        feature_words = list(ig_c_t.keys())[:feature_words_num]
        not_feature_words = list(set(words_all.keys()) - set(feature_words))
        self.vocab = feature_words + ["<UNK>"]
        feature_words_times = {}
        for i in range(1, 11):
            feature_words_times[str(i)] = {word: words_times[str(i)].get(word, 0) for word in feature_words}
            feature_words_times[str(i)]["<UNK>"] = sum([words_times[str(i)].get(word, 0) for word in not_feature_words])

        # get p(t|c)
        self.p_t_c = {}
        for word in self.vocab:
            self.p_t_c[word] = {}
        for i in range(1, 11):
            for word in self.vocab:
                self.p_t_c[word][str(i)] = 1 + feature_words_times[str(i)][word]
            tot = sum(self.p_t_c[word][str(i)] for word in self.vocab)
            for word in self.vocab:
                self.p_t_c[word][str(i)] /= tot

        with open(os.path.join(output_path, "NB_p_t_c.json"), "w") as f:
            json.dump(self.p_t_c, f, ensure_ascii=False, indent=4)

    def predict(self, text, prt=False):
        str_words = []
        for n_gram in self.n_grams:
            str_words += jieba_cut(text, n_gram)
        str_words = [word if word in self.vocab else "<UNK>" for word in str_words]
        if prt:
            print(str_words)

        log_pred = {}
        for i in range(1, 11):
            p = math.log(self.p_c[str(i)])
            for word in str_words:
                p += math.log(self.p_t_c[word][str(i)])
            log_pred[str(i)] = p

        pred = {}
        for i in range(1, 11):
            pred[str(i)] = 1 / sum([math.exp(log_pred[str(j)] - log_pred[str(i)]) for j in range(1, 11)])

        return pred

    def get_emotional_words(self, text, pred_max, prt=False):
        str_words = []
        for n_gram in self.n_grams:
            str_words += jieba_cut(text, n_gram)
        str_words = [word if word in self.vocab else "<UNK>" for word in str_words]
        
        emotional_words = {}
        for word in str_words:
            emotional_words[word] = math.log(self.p_t_c[word][str(pred_max)]) - \
                sum([math.log(self.p_t_c[word][str(i)]) for i in range(1, 11)])
        emotional_words = sorted(emotional_words.items(), key=lambda x: x[1], reverse=True)
        
        if prt:
            print("emotional words weight:", emotional_words)

        return emotional_words

