import jieba
import os
from collections import Counter, OrderedDict
import json
import math

def jieba_cut(text, n_gram=1):
    str_words = jieba.lcut(text)
    if n_gram == 1:
        return str_words
    else:
        n_gram_words = []
        for i in range(len(str_words) - n_gram + 1):
            n_gram_words.append("".join(str_words[i:i+n_gram]))
        return n_gram_words
    

def count_words(subject_id_list, data_path="data", n_grams=[1,2]):
    """
    return value:
    - words:            for each score, count the number of comments containing the word
    - words_times:      for each score, count the number of times the word appears in comments
    - comment_num:      for each score, count the number of comments
    - words_all:        count the number of comments containing the word
    - comment_num_all:  count the number of comments
    """
    words = {}
    words_times = {}
    comment_num = {}
    for i in range(1, 11):
        words[str(i)] = Counter()
        words_times[str(i)] = Counter()
        comment_num[str(i)] = 0
    for subject_id in subject_id_list:
        with open(os.path.join(data_path, str(subject_id) + ".json"), "r") as f:
            subject = json.load(f)
        for score, comment_list in subject.items():
            if not score.isdigit():
                continue
            for comment in comment_list:
                comment_num[score] += 1
                comment_cut = []
                for n_gram in n_grams:
                    comment_cut += jieba_cut(comment, n_gram)
                words[score].update(set(comment_cut))
                words_times[score].update(comment_cut)
        print(subject_id, "done")

    words_all = Counter()
    for i in range(1, 11):
        words_all.update(words[str(i)])

    comment_num_all = sum(comment_num.values())

    return words, words_times, comment_num, words_all, comment_num_all


def estimate_p(words, words_times, comment_num, words_all, comment_num_all):
    p_c = {str(i): comment_num[str(i)] / comment_num_all for i in range(1, 11)}
    p_t = {word: num / comment_num_all for word, num in words_all.items()}

    p_c_t = {}
    p_c_nt = {}
    for word in words_all.keys():
        if word not in p_c_t:
            p_c_t[word] = {}
        if word not in p_c_nt:
            p_c_nt[word] = {}
        for i in range(1, 11):
            p_c_t[word][str(i)] = words[str(i)].get(word, 0) / words_all[word]
            p_c_nt[word][str(i)] = (comment_num[str(i)] - words[str(i)].get(word, 0)) / (comment_num_all - words_all[word])
    for word in words_all.keys():
        p_c_t[word]["p_t"] = p_t[word]
        p_c_nt[word]["p_nt"] = 1 - p_t[word]

    return p_c, p_c_t, p_c_nt

def get_h_and_ig(words_all, p_c, p_c_t, p_c_nt):
    h_c_t = {}
    for word in words_all.keys():
        h_c_t[word] = 0
        for i in range(1, 11):
            if p_c_t[word][str(i)] > 0:
                h_c_t[word] -= p_c_t[word]["p_t"] * p_c_t[word][str(i)] * math.log(p_c_t[word][str(i)])
            if p_c_nt[word][str(i)] > 0:
                h_c_t[word] -= p_c_nt[word]["p_nt"] * p_c_nt[word][str(i)] * math.log(p_c_nt[word][str(i)])

    h_c = 0
    for i in range(1, 11):
        h_c -= p_c[str(i)] * math.log(p_c[str(i)])

    ig_c_t = {}
    for word in words_all.keys():
        ig_c_t[word] = h_c - h_c_t[word]

    return h_c_t, h_c, ig_c_t