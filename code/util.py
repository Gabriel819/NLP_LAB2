import torch
import pandas as pd
from typing import Union, List, Dict

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag

from sklearn.metrics import average_precision_score

def load_data(filepath: str = 'YOUR/CSV/FILE/PATH'):
    """
        TO DO: read .csv file and load data
        filepath: str = 'YOUR/CSV/FILE/PATH' → Union[List[str], List[int]]
    """
    df = pd.read_csv(filepath)

    data = list(df['sentence'])
    targets = list(df['label'])

    return data, targets


def tokenization(sents: List[str], N_s):
    """
        TO DO: tokenize sentences into the list of words
        sents: List[str] → List[List[str]]
    """
    tokens = []

    for sen in sents:
        tmp_token = word_tokenize(sen)
        if len(tmp_token) < N_s:  # padding
            # If the sentence's length is smaller than 20, pad 'pad_word' at the last part of the sentence
            tmp_token += ['pad_word'] * (N_s - len(tmp_token))
            tokens.append(tmp_token)
        elif len(tmp_token) > N_s:  # cut
            # If the sentence's length exceed 20, cut the rest
            tokens.append(tmp_token[0:N_s])
        else:  # len(sen) == N_s
            # If the sentence's length is just 20, just use the all words
            tokens.append(tmp_token)

    return tokens

def pos_tagging(words: List[str]):
    """
        Use this method when lemmatizing
        words: List[str]) → Dict[str, str]

        Input: list of words
        Output: {word: tag}
    """
    words_only_alpha = [w for w in words if w.isalpha()]

    def format_conversion(v, pos_tags=['n', 'v', 'a', 'r', 's']):
        w, p = v
        p_lower = p[0].lower()
        p_new = 'n' if p_lower not in pos_tags else p_lower
        return w, p_new

    res_pos = pos_tag(words_only_alpha)
    word2pos = {w:p for w, p in list(map(format_conversion, res_pos))}

    for w in words:
        if w not in word2pos:
            word2pos[w] = 'n'

    return word2pos


def lemmatization(tokens: List[List[str]]):
    """
        TO DO: lemmatize stem from the words
        tokens: List[List[str]] → List[List[str]]
    """
    lemmatizer = WordNetLemmatizer()
    lemmas = []

    for sen in tokens:
        pos_list = pos_tagging(sen)
        tmp = []
        for word in sen:
            print(word, " → ", pos_list[word])  # I print out each word's pos to check if it is applied well
            tmp.append(lemmatizer.lemmatize(word, pos=pos_list[word]))
        lemmas.append(tmp)

    return lemmas

def make_unique_char_dic(lemmas: List[List[str]]):
    """
		TO DO: extract unique characters in lemmatized words
		lemmas: List[List[str]] → Dictionary
	"""
    unique_char_list = ['P','U'] # Put Padding char and Unknown char in advance
    unique_char_dict = {}

    for sen in lemmas:
        for word in sen:
            if word == 'pad_word':
                continue
            for i in range(len(word)):
                if word[i] not in unique_char_list:
                    unique_char_list.append(word[i])

    idxs = list(range(len(unique_char_list)))
    unique_char_dict = dict(zip(unique_char_list, idxs))

    return unique_char_dict

def char_onehot(lemmas: List[List[str]], unique_char_dict, N_w):
	"""
		TO DO: convert characters in lemmatized word to one-hot vector
		lemmas: List[List[str]] → Tensor
	"""
	len_unique = len(unique_char_dict)
	v = torch.zeros((len(lemmas), len(lemmas[0]), N_w, len_unique))

	for i, sen in enumerate(lemmas):
		for j, word in enumerate(sen):
			if word == 'pad_word':
				v[i, j, :, unique_char_dict['P']] = 1
				continue
			if len(word) < N_w:
				# if the number of character in word doesn't exceed N_w, pad the 'P' alphabet
				word += ('P' * (N_w - len(word)))
			for k, alphabet in enumerate(word):
				if k >= N_w: # max length of each word is N_s, which was set to be 10 here.
					break
				if alphabet not in unique_char_dict.keys(): # some alphabet in test set may not be in char dict
					v[i, j, k, unique_char_dict['U']] = 1
				else:
					v[i, j, k, unique_char_dict[alphabet]] = 1

	return v

def get_ap_score(y_true, y_scores):
    """
    Get average precision score between 2 1-d numpy arrays
    Args:
        y_true: batch of true labels
        y_scores: batch of confidence scores

    Returns:
        sum of batch average precision
    """
    scores = 0.0

    for i in range(y_true.shape[0]):
        scores += average_precision_score(y_true=y_true[i], y_score=y_scores[i])

    return scores