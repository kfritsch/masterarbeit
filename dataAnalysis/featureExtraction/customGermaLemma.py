import sys
import os
import codecs
import pickle
from collections import defaultdict
from importlib import import_module
from .customIWNLPLemmatizer import CustomIWNLPLemmatizer

from pyphen import Pyphen
from os.path import dirname, realpath, join

FILE_PATH = dirname(realpath(__file__))


DEFAULT_LEMMATA_PICKLE = 'data/lemmata.pickle'

# valid part-of-speech prefixes
VALID_POS_PREFIXES = ('N', 'V', 'ADJ', 'ADV')

# German language adjective suffixes
ADJ_SUFFIXES_BASE = (
    'bar',
    'haft',
    'ig',
    'isch',
    'lich',
    'los',
    'sam',
    'en',
    'end',
    'ern'
)

ADJ_SUFFIXES_FLEX = (
    'e',
    'er',
    'es',
    'en',
    'em',
    'ere',
    'erer',
    'eres',
    'eren',
    'erem',
    'ste',
    'ster',
    'stes',
    'sten',
    'stem',
)

ADJ_SUFFIXES_DICT = {}

for suffix in ADJ_SUFFIXES_BASE:
    for flex in ADJ_SUFFIXES_FLEX:
        ADJ_SUFFIXES_DICT[suffix + flex] = suffix


class CustomGermaLemma(object):
    """
    Lemmatizer for German language text main class.
    """
    pyphen_dic = Pyphen(lang='de')

    def __init__(self, **kwargs):
        if('tiger_corpus' in kwargs):
            self.lemmata, self.lemmata_lower = self.load_corpus_lemmata(kwargs['tiger_corpus'])
        elif('pickle' in kwargs):
            self.load_from_pickle(kwargs['pickle'])
        self.pattern_module = import_module('pattern.de')
        self.iwnlpLemmatizer = CustomIWNLPLemmatizer(join(FILE_PATH,"lib","IWNLP.Lemmatizer_20170501.json"))

    def find_lemma(self, w, pos, props=None):
        # do not process empty strings
        if(not(w)):
            raise ValueError("Empty String!")
        # valid pos = N,V,ADJ,ADV
        elif(not(pos in ["NOUN","VERB","ADJ","ADV","AUX"])):
            return word
        iwnlpLemmas = self.iwnlpLemmatizer.lemmatize(w, pos)
        if(iwnlpLemmas): return iwnlpLemmas, None

        if(pos.startswith('N') or pos.startswith('V')):
            pos = pos[0]
        elif(pos.startswith('ADJ') or pos.startswith('ADV')):
            pos = pos[:3]
        elif(pos=="AUX"):
            pos = "V"
        # look if we can directly find `w` in the lemmata dictionary
        res = self.dict_search(w, pos)
        composita = None
        if(not(res)):
            # try to split nouns that are made of composita
            if(pos == 'N'):
                compositaRes = self._composita_lemma(w)
                res = compositaRes[0]
                if(len(compositaRes)>1):
                    composita = compositaRes[1:]

            # try to lemmatize adjectives using prevalent German language adjective suffixe
            elif pos == 'ADJ':
                res = self._adj_lemma(w)

        # try to use pattern.de module
        if(not(res) and props and self.pattern_module):
            res_pattern = self._lemma_via_patternlib(w, pos, props)
            if res_pattern != w:
                res = res_pattern

        if(res):
            # nouns always start with a capital letter
            if(pos == 'N'):
                if len(res) > 1 and res[0].islower():
                    res = res[0].upper() + res[1:]
            else:
                res = res.lower()
            return [res], composita

        return res, composita

    def dict_search(self, w, pos, use_lower=False):
        """
        Lemmata dictionary lookup for word `w` with POS tag `pos`.
        Return lemma if found, else None.
        """
        pos_lemmata = self.lemmata_lower[pos] if use_lower else self.lemmata[pos]

        return pos_lemmata.get(w, None)

    def _adj_lemma(self, w):
        """
        Try to lemmatize adjectives using prevalent German language adjective suffixes. Return possibly lemmatized
        adjective.
        """
        for full, reduced in ADJ_SUFFIXES_DICT.items():
            if w.endswith(full):
                return w[:-len(full)] + reduced

        return None

    def _composita_lemma(self, w):
        """
        Try to split a word `w` that is possibly made of composita.
        Return the lemma if found, else return None.
        """
        # find most important split position first, only right part needs to exist
        try:
            split_positions = [w.rfind('-') + 1]
        except ValueError:
            split_positions = []
        split_positions.extend([p for p in self.pyphen_dic.positions(w) if p not in split_positions])

        for hy_pos in split_positions:
            left, right = w[:hy_pos], w[hy_pos:]
            if(left and right and not(right.endswith('innen'))):
                resRight = self.dict_search(right, 'N', use_lower=right[0].islower())
                if(not(resRight)):
                    resRight = self.iwnlpLemmatizer.lemmatize(right, "NOUN")
                    if(resRight): resRight = resRight[0]
                if resRight:
                    resLeft = self.dict_search(left, 'N', use_lower=left[0].islower())
                    if(not(resLeft)):
                        resLeft = self.iwnlpLemmatizer.lemmatize(left, "NOUN")
                        if(resLeft): resLeft = resLeft[0]
                    if(not(resLeft)):
                        resLeft = self.dict_search(left[:-1], 'N', use_lower=left[0].islower())
                    if(not(resLeft)):
                        resLeft = self.iwnlpLemmatizer.lemmatize(left[:-1], "NOUN")
                        if(resLeft): resLeft = resLeft[0]
                    # concatenate the left side with the found partial lemma
                    if left[-1] == '-':
                        res = left + resRight.capitalize()
                    else:
                        res = left + resRight.lower()

                    resList = []
                    if w.isupper():
                        resList.append(res.upper())
                    else:
                        resList.append(res.capitalize())
                    resList.append(resRight.capitalize())
                    if(resLeft): resList.append(resLeft.capitalize())
                    return resList

        # try other split positions, both parts need to exist
        split_positions = [i for i in range(3,len(w)-2) if not(i in split_positions)]

        for hy_pos in split_positions:
            left, right = w[:hy_pos], w[hy_pos:]
            if(left and right and not(right.endswith('innen'))):
                resRight = self.dict_search(right, 'N', use_lower=right[0].islower())
                if(not(resRight)):
                    resRight = self.iwnlpLemmatizer.lemmatize(right, "NOUN")
                    if(resRight): resRight = resRight[0]
                resLeft = self.dict_search(left, 'N', use_lower=left[0].islower())
                if(not(resLeft)):
                    resLeft = self.iwnlpLemmatizer.lemmatize(left, "NOUN")
                    if(resLeft): resLeft = resLeft[0]
                if(not(resLeft)):
                    resLeft = self.dict_search(left[:-1], 'N', use_lower=left[0].islower())
                if(not(resLeft)):
                    resLeft = self.iwnlpLemmatizer.lemmatize(left[:-1], "NOUN")
                    if(resLeft): resLeft = resLeft[0]
                if(resRight and resLeft):
                    res = left + resRight.lower()
                    resList = []
                    if w.isupper():
                        resList.append(res.upper())
                    else:
                        resList.append(res.capitalize())
                    resList.append(resRight.capitalize())
                    resList.append(resLeft.capitalize())
                    return resList

        return [None]

    def _lemma_via_patternlib(self, w, pos, props={}):
        """
        Try to find a lemma for word `w` that has a Part-of-Speech tag `pos_tag` by using pattern.de module's functions.
        Return the lemma or `w` if lemmatization was not possible with pattern.de
        """
        if(not(self.pattern_module)):
            raise RuntimeError('pattern.de module not loaded')
        if(pos.startswith('N') and "number" in props and props["number"]!="Sg"): # pos == 'NP': singularize noun
            return self.pattern_module.singularize(w)
        elif(pos.startswith('V') and "form" in props and props["form"]!="INF"):  # get infinitive of verb
            return self.pattern_module.conjugate(w)
        elif(pos.startswith('ADJ') or pos.startswith('ADV')):  # get baseform of adjective or adverb
            return self.pattern_module.predicative(w)

        return w

    @staticmethod
    def add_to_lemmata_dicts(lemmata, lemmata_lower, token, lemma, pos):
        for pos_prefix in VALID_POS_PREFIXES:
            if pos.startswith(pos_prefix):
                if token not in lemmata[pos_prefix]:
                    lemmata[pos_prefix][token] = lemma
                if lemma not in lemmata[pos_prefix]:  # for quicker lookup
                    lemmata[pos_prefix][lemma] = lemma

                if pos_prefix == 'N':
                    token_lower = token.lower()
                    if token_lower not in lemmata_lower[pos_prefix]:
                        lemmata_lower[pos_prefix][token_lower] = lemma
                    lemma_lower = lemma.lower()
                    if lemma_lower not in lemmata_lower[pos_prefix]:
                        lemmata_lower[pos_prefix][lemma_lower] = lemma

                return

    @classmethod
    def load_corpus_lemmata(cls, corpus_file):
        lemmata = defaultdict(dict)
        lemmata_lower = defaultdict(dict)

        with codecs.open(corpus_file, encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) == 15:
                    token, lemma = parts[1:3]
                    pos = parts[4]
                    cls.add_to_lemmata_dicts(lemmata, lemmata_lower, token, lemma, pos)

        return lemmata, lemmata_lower

    def save_to_pickle(self, pickle_file):
        with open(pickle_file, 'wb') as f:
            pickle.dump((self.lemmata, self.lemmata_lower), f, protocol=2)

    def load_from_pickle(self, pickle_file):
        with open(pickle_file, 'rb') as f:
            self.lemmata, self.lemmata_lower = pickle.load(f)
