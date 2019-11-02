import logging
import json
import io
from os.path import dirname, realpath, join

FILE_PATH = dirname(realpath(__file__))


class CustomIWNLPLemmatizer(object):
    def __init__(self, lemmatizer_path=join(FILE_PATH,"lib","IWNLP.Lemmatizer_20170501.json")):
        self.lemmatizer = {}
        with io.open(lemmatizer_path, encoding='utf-8') as data_file:
            raw = json.load(data_file)
            for entry in raw:
                self.lemmatizer[entry["Form"]] = entry["Lemmas"]
        # parser error in 20170501.json
        self.remove_entry("die", "Noun", "Adsorbens")

    def remove_entry(self, form, pos, lemma):
        key = form.lower().strip()
        if key in self.lemmatizer:
            wrong_entry = {"POS": pos, "Form": form, "Lemma": lemma}
            if wrong_entry in self.lemmatizer[key]:
                self.lemmatizer[key].remove(wrong_entry)

    def contains_entry(self, word, pos=None, ignore_case=False):
        key = word.lower().strip()
        if not pos:
            if ignore_case:
                return key in self.lemmatizer
            else:
                return key in self.lemmatizer and any(filter(lambda x: x["Form"] == word, self.lemmatizer[key]))
        elif not isinstance(pos, list):
            if ignore_case:
                return key in self.lemmatizer and any(filter(lambda x: x["POS"] == pos, self.lemmatizer[key]))
            else:
                return key in self.lemmatizer and any(
                    filter(lambda x: x["POS"] == pos and x["Form"] == word, self.lemmatizer[key]))
        else:
            for pos_entry in pos:
                if self.contains_entry(word, pos_entry, ignore_case):
                    return True
            return False

    def get_entries(self, word, pos=None, ignore_case=False):
        entries = []
        key = word.lower().strip()
        if not pos:
            if ignore_case:
                entries = self.lemmatizer[key]
            else:
                entries = list(filter(lambda x: x["Form"] == word, self.lemmatizer[key]))
        elif not isinstance(pos, list):
            if ignore_case:
                entries = list(filter(lambda x: x["POS"] == pos, self.lemmatizer[key]))
            else:
                entries = list(filter(lambda x: x["POS"] == pos and x["Form"] == word, self.lemmatizer[key]))
        else:
            for pos_entry in pos:
                if self.contains_entry(word, pos=pos_entry, ignore_case=ignore_case):
                    entries.extend(self.get_entries(word, pos_entry, ignore_case))
        return entries

    def get_lemmas(self, word, pos=None, ignore_case=False):
        """
        Return all lemmas for a given word. This method assumes that the specified word is present in the dictionary
        :param word: Word that is present in the IWNLP lemmatizer
        """
        entries = self.get_entries(word, pos, ignore_case)
        lemmas = list(set([entry["Lemma"] for entry in entries]))
        return sorted(lemmas)

    def lemmatize_plain(self, word, ignore_case=False):
        if self.contains_entry(word, ignore_case=ignore_case):
            return self.get_lemmas(word, ignore_case=ignore_case)
        else:
            return None

    def lemmatize(self, word, udPos):
        """
        Python port of the lemmatize method, see https://github.com/Liebeck/IWNLP.Lemmatizer/blob/master/IWNLP.Lemmatizer.Predictor/IWNLPSentenceProcessor.cs

        """
        # do not process empty strings
        if(not(word)):
            raise ValueError("Empty String!")
        # valid pos = N,V,ADJ,ADV
        elif(not(udPos in ["NOUN","VERB","ADJ","ADV","AUX"])):
            return word

        if udPos == 'NOUN':
            if len(word) > 1 and word[0].islower():
                word = word[0].upper() + word[1:]
        else:
            word = word.lower()

        if udPos == "NOUN":
            if self.contains_entry(word, "Noun"):
                return self.get_lemmas(word, "Noun")
            elif self.contains_entry(word, "X"):
                return self.get_lemmas(word, "X")
            elif self.contains_entry(word, "AdjectivalDeclension"):
                return self.get_lemmas(word, "AdjectivalDeclension")
            elif self.contains_entry(word, ["Noun", "X"], ignore_case=True):
                return self.get_lemmas(word, ["Noun", "X"], ignore_case=True)
            else:
                return None
        elif udPos in ["ADJ", "ADV"]:
            if self.contains_entry(word, "Adjective"):
                return self.get_lemmas(word, "Adjective")
            elif self.contains_entry(word, "Adjective", ignore_case=True):
                return self.get_lemmas(word, "Adjective", ignore_case=True)
            # Account for possible errors in the POS tagger. This order was fine-tuned in terms of accuracy
            elif self.contains_entry(word, "Noun", ignore_case=True):
                return self.get_lemmas(word, "Noun", ignore_case=True)
            elif self.contains_entry(word, "X", ignore_case=True):
                return self.get_lemmas(word, "X", ignore_case=True)
            elif self.contains_entry(word, "Verb", ignore_case=True):
                return self.get_lemmas(word, "Verb", ignore_case=True)
            else:
                return None
        elif udPos in ["VERB", "AUX"]:
            if self.contains_entry(word, "Verb", ignore_case=True):
                return self.get_lemmas(word, "Verb", ignore_case=True)
            else:
                return None
        else:
            return None
