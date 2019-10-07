# Germalemmatizer = GermaLemma(tiger_corpus="tiger/tiger_release_aug07.corrected.16012013.conll09")
# Germalemmatizer.save_to_pickle("tiger/tiger_lemmas.pkl")
from subprocess import check_output
import spacy
from .coreNlp import StanfordCoreNLP
import re
import json
import hunspell
from pyxdameraulevenshtein import damerau_levenshtein_distance_ndarray
from nltk.stem import WordNetLemmatizer
import numpy as np
import nltk
from nltk.corpus import wordnet
from nltk.tag import pos_tag
from os.path import dirname, realpath, join

FILE_PATH = dirname(realpath(__file__))

PEN_UD_MAP = {
    "#":"SYM",
    "$":"SYM",
    "''":"PUNCT",
    "``":"PUNCT",
    ",":"PUNCT",
    "-LRB-":"PUNCT",
    "-RRB-":"PUNCT",
    ".":"PUNCT",
    ":":"PUNCT",
    "ADD":"X",
    "AFX":"ADJ",
    "BES":"VERB",
    "CC":"CCONJ",
    "CD":"NUM",
    "DT":"DET",
    "EX":"PRON",
    "FW":"X",
    "GW":"X",
    "HVS":"VERB",
    "HYPH":"PUNCT",
    "IN":"ADP",
    "JJ":"ADJ",
    "JJR":"ADJ",
    "JJS":"ADJ",
    "LS":"X",
    "MD":"VERB",
    "NFP":"PUNCT",
    "NIL":"X",
    "NN":"NOUN",
    "NNP":"PROPN",
    "NNPS":"PROPN",
    "NNS":"NOUN",
    "PDT":"DET",
    "POS":"PART",
    "PRP":"PRON",
    "PRP$":"PRON",
    "RB":"ADV",
    "RBR":"ADV",
    "RBS":"ADV",
    "RP":"ADP",
    "SYM":"SYM",
    "TO":"PART",
    "UH":"INTJ",
    "VB":"VERB",
    "VBD":"VERB",
    "VBG":"VERB",
    "VBN":"VERB",
    "VBP":"VERB",
    "VBZ":"VERB",
    "WDT":"DET",
    "WP":"PRON",
    "WP$":"PRON",
    "WRB":"ADV",
    "XX":"X"
}

SHORT_LEMMA_POS = ["n", "a", "v"]

NEGATION_WORDS = ["no", "not", "neither", "never", "nobody", "none", "nor", "nothing", "nowhere"]
NEGATION_PREFIXES = [ "a", "dis", "il", "im", "in", "ir", "non", "un"]

class TokenAnnotatorEnglish(object):
    WORDNET_TAGS = {"ADJ": wordnet.ADJ,
                "NOUN": wordnet.NOUN,
                "PROPN": wordnet.NOUN,
                "VERB": wordnet.VERB,
                "ADV": wordnet.ADV,
                "ADP": wordnet.ADV}

    def __init__(self):
        self.stanfordCoreNLP = StanfordCoreNLP('http://localhost', port=9500)
        self.spacyNLP = spacy.load('en_core_web_sm')
        with open(join(FILE_PATH,"infVocab.json"), "r") as f:
            self.infoVocab = json.load(f)
        self.spellchecker = hunspell.HunSpell('/usr/share/hunspell/en_US.dic', '/usr/share/hunspell/en_US.aff')
        self.lemmatizer = WordNetLemmatizer()

    def getAlignmentAnnotation(self, text):
        annotation = self.stanfordCoreNLP.annotate(text, properties={
            'annotators': 'ssplit, depparse, ner, lemma',
            'pipelineLanguage':'en',
            'outputFormat':'json'
        })
        annotation = json.loads(annotation)
        return {"sentences":[
            {
                "text": "".join([token["originalText"] + " " if((i+1)<(len(sent["tokens"])) and sent["tokens"][i+1]["characterOffsetBegin"]>token["characterOffsetEnd"]) else token["originalText"] for i,token in enumerate(sent["tokens"])]),
                "dependencies": [
                    ["root" if(dep["dep"]=="ROOT") else dep["dep"], dep["governorGloss"]+"-"+str(dep["governor"]), dep["dependentGloss"]+"-"+str(dep["dependent"])] for dep in sent["basicDependencies"] if dep["dep"]!="punct"
                ],
                "words": [[token["originalText"],{
                    "CharacterOffsetBegin": str(token["characterOffsetBegin"]),
                    "CharacterOffsetEnd": str(token["characterOffsetEnd"]),
                    "Lemma": token["lemma"],
                    "NamedEntityTag": token["ner"],
                    "PartOfSpeech": token["pos"]
                }] for token in sent["tokens"]]
            } for sent in annotation["sentences"]
        ]}

    def isInfoVocab(self,word):
        if(word in self.infoVocab): return self.infoVocab[word]
        if(word[0].isupper()): word = word.lower()
        if(word[0].islower()): word = word.upper()
        if(word in self.infoVocab): return self.infoVocab[word]
        return False

    def spellcheck(self, token, questionVocab):
        if(len(token["text"]) > 2 and token["lemmas"][0] == token["text"]):
            if(not(self.spellchecker.spell(token["text"]))):
                suggestions = self.spellchecker.suggest(token["text"])
                if(len(suggestions)>0):
                    for sug in suggestions:
                        if(sug in questionVocab):
                            newPos = PEN_UD_MAP[pos_tag([sug])[0][1]]
                            token["text"] = sug
                            token["lemmas"] = [sug]
                            token["lemmaPos"] = [newPos]
                            return
                    distances = damerau_levenshtein_distance_ndarray(token["text"], np.array(suggestions))
                    val, idx = min((val, idx) for (idx, val) in enumerate(distances))
                    newWord = suggestions[idx]
                    newPos = PEN_UD_MAP[pos_tag([newWord])[0][1]]
                    # TODO take most frequent word or word in vocabulary
                    if(newPos in ["ADJ", "NOUN", "VERB", "ADV", "ADP"]):
                        newLemma = self.lemmatizer.lemmatize(newWord, TokenAnnotatorEnglish.WORDNET_TAGS[newPos])
                    else:
                        newLemma = newWord
                    token["text"] = newWord
                    token["lemmas"] = [newLemma]
                    token["lemmaPos"] = [newPos]


    def annotateText(self, text, questionVocab={}):
        text = " ".join(text.split())
        text = text.replace("won't", "will not")
        text = text.replace("n't", " not")
        text = text.replace('n"t', " not")
        spacyAnn = self.spacyNLP(text)
        coreNlpText = ""
        tokens = []
        for sent in spacyAnn.sents:
            for spacyToken in sent:
                mergedToken = {
                    "text": spacyToken.text,
                    "spacyPos": spacyToken.tag_,
                    "lemmaPos": [PEN_UD_MAP[spacyToken.tag_]],
                    "lemmas": [spacyToken.lemma_],
                    "id":spacyToken.i,
                    "headId":spacyToken.head.i,
                    "dep":spacyToken.dep_
                }
                tokens.append(mergedToken)
                coreNlpText += spacyToken.text + " "
            coreNlpText = coreNlpText[:-1] + "\n"
        coreNlpText = coreNlpText[:-2]

        # add coreNlp Annotation
        coreNlpAnn = self.stanfordCoreNLP.pos_tag_lemma(coreNlpText)
        for i,(word,pos,lemma) in enumerate(coreNlpAnn):
            token = tokens[i]
            lemmaPos = PEN_UD_MAP[pos]
            if(not(lemma in token["lemmas"])):
                token["lemmas"].append(lemma)
            if(not(lemmaPos in token["lemmaPos"])):
                token["lemmaPos"].append(lemmaPos)
            token["coreNlpPos"] = pos

        for token in tokens:
            self.addLemmas(token, questionVocab)
            self.addSlPos(token)
            self.lower(token)
            self.addNegationInfo(token)
        for token in tokens:
            token["head"] = tokens[token["headId"]]["lemmas"][0] if(tokens[token["headId"]]["id"] == token["headId"]) else print("Head Exception")
        return tokens

    def addNegationInfo(self, token):
        if(token["lemmas"][0] in NEGATION_WORDS):
            token["negWord"] = True
            return
        for prefix in NEGATION_PREFIXES:
            if(token["lemmas"][0].startswith(prefix) and self.spellchecker.spell(token["lemmas"][0][len(prefix):])):
                token["negPrefix"] = True
                return

    def addLemmas(self, token, questionVocab):
        infoLemma = self.isInfoVocab(token["text"])
        if(infoLemma):
            token["lemmas"] =  [infoLemma]
        else:
            self.spellcheck(token, questionVocab)
        token["udPos"] = token["lemmaPos"][-1]

    def addSlPos(self, token):
        for idx, set in enumerate([["PROPN","NOUN"],["ADJ","ADV"],["VERB","AUX"]]):
            if(token["udPos"] in set):
                token["slPos"] = SHORT_LEMMA_POS[idx]
                break

    def lower(self, token):
        if(token["text"][0].isupper() and not(token["udPos"] in ["PROPN", "X"] )):
            token["text"] = token["text"].lower()
            token["lemmas"] = [lem.lower() for lem in token["lemmas"]]

if __name__ == "__main__":
    tANN = TokenAnnotatorEnglish()
    tANN.annotateText("The Integer is a nuber.")
