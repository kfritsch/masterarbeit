import os, csv, math, sys, re
from os.path import join, splitext, isdir, exists, dirname, realpath, abspath
import numpy as np
import pandas as pd
from scipy import spatial
from scipy.stats.stats import pearsonr
from collections import Counter

from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from lmdb_embeddings.reader import LmdbEmbeddingsReader
from lmdb_embeddings.exceptions import MissingWordError
from lmdb_embeddings.writer import LmdbEmbeddingsWriter
from nltk.corpus import wordnet_ic
FILE_PATH = dirname(realpath(__file__))

BROWN_IC = wordnet_ic.ic('ic-brown.dat')
Synset = None

GN_TEST_PATH = join(os.path.dirname(os.path.realpath(__file__)), "germanet_test_data")

AUX = {
    "en": ["be", "can", "could", "dare", "do", "have", "may", "might", "must", "need", "ought", "shall", "should", "will", "would"],
    "de": ["sein", "haben", "werden", "können", "dürfen", "müssen", "sollen", "wollen", "mögen"]
}

def replace_umlaute(text):
    res = text
    res = res.replace('ä', 'ae')
    res = res.replace('ö', 'oe')
    res = res.replace('ü', 'ue')
    res = res.replace('Ä', 'Ae')
    res = res.replace('Ö', 'Oe')
    res = res.replace('Ü', 'Ue')
    res = res.replace('ß', 'ss')
    return res



def saveEmbeddinigsAsLMDB(modelname, lang="de"):
    path = join(SemSim.EMBEDDINGS_DIR,lang,SemSim.MODEL_PATHS[modelname])
    outpath = join(SemSim.EMBEDDINGS_DIR,lang,modelname + "LMDB")
    binary = splitext(SemSim.MODEL_PATHS[modelname])[1]==".model"
    model = KeyedVectors.load_word2vec_format(path, binary=binary)
    def iter_embeddings():
        for word in model.vocab.keys():
            yield word, model[word]
    print('Writing vectors to a LMDB database...')
    writer = LmdbEmbeddingsWriter(iter_embeddings()).write(outpath)

class SemSim(object):
    EMBEDDINGS_DIR = join(FILE_PATH, "lib","embeddings")
    MODEL_PATHS = {
        "glove": "glove_w2v.txt", # dims 853624 300, only lowercase
        "small": "small_w2v.model", # dims 608130 300,keep capitalized, replace umlaute
        "fasttext": "fasttext_w2v.txt" # dims 2000000 300, as is
    }
    PREPROCESSORS = {
        "small": lambda text: replace_umlaute(text),
        "glove": lambda text: text.lower(),
        "fasttext": lambda text: text,
    }
    VECTOR_SIZE = 300
    SYNSET_DIST_THRES = {"n":0, "v":1, "a":2}
    MIN_IC = {"en":4, "de":5}

    def __init__(self, modelname, lang="de", lmdb=True):
        self.modelname = modelname
        self.lang = lang
        self.aux = AUX[lang]
        self.lmdb = lmdb
        self.minIC = SemSim.MIN_IC[lang]
        global Synset
        if(lang=="de"):
            from .germanet import load_germanet, Synset
            self.wordNet = load_germanet(host = "localhost", port = 27027, database_name = 'germanet')
        else:
            # changed jcn_similarity to jcn_distance
            from nltk.corpus.reader.wordnet import Synset
            from nltk.corpus import wordnet
            self.wordNet = wordnet


        if(lmdb):
            self.embeddings = LmdbEmbeddingsReader(join(SemSim.EMBEDDINGS_DIR,lang,modelname + "LMDB"))
        else:
            self.model = KeyedVectors.load_word2vec_format(join(SemSim.EMBEDDINGS_DIR,lang, SemSim.MODEL_PATHS[modelname]), binary=splitext(SemSim.MODEL_PATHS[modelname])[1]==".model")
            self.model.init_sims(replace=True)

    # def setSimDF(self, question):
    #     self.vocabPath = join(SemSim.EMBEDDINGS_DIR,"questionVocab","q_" + question["id"] + "_" + modelname + ".csv")
    #     if(exists(self.vocabPath)):
    #         self.simDF = pd.read_csv(self.vocabPath)
    #     else:
    #         ppLemmas = [SemSim.PREPROCESSORS[self.modelname](w) for w in question["referenceAnswer"]["lemmas"]]
    #         self.simDF = pd.DataFrame(columns = ["WORDS"] + ppLemmas)
    #     self.simDF.set_index("WORDS",inplace=True)

    # def saveSimDF(self):
    #     if(not(exists(dirname(self.vocabPath)))):
    #         os.makedirs(dirname(self.vocabPath))
    #     self.simDF.to_csv(self.vocabPath)
    def information_content(self,synset):
        if(hasattr(synset,"infocont")):
            count = max(.0000000150070867666,synset.infocont)
        else:
            count = max(BROWN_IC[synset._pos][synset._offset]/BROWN_IC[synset._pos][0],5/BROWN_IC[synset._pos][0])
        return -math.log(count)


    def getWordsSimilarityMatrix(self, words1, words2, asDf=False):
        simMat = np.ones((len(words1), len(words2))) * 2
        w1Vectors = [self.getWordVector(w) for w in words1]
        w2Vectors = [self.getWordVector(w) for w in words2]

        for idx,wv1 in enumerate(w1Vectors):
            for jdx,wv2 in enumerate(w2Vectors):
                if(simMat[idx,jdx]!=2):
                    continue
                if(wv1 is None):
                    simMat[idx,:] = np.nan
                if(wv2 is None):
                    simMat[:,jdx] = np.nan
                if(not(wv1 is None) and not(wv2 is None)):
                    simMat[idx,jdx] = self.cosineSimilarity(wv1, wv2)
        if(asDf):
            simDF = pd.DataFrame(simMat, columns = words2)
            simDF.insert(0,'WORDS',words1)
            simDF = simDF.set_index("WORDS")
            return simDF
        return simMat

    def getTokenSimilarityMatrix(self, tokens1, tokens2, asDf=False, lemmaIdc=True):
        if(len(tokens1)==0 or len(tokens2)==0):
            return None
        simMat = np.ones((len(tokens1), len(tokens2))) * 2
        t1Vectors = [self.getTokenVector(t) for t in tokens1]
        t2Vectors = [self.getTokenVector(t) for t in tokens2]

        for idx,tok1 in enumerate(tokens1):
            for jdx,tok2 in enumerate(tokens2):
                if(simMat[idx,jdx]!=2):
                    continue
                simpleSim = self.checkSimpleCase(tok1["lemmas"][0],tok2["lemmas"][0])
                if(not(simpleSim is None)):
                    simMat[idx,jdx] = simpleSim
                    continue
                wnSim = self.getWordNetSim(tok1, tok2)
                if(not(wnSim is None)):
                    simMat[idx,jdx] = wnSim
                    continue
                vec1 = t1Vectors[idx]
                vec2 = t2Vectors[jdx]
                if(vec1 is None):
                    simMat[idx,:] = np.nan
                if(vec2 is None):
                    simMat[:,jdx] = np.nan
                if(not(vec1 is None) and not(vec2 is None)):
                    # if(tok1["text"]=="Rückgabetype" and tok2["text"]=="das"):
                    #     print(vec1, vec2)
                    #     vec1 = np.ones(len(vec1))*0.5
                    #     vec2 = np.ones(len(vec1))
                    simMat[idx,jdx] = self.cosineSimilarity(vec1, vec2)
        if(asDf):
            simDF = pd.DataFrame(simMat, columns = [token["lemmas"][0] if lemmaIdc else token["text"] for token in tokens2])
            simDF.insert(0,'WORDS',[token["lemmas"][0] if lemmaIdc else token["text"] for token in tokens1])
            simDF = simDF.set_index("WORDS")
            return simDF
        return simMat

    def getWordVector(self, word):
        ppword = SemSim.PREPROCESSORS[self.modelname](word)
        if(self.lmdb):
            try:
                return self.embeddings.get_word_vector(ppword)
            except MissingWordError:
                return None
        else:
            return self.model.get_vector(ppword) if(ppword in self.model.index2word) else None

    def getTokenVector(self, token):
        vec = self.getWordVector(token["text"])
        if(vec is None):
            vec = self.getMeanWordsEmbedding(token["lemmas"])
        if((vec is None) and "composita" in token):
            vec = self.getMeanWordsEmbedding(token["composita"])
        return vec

    def getSetenceEmbedding(self, words):
        wordVectors = []
        for w in words:
            vector = self.getWordVector(w)
            if(not(vector is None)):
                wordVectors.append(vector)
            else:
                wordVectors.append(np.zeros(SemSim.VECTOR_SIZE, dtype=np.float32))
        return np.sum(np.array(wordVectors), axis=0)

    def cosineSimilarity(self, v1, v2):
        return 1 - spatial.distance.cosine(v1, v2)

    def checkSimpleCase(self, w1, w2):
        if(w1==w2): return 1
        if(not(re.search('[^a-zA-Z]', w1)is None) or not(re.search('[^a-zA-Z]', w2) is None)):
            return int(w1 == w2)
        if(w1 in self.aux or w2 in self.aux):
            return 0
        return None

    def getTokenSimilarity(self, token1, token2, gn=True):
        simpleSim = self.checkSimpleCase(token1["lemmas"][0],token2["lemmas"][0])
        if(not(simpleSim is None)):
            return simpleSim
        if(gn and self.wordNet):
            wnSim = self.getWordNetSim(token1, token2)
            if(not(wnSim is None)):
                return wnSim
        vec1 = self.getTokenVector(token1)
        vec2 = self.getTokenVector(token2)
        if(vec1 is None or vec2 is None):
            return None
        else:
            return self.cosineSimilarity(vec1, vec2)

    def getWordSimilarity(self, word1, word2):
        vector1 = self.getWordVector(word1)
        vector2 = self.getWordVector(word2)
        if(vector1 is None or vector2 is None):
            return None
        else:
            return self.cosineSimilarity(vector1, vector2)

    def getSynsets(self,token):
        synsets = self.wordNet.synsets(token["text"])
        shortestPathInc = 0
        if(not(synsets)):
            for lemma in token["lemmas"]:
                synsets += self.wordNet.synsets(lemma)
        if(not(synsets) and "composita" in token):
            for component in token["composita"]:
                synsets += self.wordNet.synsets(component)
            shortestPathInc = 1
        return synsets, shortestPathInc

    def getLemmaRelatedSynsets(self, lemmas, relation):
        return [relatedLemma.synset for lemma in lemmas for relatedLemma in getattr(lemma, relation, [])]

    def getWordNetSim(self, tok1, tok2):
        return None
        # only words of specific POS are in wordnet
        if("slPos" in tok1 and "slPos" in tok2):
            synsets1, shortestPathInc1 = self.getSynsets(tok1)
            synsets2, shortestPathInc2 = self.getSynsets(tok2)
            # frewuent words are excluded
            for tok,synsets in [(tok1,synsets1), (tok2,synsets2)]:
                if(tok["slPos"] =="v"):
                    ics = [self.information_content(synset) for synset in synsets if(synset.pos()=="v")]
                    if(len(ics)>0 and min(ics)<self.minIC):
                        return 0
            if(synsets1 and synsets2):
                lem1 = [lem for syn in synsets1 for lem in syn.lemmas]
                lem2 = [lem for syn in synsets2 for lem in syn.lemmas]
                for relation in ["participles", "pertainyms"]:
                    synsets1 += self.getLemmaRelatedSynsets(lem1, relation)
                    synsets2 += self.getLemmaRelatedSynsets(lem2, relation)
                if(not(next(filter(set(synsets1).__contains__, synsets2), None) is None)):
                    return 1
                antonymSynsets1 = self.getLemmaRelatedSynsets(lem1, "antonyms")
                antonymSynsets2 = self.getLemmaRelatedSynsets(lem2, "antonyms")
                if(not(next(filter(set(synsets1).__contains__, antonymSynsets2), None) is None) or not(next(filter(set(antonymSynsets2).__contains__, synsets2), None) is None)):
                    return 0
                distances = []
                for ss1 in synsets1:
                    for ss2 in synsets2:
                        if(ss1.pos()[0].lower()=="v" and ss2.pos()[0].lower()=="v"):
                            continue
                        dist = Synset.shortest_path_distance(ss1, ss2)
                        if(not(dist is None)):
                            dist = max(0, dist-min(SemSim.SYNSET_DIST_THRES[ss1.pos()[0].lower()], SemSim.SYNSET_DIST_THRES[ss2.pos()[0].lower()]))
                            distances.append(dist)
                minSynsetsDist = min(distances) if len(distances) > 0 else 1000
                if(minSynsetsDist==0):
                    return 1
        return None

    def getMeanWordsEmbedding(self, words):
        wordVectors = []
        for w in words:
            vector = self.getWordVector(w)
            if(not(vector is None)):
                wordVectors.append(vector)
        if(not(wordVectors)):
            return np.zeros(SemSim.VECTOR_SIZE, dtype=np.float64)
        return np.mean(np.array(wordVectors), axis=0)

def getTigerWordCount():
    with open(join(FILE_PATH,"lib",'sdewac-gn-words.tsv'), 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter='\t')
        totalCount = 0
        for row in spamreader:
            totalCount += int(row[0])
        print(totalCount)

def testSimilarityMetrics(gurFilename):
    from customGermaLemma import CustomGermaLemma
    from germanet import load_germanet, Synset
    GERMANET = load_germanet(host = "localhost", port = 27027, database_name = 'germanet')
    semSim =SemSim("fasttext")
    germaLemmatizer = CustomGermaLemma(pickle="tiger/tiger_lemmas.pkl")
    GN_MAX_POS_DEPTH = {}
    for key,val in GERMANET.max_min_depths.items():
        GN_MAX_POS_DEPTH[key[0]] = val # {'adj': 11, 'nomen': 21, 'verben': 16}

    RESNIK_MAX_PROP = 0.04561224838 # 15196903/333175924
    GNROOT_PROP = 1 # for the GNROOT as representing all words
    RESNIK_MIN_PROP = .0000000150070867666 # 5/333175924
    SYNSET_DIST_THRES = {"n":0, "v":1, "a":0}

    SIM_NORMS = {
        "res": (0,-math.log(RESNIK_MIN_PROP)),
        "lin": (0,(2*-math.log(RESNIK_MAX_PROP))/(2*-math.log(RESNIK_MAX_PROP))),
        "jcn": (0,((2*-math.log(RESNIK_MIN_PROP))-(2*-math.log(1)))),
        "lch": dict([(GN_MAX_POS_DEPTH[key[0]], (0,-math.log(1/(2*GN_MAX_POS_DEPTH[key[0]])))) for key in GN_MAX_POS_DEPTH.keys()]),
        "sp": dict([(GN_MAX_POS_DEPTH[key[0]], (1/(2*(GN_MAX_POS_DEPTH[key[0]])),1)) for key in GN_MAX_POS_DEPTH.keys()])
    }

    def normalizeSimScores(val, metric, posDepth):
        if(metric in ["lch", "sp"]):
            (minV, maxV) = SIM_NORMS[metric][posDepth]
        else:
            (minV, maxV) = SIM_NORMS[metric]
        if(metric in ["jcn"]):
            return 1 - (val - minV) / (maxV -minV)
        if(metric in ["sp"]):
            val = 1/(val)
        return (val - minV) / (maxV -minV)

    simData = []
    with open(join("germanet_test_data",gurFilename), 'r', encoding='utf-8') as csvFile:
        csvReader = csv.reader((row for row in csvFile if not row.startswith('#')), delimiter=',', quotechar='|')
        header = next(csvReader)
        for row in csvReader:
            row[2] = float(row[2])
            simData.append(tuple(row))
    simData = np.core.records.array(simData, dtype=np.dtype({'formats': ['U30', 'U30', '<f8','U30', 'U30'],'names': header}))

    # select those words which are found in GermaNet
    wordsInGN = lambda w1, w2: bool(GERMANET.synsets(w1) and GERMANET.synsets(w2))

    sim_funcs = [('lch', Synset.lch_similarity,  np.max),
                 ('res', Synset.res_similarity,  np.max),
                 ('jcn', Synset.jcn_distance, np.min),
                 ('lin', Synset.lin_similarity,  np.max),
                 ('sp', Synset.shortest_path_distance,  np.min)]

    targets = np.zeros(len(simData))
    scores = np.zeros((len(simData),6))
    posList = []

    def tryComposita(w):
        parts = germaLemmatizer._composita_lemma(w)
        synsets = []
        if(len(parts) > 1):
            for part in parts[1:]:
                synsets += GERMANET.synsets(part)
        for synset in synsets1: synset.infocont = RESNIK_MIN_PROP
        return synsets

    def tryLemmatize(w):
        lemmas = germaLemmatizer.find_lemma(w, "NOUN")
        synsets = []
        if(lemmas):
            for lemma in lemmas:
                synsets += GERMANET.synsets(lemma)
        return synsets

    for idx, row in enumerate(simData):
        (word1, word2, human, pos1, pos2) = row
        targets[idx] = human
        posList.append(pos1 if(pos1==pos2) else "m")
        shortestPathInc = 0
        synsets1 = GERMANET.synsets(word1)
        synsets2 = GERMANET.synsets(word2)
        if(not(synsets1) and pos1=="n"):
            synsets1 = tryLemmatize(word1)
        if(not(synsets1) and pos1=="n"):
            synsets1 = tryComposita(word1)
            # print(word1, word2, synsets1)
            shortestPathInc += 1
        if(not(synsets2) and pos2=="n"):
            synsets2 = tryLemmatize(word2)
        if(not(synsets2) and pos2=="n"):
            synsets2 = tryComposita(word2)
            # print(word2, word1, synsets2)
            shortestPathInc += 1

        if(synsets1 and synsets2):
            minSynsetsDist = np.min(np.array([Synset.shortest_path_distance(ss1, ss2) + shortestPathInc for ss1 in synsets1 for ss2 in synsets2]))
            if(minSynsetsDist<=SYNSET_DIST_THRES[pos1]):
                scores[idx,:] = 1 #- (minSynsetsDist*0.05*(minSynsetsDist>=2))
            else:
                for jdx, simFuncTupl in enumerate(sim_funcs):
                    (sim_name, sim_func, comb_func) = simFuncTupl
                    synsetScores = np.array([sim_func(ss1, ss2) for ss1 in synsets1 for ss2 in synsets2 ])
                    bestScore = comb_func(synsetScores)
                    if(sim_name=="sp"):
                        bestScore += shortestPathInc

                    score = normalizeSimScores(bestScore,sim_name,max(GN_MAX_POS_DEPTH[pos1[0]],GN_MAX_POS_DEPTH[pos2[0]]))
                    scores[idx,jdx] = score
        else:
            scores[idx,:] = np.nan
        if(not(scores[idx,-1] == 1)):
            score = semSim.getWordSimilarity(word1, word2)
            if(score): scores[idx,-1] = score

    posList = np.array(posList)
    posDist = Counter(posList)
    print(posDist)
    meanReducedTargets = np.copy(targets)
    meanReducedScore = np.copy(scores)
    for pos in ["n","a","v","m"]:
        posIdx = posList==pos
        if(np.sum(posIdx) == 0): continue
        meanReducedTargets[posIdx] = targets[posIdx] - np.mean(targets[posIdx])
    simMetrics = ['lch','res','jcn','lin','sp','ft']
    for i in range(len(simMetrics)):
        print("----" + simMetrics[i] + "----")
        notNan = np.invert(np.isnan(scores[:,i]))
        if(not(gurFilename=="gur65.csv")):
            for pos in posDist.keys():
                isCurrPos = posList==pos
                posAndNotNan = np.logical_and(notNan,isCurrPos)
                meanReducedScore[posAndNotNan,i] = scores[posAndNotNan,i] - np.mean(scores[posAndNotNan,i])
                print(pos, sum(posAndNotNan), pearsonr(scores[posAndNotNan,i],targets[posAndNotNan])[0])
        print("allReducedMean",sum(notNan), pearsonr(meanReducedScore[notNan,i],meanReducedTargets[notNan])[0])

def wordDistance(w1, w2):
    GERMANET = load_germanet(host = "localhost", port = 27027, database_name = 'germanet')
    return np.min(np.array([Synset.shortest_path_distance(ss1, ss2) for ss1 in GERMANET.synsets(w1) for ss2 in GERMANET.synsets(w2)]))

# for gurFile in ["gur65.csv","gur350.csv","inf-gurxx.csv"]:
#     testSimilarityMetrics(gurFile)

# print(wordDistance("selbe", "gleich"))
# semSim = SemSim(modelname="fasttext", lang="en")
# print(semSim.getTokenSimilarity({"text":"positive","lemmas":["positive"], "slPos":"a"},{"text":"negative","lemmas":["negative"], "slPos":"a"}, gn=True))
