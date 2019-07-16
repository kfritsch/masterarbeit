import os, csv, math, sys
from os.path import join, splitext, isdir, exists, dirname, abspath
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

from germanet import load_germanet, Synset

GN_TEST_PATH = join(os.path.dirname(os.path.realpath(__file__)), "germanet_test_data")

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

def saveEmbeddinigsAsLMDB(modelname):
    path = join(SemSim.EMBEDDINGS_DIR,SemSim.MODEL_PATHS[modelname])
    outpath = join(SemSim.EMBEDDINGS_DIR,modelname + "LMDB")
    binary = splitext(SemSim.MODEL_PATHS[modelname])[1]==".model"
    model = KeyedVectors.load_word2vec_format(path, binary=binary)
    def iter_embeddings():
        for word in model.vocab.keys():
            yield word, model[word]
    print('Writing vectors to a LMDB database...')
    writer = LmdbEmbeddingsWriter(iter_embeddings()).write(outpath)

class SemSim(object):
    EMBEDDINGS_DIR = "./lib/embeddings/"
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

    def __init__(self, modelname, lmdb=True):
        self.modelname = modelname
        self.germanet = load_germanet(host = "localhost", port = 27027, database_name = 'germanet')

        self.lmdb = lmdb
        if(lmdb):
            self.embeddings = LmdbEmbeddingsReader(join(SemSim.EMBEDDINGS_DIR,modelname + "LMDB"))
        else:
            self.model = KeyedVectors.load_word2vec_format(join(SemSim.EMBEDDINGS_DIR, SemSim.MODEL_PATHS[modelname]), binary=splitext(SemSim.MODEL_PATHS[modelname])[1]==".model")
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
        simMat = np.ones((len(tokens1), len(tokens2))) * 2
        t1Vectors = [self.getTokenVector(t) for t in tokens1]
        t2Vectors = [self.getTokenVector(t) for t in tokens2]

        for idx,tok1 in enumerate(tokens1):
            for jdx,tok2 in enumerate(tokens2):
                if(simMat[idx,jdx]!=2):
                    continue
                if(self.isSynset(tok1, tok2)):
                    simMat[idx,jdx] = 1
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
                wordVectors.append(wordVectors)
            else:
                wordVectors.append(np.zeros(SemSim.VECTOR_SIZE, dtype=np.float64))
        return np.sum(wordVectors, axis=0)

    def cosineSimilarity(self, v1, v2):
        return 1 - spatial.distance.cosine(v1, v2)

    def getTokenSimilarity(self, token1, token2, gn=True):
        if(gn and self.germanet and self.isSynset(token1, token2)):
            return 1
        vec1 = self.getTokenVector(token1)
        vec2 = self.getTokenVector(token2)
        if(vec1 is None or vec2 is None):
            return None
        else:
            return self.cosineSimilarity(vector1, vector2)

    def getWordSimilarity(self, word1, word2):
        vector1 = self.getWordVector(word1)
        vector2 = self.getWordVector(word2)
        if(vector1 is None or vector2 is None):
            return None
        else:
            return self.cosineSimilarity(vector1, vector2)

    def getSynsets(self,token):
        synsets = self.germanet.synsets(token["text"])
        shortestPathInc = 0
        if(not(synsets)):
            for lemma in token["lemmas"]:
                synsets += self.germanet.synsets(lemma)
        if(not(synsets) and "composita" in token):
            for component in token["composita"]:
                synsets += self.germanet.synsets(component)
            shortestPathInc = 1
        return synsets, shortestPathInc

    def isSynset(self, token1, token2):
        if("slPos" in token1 and "slPos" in token2 and token1["slPos"] == token2["slPos"]):
            synsets1, shortestPathInc1 = self.getSynsets(token1)
            synsets2, shortestPathInc2 = self.getSynsets(token2)
            incs = shortestPathInc1 + shortestPathInc2
            if(synsets1 and synsets2):
                minSynsetsDist = np.min(np.array([Synset.shortest_path_length(ss1, ss2) + incs for ss1 in synsets1 for ss2 in synsets2]))
                if(minSynsetsDist<=SemSim.SYNSET_DIST_THRES[token1["slPos"]]):
                    return True
        return False

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
    with open(join(FILE_PATH,os.path.pardir,'sdewac-gn-words.tsv'), 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter='\t')
        totalCount = 0
        for row in spamreader:
            totalCount += int(row[0])
        print(totalCount)

def testSimilarityMetrics(gurFilename):
    from customGermaLemma import CustomGermaLemma
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
    print(len(simData))
    simData = np.core.records.array(simData, dtype=np.dtype({'formats': ['U30', 'U30', '<f8','U30', 'U30'],'names': header}))

    # select those words which are found in GermaNet
    wordsInGN = lambda w1, w2: bool(GERMANET.synsets(w1) and GERMANET.synsets(w2))

    sim_funcs = [('lch', Synset.sim_lch,  np.max),
                 ('res', Synset.sim_res,  np.max),
                 ('jcn', Synset.dist_jcn, np.min),
                 ('lin', Synset.sim_lin,  np.max),
                 ('sp', Synset.shortest_path_length,  np.min)]

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
            minSynsetsDist = np.min(np.array([Synset.shortest_path_length(ss1, ss2) + shortestPathInc for ss1 in synsets1 for ss2 in synsets2]))
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
    return np.min(np.array([Synset.shortest_path_length(ss1, ss2) for ss1 in GERMANET.synsets(w1) for ss2 in GERMANET.synsets(w2)]))

# for gurFile in ["gur65.csv","gur350.csv","inf-gurxx.csv"]:
#     testSimilarityMetrics(gurFile)

# print(wordDistance("selbe", "gleich"))
