import sys, math, re
import numpy as np
import pandas as pd
import copy
from scipy.stats.stats import pearsonr
from os.path import join, dirname, realpath
import cProfile, pstats, io
from nltk.corpus import wordnet_ic
brown_ic = wordnet_ic.ic('ic-brown.dat')

from .TokenAnnotator import TokenAnnotator
from .TokenAnnotatorEnglish import TokenAnnotatorEnglish
from .SemSim import SemSim
from .sultanAlign.align import suAlignmentScore

FILE_PATH = dirname(realpath(__file__))

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

class FeatureExtractor(object):
    QUESTION_TYPES = {
        "definition": 0,
        "reason": 1,
        "procedure": 2,
        "comparison": 3
    }
    LABEL_DICT = {
        "correct": 0,
        "binary_correct": 1,
        "partially_correct": 2,
        "missconception": 3,
        "concept_mix-up": 4,
        "guessing": 5,
        "none": 6
    }
    EMB_SIM_THRES= 0.7
    GN_SIM_THRES= 0.9
    DIST_WEIGHTS_MIN= 3
    SYN_THRES = {
        "a":0.7,
        "n":0.7,
        "v":0.7,
        "o":0.7
    }

    def __init__(self, simMeasure="small", lang="de"):
        if(lang=="de"):
            self.tokenAnnotator = TokenAnnotator()
        else:
            self.tokenAnnotator = TokenAnnotatorEnglish()
        self.semSim = SemSim(modelname=simMeasure, lang=lang, lmdb=True)
        with open(join(FILE_PATH, "lib", "stopwords", lang + ".txt"), "r") as txtFile:
            self.stopwords = [line.strip() for line in txtFile]

    # def setSimMeasure(self, simMeasure):
    #     self.semSim = SemSim(simMeasure)
    #     if(self.question):
    #         self.semSim.setSimDF(self.question)

    def setQuestion(self, question, resetVocabulary=True, simMatrix=None):
        self.question = question
        self.questionDemotion = False
        self.simMatrix = simMatrix
        if(not("stopwordExceptions" in self.question)):
            self.question["stopwordExceptions"] = []
        if(not("tokens" in self.question)):
            self.annotateAnswer(self.question)
        if("assignedWeights" in self.question):
            self.setAnnWeights(self.question["assignedWeights"])
            self.questionDemotion = True
        self.vocabDistWeights = None
        # self.semSim.setSimDF(self.question)
        self.questionContentLemmas = dict.fromkeys([token["lemmas"][0] for token in self.question["tokens"] if(token["contentWord"])], 0)
        self.questionFilterLemmas = {}
        if(self.questionDemotion):
            self.questionFilterLemmas = self.question["assignedWeights"]

        if(not("vocabulary" in self.question) or resetVocabulary):
            self.resetVocabulary()
        else:
            columns = self.question["vocabulary"].keys()
            data = [np.array(vocab["simScores"], dtype=np.float) for lemma, vocab in self.question["vocabulary"].items()]
            self.simFrame = pd.DataFrame(np.array(data),index=columns, columns=columns)

    def prepareReferenceAnswers(self, useCorrectAsRef=False):
        self.referenceAnswers = self.question["referenceAnswers"]
        if(useCorrectAsRef):
            self.referenceAnswers += self.question["studentReferenceAnswers"]

        self.rATSs = []
        for refAns in self.referenceAnswers:
            if(not("tokens" in refAns)):
                self.annotateAnswer(refAns)
            self.rATSs.append(self.getTokenSelections(refAns))
            qRefOverlap = dict.fromkeys(["a","n","v","o"], 0)
            for token in self.rATSs[-1]["contentToken"]:
                if(token["lemmas"][0] in self.questionContentLemmas):
                    qRefOverlap[token["slPos"] if "slPos" in token else "o"] += 1
            self.rATSs[-1]["qRefOverlap"] = qRefOverlap

    def setAnnWeights(self, assignedWeights):
        meanCWW = np.mean([val for key,val in assignedWeights.items()])
        self.annWeights = {}
        for key,val in assignedWeights.items():
            self.annWeights[key] = (val-meanCWW)+1

    def setDistWeights(self):
        nC, pC = self.question["classCounts"]
        self.vocabDistWeights= {}
        for key,val in self.question["vocabulary"].items():
            posRatio = val["predDist"][1]/pC if pC else 0
            negRatio = val["predDist"][0]/nC if nC else 0
            if(sum(val["predDist"])>=FeatureExtractor.DIST_WEIGHTS_MIN):
                self.vocabDistWeights[key] = [posRatio-negRatio, sum(val["predDist"])/np.sum(self.question["classCounts"])]
        self.semCatDistWeights = {}
        for lemma,entry in self.question["semCats"].items():
            if(sum(entry["predDist"])>=FeatureExtractor.DIST_WEIGHTS_MIN):
                posRatio = entry["predDist"][1]/pC if pC else 0
                negRatio = entry["predDist"][0]/nC if nC else 0
                self.semCatDistWeights[lemma] = {
                    "dist": posRatio-negRatio,
                    "weight": sum(entry["predDist"])/np.sum(self.question["classCounts"])
                }
                headDict = {}
                for key,val in entry["headCats"].items():
                    weight = sum(val)/sum(self.question["classCounts"])
                    posRatio = val[1]/pC if pC else 0
                    negRatio = val[0]/nC if nC else 0
                    # TODO why worse dist
                    dist = posRatio - negRatio if(sum(val)>=FeatureExtractor.DIST_WEIGHTS_MIN) else self.semCatDistWeights[lemma]["dist"] * (sum(val)/sum(entry["predDist"]))
                    headDict[key] = [dist, weight]
                self.semCatDistWeights[lemma]["headDistWeights"] = headDict

    def resetVocabulary(self):
        self.question["vocabulary"] = {}
        self.question["semCats"] = {}
        self.question["classCounts"] = [0,0]
        self.simFrame = pd.DataFrame()
        for refAnswer in self.question["referenceAnswers"]:
            self.updateVocab(refAnswer,refTok=True, allowMerge=True)
            self.updateVocabPredDist(refAnswer, "correct")

    def getTokenSelections(self,answer):
        tokenSelections = {
            "words": [],
            "wordsSet": [],
            "allContentToken": [],
            "tokenSet": [],
            "contentToken": [],
            "contentLemmas": [],
            "contentPos": []
        }
        for token in answer["tokens"]:
            tokenSelections["words"].append(token["text"])
            tokenText = token["text"]
            if(not(tokenText in tokenSelections["wordsSet"])):
                tokenSelections["wordsSet"].append(tokenText)
                tokenSelections["tokenSet"].append(token)
            lemma = token["lemmas"][0]
            if(token["contentWord"] and not(lemma in self.questionFilterLemmas and self.questionFilterLemmas[lemma]==0)):
                tokenSelections["allContentToken"].append(token)
                if(not(lemma in tokenSelections["contentLemmas"])):
                    tokenSelections["contentLemmas"].append(lemma)
                    tokenSelections["contentToken"].append(token)
                    tokenSelections["contentPos"].append(token["udPos"])
        return tokenSelections

    def extractFeatures(self, pupAns, predDist, vocab=False, bins=6):
        if(not("tokens" in pupAns)):
            self.annotateAnswer(pupAns)

        pATS = self.getTokenSelections(pupAns)
        pupAns["features"] = self.getEmptyFeatureDict(predDist, vocab)

        if(len(self.referenceAnswers)>1 and len(pATS["contentToken"]) > 0):
            maxSim = 0
            refIdx = 0
            pupAnsFrame = self.simFrame.loc[pATS["contentLemmas"]]
            for rIdx, refAns in enumerate(self.referenceAnswers):
                if(pupAns["id"] == refAns["id"]):
                    continue
                rATS = self.rATSs[rIdx]
                if(len(rATS["contentLemmas"]) > 0):
                    contentSimDf = pupAnsFrame[rATS["contentLemmas"]]
                    similarWordsRecall = self.getMaxWordsSim(contentSimDf,rATS["contentToken"], axis=0)
                    sim = self.getContentScore(similarWordsRecall, refIdx=0, weights=False)
                    if(sim>maxSim):
                        refIdx = rIdx
                        maxSim = sim
        else:
            refIdx = 0
        refAns = self.referenceAnswers[refIdx]
        rATS = self.rATSs[refIdx]
        pupAns["refAnsId"] = refAns["id"]
        pupAns["features"] = self.getEmptyFeatureDict(predDist, vocab)
        pupAns["features"]["qRefOverlap"] = rATS["qRefOverlap"]
        self.addCountFeatures(pupAns, refAns, pATS, rATS)
        self.addPosDistFeatures(pupAns, refAns)
        if(len(pATS["tokenSet"]) > 0):
            self.addSentenceEmbeddingFeatures(pupAns, pATS["words"], rATS["words"])
            # self.addSulAlign(pupAns, refAns, pATS, rATS)
        if(len(pATS["contentToken"]) > 0):
            self.addContentSimilarityFeatures(pupAns, pATS, rATS, bins=bins)
            if(predDist):
                self.addPredDistBasedFeatures(pupAns, pATS, bins=bins)
            if(vocab):
                self.addVocabFeatures(pupAns)

    def getEmptyFeatureDict(self, predDist=False, vocab=False, bins=6):
        features = {
            # "pupAbsLen": 0,
            # "refAbsLen": 0,
            # "absLenDiff": 0,
            # "pupContentLen": 0,
            # "refContentLen":0,
            # "contentLenDiff": 0,
            # "pupNegWordCount": 0,
            # "pupNegPrefixCount": 0,
            # "refNegWordCount": 0,
            # "refNegPrefixCount": 0,
            # "pupPosDist": dict.fromkeys(["AD","NOUN","VERB","CONJ","OTHER"], 0),
            # "refPosDist": dict.fromkeys(["AD","NOUN","VERB","CONJ","OTHER"], 0),
            "lemmaRec": 0,
            "lemmaHeadRec": 0,
            "contentRec": 0,
            "contentHeadRec": 0,
            "contentPrec": 0,
            "sulAlign": [0,0],
            "simHist": [0]*bins,
            "posSimHist": {"a":[0]*bins, "n":[0]*bins, "v":[0]*bins, "o":[0]*bins},
            "embDiff": 1,
            "embSim":0,
            "embMQDiff": 1,
            "embMQSim":0,
            # "qRefOverlap": dict.fromkeys(["a","n","v","o"], 0),
            "qRefAnsOverlap": dict.fromkeys(["a","n","v","o"], 0)
        }
        if(hasattr(self, 'annWeights')):
            features["annWContentPrec"] = 0
            features["annWContentRec"] = 0
        if("type" in self.question):
            features["qType"] = FeatureExtractor.QUESTION_TYPES[self.question["type"]]
        if(predDist):
            features["distWeightContent"] = 0
            features["distWeightHeadContent"] = 0
            features["posDistWeightHist"] = {"a":[0]*bins, "n":[0]*bins, "v":[0]*bins, "o":[0]*bins}
        if(vocab):
            features["vocab"] = dict.fromkeys([key for key, val in self.question["vocabulary"].items() if "first" in val and sum(val["predDist"]) > 1], 0)
            features["semCats"] = dict.fromkeys([key for key, val in self.question["semCats"].items() if "first" in val and sum(val["predDist"]) > 1], 0)
            features["semCatHeads"] = dict.fromkeys([(semkey + "_" + headkey) for semkey, semval in self.question["semCats"].items() for headkey, headval in semval["headCats"].items() if "first" in semval and sum(headval) > 1], 0)
        return features

    def addCountFeatures(self, pupAns, refAns, pATS, rATS):
        pupAns["features"]["pupAbsLen"] = len(pATS["words"])
        pupAns["features"]["refAbsLen"] = len(rATS["words"])
        pupAns["features"]["absLenDiff"] = len(pATS["words"]) - len(rATS["words"])
        pupAns["features"]["pupContentLen"] = len(pATS["contentToken"])
        pupAns["features"]["refContentLen"] = len(rATS["contentToken"])
        pupAns["features"]["contentLenDiff"] = len(pATS["contentToken"]) - len(rATS["contentToken"])
        pupAns["features"]["pupNegWordCount"] = sum(["negWord" in token for token in pupAns["tokens"]])
        pupAns["features"]["pupNegPrefixCount"] = sum(["negPrefix" in token for token in pupAns["tokens"]])
        pupAns["features"]["refNegWordCount"] = sum(["negWord" in token for token in refAns["tokens"]])
        pupAns["features"]["refNegPrefixCount"] = sum(["negPrefix" in token for token in refAns["tokens"]])

    def addPosDistFeatures(self, pupAns, refAns):
        pupAns["features"]["pupPosDist"] = self.getPosDist(pupAns["tokens"])
        pupAns["features"]["refPosDist"] = self.getPosDist(refAns["tokens"])

    def addSentenceEmbeddingFeatures(self, pupAns, pupWords, refWords):
        pupEmb = self.semSim.getSetenceEmbedding(pupWords)
        refEmb = self.semSim.getSetenceEmbedding(refWords)
        questionEmb = self.semSim.getSetenceEmbedding([token["text"] for token in self.question["tokens"]])
        pupMQEmb = pupEmb - questionEmb
        refMQEmb = refEmb - questionEmb
        pupAns["features"]["embDiff"] = float(np.mean(np.abs(pupEmb-refEmb)))
        pupAns["features"]["embSim"] = float(self.semSim.cosineSimilarity(pupEmb,refEmb))
        pupAns["features"]["embMQDiff"] = float(np.mean(np.abs(pupMQEmb-refMQEmb)))
        pupAns["features"]["embMQSim"] = float(self.semSim.cosineSimilarity(pupMQEmb,refMQEmb))

    def addContentSimilarityFeatures(self, pupAns, pATS, rATS, bins=6):
        # TODO: consider embedding difference rank
        # self.contentSimDf = self.semSim.getTokenSimilarityMatrix(pATS["contentToken"], rATS["contentToken"], asDf=True, lemmaIdc=True)
        contentSimDf = self.simFrame[rATS["contentLemmas"]].loc[pATS["contentLemmas"]]
        similarWordsRecall = self.getMaxWordsSim(contentSimDf,rATS["contentToken"], axis=0)
        similarWordsPrecision = self.getMaxWordsSim(contentSimDf,pATS["contentToken"],  axis=1)

        pupAns["features"]["lemmaRec"] = sum([1 for lemma in rATS["contentLemmas"] if lemma in pATS["contentLemmas"]])
        pupAns["features"]["lemmaHeadRec"] = self.getHeadRecall(pATS["allContentToken"], rATS["allContentToken"])
        pupAns["features"]["contentRec"] = self.getContentScore(similarWordsRecall, refIdx=0, weights=False)
        pupAns["features"]["contentHeadRec"] = self.getContentHeadScore(similarWordsRecall, pATS["allContentToken"], rATS["allContentToken"])
        pupAns["features"]["contentPrec"] = self.getContentScore(similarWordsPrecision, refIdx=2, weights=False)
        pupAns["features"]["posSimHist"] = self.getPosTagSimHist(similarWordsRecall, rATS["contentPos"], bins=bins)
        pupAns["features"]["simHist"] = self.getSimHist(similarWordsRecall, bins=bins)
        pupAns["features"]["qRefAnsOverlap"] = self.getQRefAnsOverlap(similarWordsRecall)
        # if(hasattr(self, 'annWeights')):
        #     pupAns["features"]["annWContentPrec"] = self.getContentScore(similarWordsPrecision, refIdx=2, weights=True)
        #     pupAns["features"]["annWContentRec"] = self.getContentScore(similarWordsRecall, refIdx=0, weights=True)
        #     pupAns["features"]["annWContentHeadRec"] = self.getContentHeadScore(similarWordsRecall, pATS["allContentToken"], rATS["allContentToken"], weights=True)

    def addPredDistBasedFeatures(self, answer, pATS, bins=6):
        if(not(self.vocabDistWeights)):
            self.setDistWeights()
        answer["features"]["distWeightHeadContent"] = self.getDistWheightHeadContent(pATS["allContentToken"])
        answer["features"]["distWeightContent"] = self.getDistWeightContent(answer, pATS)
        answer["features"]["posDistWeightHist"] = self.getPosDistWeightHist(answer, bins=bins)

    def addVocabFeatures(self, answer):
        vocab = dict.fromkeys([key for key, val in self.question["vocabulary"].items() if "first" in val and sum(val["predDist"]) > 1], 0)
        semCats = dict.fromkeys([key for key, val in self.question["semCats"].items() if "first" in val and sum(val["predDist"]) > 1], 0)
        semCatHeads = dict.fromkeys([(semkey + "_" + headkey) for semkey, semval in self.question["semCats"].items() for headkey, headval in semval["headCats"].items() if "first" in semval and sum(headval) > 1], 0)
        for token in answer["tokens"]:
            lemma = token["lemmas"][0]
            if(lemma in vocab):
                vocab[lemma] = 1
                semCat = self.question["vocabulary"][lemma]["semCat"]
                semCats[semCat] = 1
                headLemma = token["head"]
                if(headLemma in self.question["vocabulary"]):
                    headLemma = self.question["vocabulary"][headLemma]["semCat"]
                    semCatHead = semCat + "_" + headLemma
                    if(semCatHead in semCatHeads):
                        semCatHeads[semCatHead] = 1
        answer["features"]["vocab"] = vocab
        answer["features"]["semCats"] = semCats
        answer["features"]["semCatHeads"] = semCatHeads

    def getHeadRecall(self, pupToken, refToken):
        lemmaHeadMatch = 0
        for rTok in refToken:
            refLemma = rTok["lemmas"][0]
            for pTok in pupToken:
                pupLemma = pTok["lemmas"][0]
                if(refLemma == pupLemma):
                    if(rTok["head"] ==  pTok["head"]):
                        lemmaHeadMatch +=1
                        break
        return lemmaHeadMatch

    def getQRefAnsOverlap(self, similarWordsRecall):
        qraOverlap = dict.fromkeys(["a","n","v","o"], 0)
        for (rWord, simScore, pWord, slPos) in similarWordsRecall:
            if(simScore>FeatureExtractor.SYN_THRES[slPos] and (rWord in self.questionContentLemmas)):
                qraOverlap[slPos] += 1
        return qraOverlap

    def addSulAlign(self, pupAns, refAns, pATS=None, rATS=None):
        if(pATS is  None):
            pATS = self.getTokenSelections(pupAns)
        if(rATS is  None):
            rATS = self.getTokenSelections(refAns)

        # subSimFrame = self.simFrame[pATS["contentLemmas"]].loc[rATS["contentLemmas"]]
        # subSimFrame.rename(columns=dict(zip(pATS["contentLemmas"],[token["text"] for token in pATS["contentToken"]])),
        #          index=dict(zip(rATS["contentLemmas"],[token["text"] for token in rATS["contentToken"]])),
        #          inplace=True)

        allSimDfWords = self.semSim.getTokenSimilarityMatrix(pATS["tokenSet"], rATS["tokenSet"], asDf=True, lemmaIdc=False)
        pupAns["features"]["sulAlign"] = suAlignmentScore(refAns, pupAns, self.question, allSimDfWords)[0:2]

    def getDistWeightContent(self, answer, pATS):
        return sum([self.vocabDistWeights[token["lemmas"][0]][0] * self.vocabDistWeights[token["lemmas"][0]][1] for token in pATS["allContentToken"] if token["lemmas"][0] in self.vocabDistWeights])/len(pATS["allContentToken"])

    def getDistWheightHeadContent(self, contentToken):
        scores = []
        for token in contentToken:
            lemma = token["lemmas"][0]
            head = token["head"]
            lemmaSemCat = self.question["vocabulary"][lemma]["semCat"]
            headSemCat = self.question["vocabulary"][head]["semCat"]
            if(lemmaSemCat in self.semCatDistWeights):
                if(headSemCat in self.semCatDistWeights[lemmaSemCat]["headDistWeights"]):
                    scores.append(self.semCatDistWeights[lemmaSemCat]["weight"]*self.semCatDistWeights[lemmaSemCat]["headDistWeights"][headSemCat][0])
                else:
                    # TODO why these denums
                    scores.append((1/sum(self.question["classCounts"]))*(self.semCatDistWeights[lemmaSemCat]["dist"]*(1/sum(self.question["semCat"][lemmaSemCat]["predDist"]))))
        return sum(scores)/len(contentToken)

    def getPosDistWeightHist(self, answer, bins=6):
        posDistWeightHist = {"a":[0]*bins, "n":[0]*bins, "v":[0]*bins, "o":[0]*bins}
        for token in answer["tokens"]:
            if(not(token["lemmas"][0] in self.vocabDistWeights)):
                continue
            tokenDist,tokenWeight = self.vocabDistWeights[token["lemmas"][0]]
            slPos = token["slPos"] if "slPos" in token else "o"
            binIdx = min(math.floor((tokenDist+1)/2*bins),bins-1)
            posDistWeightHist[slPos][binIdx] += tokenWeight
        return posDistWeightHist

    def simMatToDf(self, simMat, rows, columns):
        simDF = pd.DataFrame(simMat, columns = columns)
        simDF.insert(0,'WORDS',rows)
        simDF = simDF.set_index("WORDS")
        return simDF

    def getContentScore(self, simWordsTuples, refIdx=0, weights=False):
        if(weights):
            if(not(self.annWeights)):
                self.setAnnWeights(self.question["referenceAnswer"])
            return sum(np.array([(simTup[1]>FeatureExtractor.SYN_THRES[simTup[3]]) * self.annWeights[simTup[refIdx]] for simTup in simWordsTuples if simTup[refIdx]]))/len(simWordsTuples)
        else:
            return sum(np.array([simTup[1]>FeatureExtractor.SYN_THRES[simTup[3]] for simTup in simWordsTuples]))/len(simWordsTuples)

    def getContentHeadScore(self, simWordsTuples,pupToken, refToken, weights=False):
        scores = []
        for (refLemma, simScore, pupLemma, slPos) in simWordsTuples:
            if(simScore>FeatureExtractor.SYN_THRES[slPos]):
                pToken = [token for token in pupToken if (token["lemmas"][0] == pupLemma)]
                rToken = [token for token in refToken if (token["lemmas"][0] == refLemma)]
                for rTok in rToken:
                    for pIdx,pTok in enumerate(pToken):
                        refHead = rTok["head"]
                        pupHead = pTok["head"]
                        refHeadSemCat = self.question["vocabulary"][refHead]["semCat"]
                        pupHeadSemCat = self.question["vocabulary"][pupHead]["semCat"]
                        if(refHeadSemCat == pupHeadSemCat):
                            scores.append(self.annWeights[refLemma] if(weights) else 1)
                            del pToken[pIdx]
                            break
        return sum(scores)/len(refToken)


    def getPosTagSimHist(self, similarWordsRecall, refContentPos, bins=6):
        posTagSimHist = {"a":[0]*bins, "n":[0]*bins, "v":[0]*bins, "o":[0]*bins}
        for idx,pos in enumerate(refContentPos):
            if(similarWordsRecall[idx][1] == 0):
                binIdx = 0
            else:
                binIdx = math.floor(similarWordsRecall[idx][1]*(bins-2))+1
            slPos = similarWordsRecall[idx][3]
            posTagSimHist[slPos][binIdx] += 1/len(refContentPos)
        return posTagSimHist

    def getSimHist(self, similarWordsRecall, bins=6):
        simHist = [0]*bins;
        for sim in similarWordsRecall:
            binIdx = math.floor(sim[1]*(bins-2))+1 if(sim[1]) else 0
            simHist[binIdx] += 1/len(similarWordsRecall)
        return simHist

    def getMaxWordsSim(self, dataFrame, sourceTokens, axis=0):
        maxIdc = dataFrame.idxmax(axis=axis)
        sourceWords = dataFrame.columns if(axis==0) else dataFrame.index
        targetWords = list(dataFrame.index) if(axis==0) else list(dataFrame.columns)
        return [(sourceWords[idx],dataFrame.loc[sourceWords[idx]][maxIdx] if(axis==1) else dataFrame.loc[maxIdx][sourceWords[idx]],maxIdx, sourceTokens[idx]["slPos"] if "slPos" in sourceTokens[idx] else "o") if isinstance(maxIdx,str) else (sourceWords[idx], 0, None, sourceTokens[idx]["slPos"] if "slPos" in sourceTokens[idx] else "o") for idx,maxIdx in enumerate(maxIdc)]

    def getPosDist(self, tokens):
        posDist = dict.fromkeys(["AD","NOUN","VERB","CONJ","OTHER"], 0)
        for token in tokens:
            p = token["udPos"]
            if(p[0:2]=="AD"):
                posDist["AD"] += 1
            elif(p in ["PROPN","NOUN"]):
                posDist["NOUN"] += 1
            elif(p in ["AUX","VERB"]):
                posDist["VERB"] += 1
            elif(p[-4:]=="CONJ"):
                posDist["CONJ"] += 1
            else:
                posDist["OTHER"] += 1
        return posDist

    def isContentWord(self, word):
        return ((word.isdigit() or len(word)>1) and not(word in self.stopwords)) or word in self.question["stopwordExceptions"]

    def annotateAnswer(self, answer):
        answer["tokens"] =  self.tokenAnnotator.annotateText(answer["correctionOrComment"] if("correctionOrComment" in answer) else answer["text"], self.question["vocabulary"] if "vocabulary" in self.question else {})
        for token in answer["tokens"]:
            token["contentWord"] = self.isContentWord(token["lemmas"][0])
        for token in answer["tokens"]:
            if(token["contentWord"]):
                headToken = answer["tokens"][token["headId"]]
                if(not(headToken["contentWord"])):
                    while(not(headToken["contentWord"]) and not(headToken["lemmas"][0]==headToken["head"])):
                        headToken = answer["tokens"][headToken["headId"]]
                    if(headToken["contentWord"]):
                        token["head"] = headToken["lemmas"][0]
                        token["headId"] = headToken["id"]
                    else:
                        token["head"] = token["lemmas"][0]
                        token["headId"] = token["id"]

        answer["alignAnn"] = self.tokenAnnotator.getAlignmentAnnotation(answer["correctionOrComment"] if("correctionOrComment" in answer) else answer["text"])

    def updateVocab(self, answer, refTok=False, allowMerge=True):
        if(not("tokens" in answer)):
            self.annotateAnswer(answer)
        for token in answer["tokens"]:
            if(not(token["contentWord"])):
                continue
            lemma = token["lemmas"][0]
            headLemma = token["head"]
            if(headLemma in self.question["vocabulary"]):
                headLemma = self.question["vocabulary"][headLemma]["semCat"]

            if(lemma in self.question["vocabulary"]):
                vocab = self.question["vocabulary"][lemma]
                vocab["count"] = vocab["count"] + 1
                semCat = self.question["semCats"][vocab["semCat"]]
                semCat["count"] = semCat["count"] + 1
                if(not(headLemma in semCat["headCats"])):
                    semCat["headCats"][headLemma] = [0,0]
                continue
            newVocab = {"predDist":[0,0], "semCat":lemma, "slPos":token["slPos"] if "slPos" in token else "o", "count": 1, "simScores": []}
            newSemCat = {
                "maxRefSim" : 0,
                "mostSimRefCat" : None,
                "catGroup": [(lemma,1)],
                "predDist": [0,0],
                "headCats": {headLemma:[0,0]},
                "count": 1
            }
            if(refTok): newSemCat["refCat"]= True
            wn = bool(self.semSim.getSynsets(token)[0])
            if(wn):
                newVocab["wn"]= True
                newSemCat["wn"] = True
            emb = not(self.semSim.getWordVector(lemma) is None)
            if(emb):
                newVocab["emb"]= True
                newSemCat["emb"] = True
            isWord = not(re.search('[a-zA-Z]', token["lemmas"][0]) is None)
            if(not(isWord)): newVocab["isWord"] = False

            if(len(self.question["semCats"].keys())==0):
                self.question["semCats"][lemma] = newSemCat
                self.question["vocabulary"][lemma] = newVocab
                self.simFrame[lemma] = pd.Series(np.array(newVocab["simScores"], dtype=np.float), index=self.simFrame.index)
                newVocab["simScores"].append(None)
                self.simFrame.loc[lemma] = np.array(newVocab["simScores"], dtype=np.float)
                continue
            else:
                catMatches = []
                maxSim = 0

                for vocabLemma, vocab in self.question["vocabulary"].items():
                    simScore = None
                    simScore = self.semSim.checkSimpleCase(token["lemmas"][0], vocabLemma)
                    if(simScore is None and "wn" in vocab and "wn" in newVocab and "slPos" in token):
                        simScore = self.semSim.getWordNetSim(token, {"text":vocabLemma,"lemmas":[vocabLemma], "slPos":vocab["slPos"]})
                    if(simScore is None and "emb" in vocab and "emb" in newVocab):
                        simScore = self.semSim.getWordSimilarity(token["lemmas"][0], vocabLemma)
                    newVocab["simScores"].append(simScore)
                    vocab["simScores"].append(simScore)
                    if(not(simScore is None)):
                        semKey = vocab["semCat"]
                        semCat = self.question["semCats"][semKey]
                        if(simScore > FeatureExtractor.SYN_THRES[token["slPos"] if "slPos" in token else "o"] and not("isWord" in newVocab) and not("isWord" in vocab)):
                            if(simScore>maxSim):
                                catMatches = [[semKey,simScore]]
                                maxSim = simScore
                            elif(simScore==maxSim):
                                catMatches.append([semKey,simScore])
                        if("refCat" in semCat and simScore>newSemCat["maxRefSim"]):
                            newSemCat["maxRefSim"] = simScore
                            newSemCat["mostSimRefCat"] = semKey
                self.question["vocabulary"][lemma] = newVocab
                self.simFrame[lemma] = pd.Series(np.array(newVocab["simScores"], dtype=np.float), index=self.simFrame.index)
                newVocab["simScores"].append(None)
                self.simFrame.loc[lemma] = np.array(newVocab["simScores"], dtype=np.float)

                if(len(catMatches) == 0):
                    self.question["semCats"][lemma] = newSemCat
                else:
                    semCatChanges = []
                    newKey = lemma
                    # the new key of a semantic group should be in wn if possible (comes before following conditions). with a simple addition the key should not change, in case of a merge the merging lemma should be key
                    if(not(allowMerge) and len(catMatches)>=1):
                        catMatches = [catMatches[0]]
                    if(len(catMatches)==1 or not("wn" in newSemCat)):
                        for catKey,simScore in catMatches:
                            semCat = self.question["semCats"][catKey]
                            if("wn" in semCat or not(allowMerge)):
                                semCatChanges.append(lemma)
                                newSemCat["wn"] = True
                                if("emb" in semCat):  newSemCat["emb"] = True
                                newKey = catKey
                                newSemCat["catGroup"] = [(lemma,simScore)]
                                newSemCat["mostSimRefCat"] = lemma
                                break
                    self.question["vocabulary"][lemma]["semCat"] = newKey
                    for catKey,simScore in catMatches:
                        semCat = self.question["semCats"][catKey]
                        # merge semantic category groups
                        for entry in semCat["catGroup"]:
                            if(entry[0]==newKey):
                                newSemCat["catGroup"].append((catKey,1))
                            elif(entry[0]==catKey):
                                newSemCat["catGroup"].append((catKey,simScore))
                            else:
                                # TODO get the new similarity scores, maybe even throw some out if new score is to low
                                newSemCat["catGroup"].append(entry)
                            self.question["vocabulary"][entry[0]]["semCat"] = newKey
                        newSemCat["predDist"][0] += semCat["predDist"][0]
                        newSemCat["predDist"][1] += semCat["predDist"][1]
                        newSemCat["count"] += semCat["count"]
                        if("first" in semCat):
                            if(not("first" in newSemCat)):
                                newSemCat["first"] = semCat["first"]
                            else:
                                newSemCat["first"] = min(semCat["first"], newSemCat["first"])
                        if(semCat["maxRefSim"] > newSemCat["maxRefSim"]):
                            newSemCat["maxRefSim"] = semCat["maxRefSim"]
                            newSemCat["mostSimRefCat"] = semCat["mostSimRefCat"]
                        if("refCat" in semCat):
                            newSemCat["refCat"] = True
                        # merge the head distributions
                        for key,val in semCat["headCats"].items():
                            if(key in newSemCat["headCats"]):
                                newSemCat["headCats"][key][0] += val[0]
                                newSemCat["headCats"][key][1] += val[1]
                            else:
                                newSemCat["headCats"][key] = val
                        if(catKey!=newKey):
                            del self.question["semCats"][catKey]
                            semCatChanges.append(catKey)
                    self.question["semCats"][newKey] = newSemCat

                    # when the semantic category changes all head occurances need to change as well
                    for semCat,entry in self.question["semCats"].items():
                        for semCatChange in semCatChanges:
                            if(semCatChange in entry["headCats"]):
                                if(newKey in entry["headCats"]):
                                    entry["headCats"][newKey][0] += entry["headCats"][semCatChange][0]
                                    entry["headCats"][newKey][1] += entry["headCats"][semCatChange][1]
                                else:
                                    entry["headCats"][newKey] = entry["headCats"][semCatChange]
                                del entry["headCats"][semCatChange]

    def updateVocabPredDist(self, answer, pred, useAnn=True, threshold=0.8, updateProcedure=1):
        if(not(useAnn)):
            binLabel = self.getBinaryAnswerPred(pred, threshold=threshold, updateProcedure=updateProcedure)
        else:
            binLabel = pred == "correct"#FeatureExtractor.LABEL_DICT[pred] <= 1
        pred = int(binLabel)
        self.question["classCounts"][pred] += 1
        lemmasCounted = []
        for token in answer["tokens"]:
            if(not(token["contentWord"])):
                continue
            lemma = token["lemmas"][0]
            if(lemma in lemmasCounted):
                continue
            lemmasCounted.append(lemma)
            vocab = self.question["vocabulary"][lemma]
            if(not("first" in vocab)):
                vocab["first"] = sum(self.question["classCounts"])
            semKey = vocab["semCat"]
            vocab["predDist"][pred] += 1

            semCat = self.question["semCats"][semKey]
            if(not("first" in semCat)):
                semCat["first"] = sum(self.question["classCounts"])
            semCat["predDist"][pred] += 1

            headLemma = token["head"]
            if(headLemma in self.question["vocabulary"]):
                groupHeadLemma = self.question["vocabulary"][headLemma]["semCat"]
            self.question["semCats"][semKey]["headCats"][groupHeadLemma][pred] += 1

    def getBinaryAnswerPred(self, pred, threshold=0.8, updateProcedure=1):
        # updateProcedure
        # 0: get correct prediction if not suffiently well predicted
        # 1: always get correct prediction
        # 2: update only if sufficiently well predicted
        # 3: always use the given prediction
        if(updateProcedure<3):
            sufficientPred = abs(pred-0.5) > (0.5 - (1-threshold))
            if(not(sufficientPred) and updateProcedure==2):
                return
            if(not(sufficientPred) or updateProcedure==1):
                pred = self.getCorrectPred(answer) # from teacher
                if(pred is None):
                    return
        binLabel = pred >= 0.5
        return binLabel
