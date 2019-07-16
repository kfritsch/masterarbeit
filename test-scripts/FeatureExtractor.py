from TokenAnnotator import TokenAnnotator
from SemSim import SemSim
from sultanAlign.align import suAlignmentScore
import sys, math
import numpy as np
import pandas as pd

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
    SYN_THRES = {
        "a":0.7,
        "n":0.6,
        "v":0.6,
        "o":0.7
    }

    def __init__(self, simMeasure="small", stopwordsFile="stopwords.txt"):
        self.tokenAnnotator = TokenAnnotator()
        with open(stopwordsFile, "r") as txtFile:
            self.stopwords = [line.strip() for line in txtFile]
        self.filterWords = [w for w in self.stopwords]
        self.semSim = SemSim(simMeasure)

    # def setSimMeasure(self, simMeasure):
    #     self.semSim = SemSim(simMeasure)
    #     if(self.question):
    #         self.semSim.setSimDF(self.question)

    def setQuestion(self, question, resetVocabulary=True):
        self.question = question
        if(not("stopwordExceptions" in self.question)):
            self.question["stopwordExceptions"] = []
        if(not("questionStopwords" in self.question)):
            self.question["questionStopwords"] = []
        if(not("contentLemmas" in self.question)):
            questionTokens = self.tokenAnnotator.annotateText(self.question["text"])
            self.question["contentLemmas"] = [token["lemmas"][0] for token in questionTokens if(self.isContentWord(token["lemmas"][0]))]
            self.question["alignAnn"] = self.tokenAnnotator.getAlignmentAnnotation(self.question["text"])
        self.filterWords = [w for w in self.stopwords] + self.question["contentLemmas"] + self.question["questionStopwords"]
        if(not("tokens" in self.question["referenceAnswer"])):
            self.annotateAnswer(self.question["referenceAnswer"])
        if("assignedWeights" in self.question["referenceAnswer"]):
            self.setAnnWeights(self.question["referenceAnswer"])
        self.distWeights = None
        # self.semSim.setSimDF(self.question)
        if(not("vocabulary") in self.question or resetVocabulary): self.resetVocabulary()

    def setAnnWeights(self, refAns):
        meanCWW = np.mean([val for key,val in refAns["assignedWeights"].items()])
        self.annWeights = {}
        for key,val in refAns["assignedWeights"].items():
            self.annWeights[key] = (val-meanCWW)+1

    def setDistWeights(self):
        self.distWeights = dict([(key, [(val["predDist"][1]-val["predDist"][0])/sum(val["predDist"]),sum(val["predDist"])/len(self.question["answersAnnotation"])]) for key,val in self.question["vocabulary"].items()])

    def resetVocabulary(self):
        self.question["vocabulary"] = {}
        self.question["classCounts"] = [0,0]
        self.updateVocab(self.question["referenceAnswer"])

    def extractFeatures(self, pupAns):
        if("features" in pupAns):
            return pupAns["features"]

        if(not("tokens" in pupAns)):
            self.annotateAnswer(pupAns)
        pupWords, pupWordsSet, pupTokenSet, pupContentToken, pupContentLemmas = [], [], [], [], []
        for token in pupAns["tokens"]:
            pupWords.append(token["text"])
            if(not(token["text"] in pupWordsSet)):
                pupWordsSet.append(token["text"])
                pupTokenSet.append(token)
            if(token["contentWord"] and not(token["lemmas"][0] in pupContentLemmas)):
                pupContentLemmas.append(token["lemmas"][0])
                pupContentToken.append(token)

        refAns = self.question["referenceAnswer"]
        refWords, refWordsSet, refTokenSet, refContentToken, refContentPos, refContentLemmas = [], [], [], [], [], []
        for token in refAns["tokens"]:
            refWords.append(token["text"])
            if(not(token["text"] in refWordsSet)):
                refWordsSet.append(token["text"])
                refTokenSet.append(token)
            if(token["contentWord"] and not(token["lemmas"][0] in refContentLemmas)):
                refContentLemmas.append(token["lemmas"][0] )
                refContentToken.append(token)
                refContentPos.append(token["udPos"])
        self.refContentToken = refContentToken
        self.pupContentToken = pupContentToken

        if(len(pupContentToken) == 0):
            features = {
                "qType": FeatureExtractor.QUESTION_TYPES[self.question["type"]],
                "pupAbsLen": len(pupAns["tokens"]),
                "refAbsLen": len(refAns["tokens"]),
                "pupContentLen": len(pupContentToken),
                "refContentLen": len(refContentToken)
            }
        else:
            # TODO: consider embedding difference rank
            allSimDfWords = self.semSim.getTokenSimilarityMatrix(pupTokenSet, refTokenSet, asDf=True, lemmaIdc=False)
            self.contentSimDf = self.semSim.getTokenSimilarityMatrix(pupContentToken, refContentToken, asDf=True, lemmaIdc=True)
            similarWordsRecall = self.getMaxWordsSim(self.contentSimDf,self.refContentToken, axis=0)
            similarWordsPrecision = self.getMaxWordsSim(self.contentSimDf,self.pupContentToken,  axis=1)

            meanCWW = np.mean([val for key,val in refAns["assignedWeights"].items()])
            scaledAnnWeights = {}
            for key,val in refAns["assignedWeights"].items():
                scaledAnnWeights[key] = (val-meanCWW)+1

            features = {
                "qType": FeatureExtractor.QUESTION_TYPES[self.question["type"]],
                "pupAbsLen": len(pupAns["tokens"]),
                "refAbsLen": len(refAns["tokens"]),
                "pupContentLen": len(pupContentToken),
                "refContentLen": len(refContentToken),
                "lemmaRecall": sum([1 if lemma in refContentLemmas else 0 for lemma in pupContentLemmas]),
                "pupPosDist": self.getPosDist(pupAns["tokens"]),
                "refPosDist": self.getPosDist(refAns["tokens"]),
                "contentRec": self.getContentScore(similarWordsRecall, refIdx=0, weights=False),
                "annWContentRec": self.getContentScore(similarWordsRecall, refIdx=0, weights=scaledAnnWeights),
                "contentPrec": self.getContentScore(similarWordsPrecision, refIdx=2, weights=False),
                "annWContentPrec": self.getContentScore(similarWordsPrecision, refIdx=2, weights=scaledAnnWeights),
                # "pupEmb": self.semSim.getSetenceEmbedding(pupWords),
                # "refEmb": self.semSim.getSetenceEmbedding(refWords),
                "sulAlign": suAlignmentScore(refAns, pupAns, self.question, allSimDfWords)[0:2],
                # "annWSulAlign":
                "posSimHist": self.getPosTagSimHist(similarWordsRecall, refContentPos, bins=6)
                # "lemWtSimHist":
        }
        pupAns["features"] = features
        return features

    def addPredDistBasedFeatures(self, answer):
        if(not("features" in answer)):
            self.extractFeatures(answer)
        if(not(self.distWeights)):
            self.setDistWeights()
        answer["features"]["distWheitContent"] = np.sum([self.distWeights[token["lemmas"][0]] for token in answer["tokens"] if token["lemmas"][0] in self.distWeights])/len(answer["tokens"])
        answer["features"]["posDistWeightHist"] = self.getPosDistWeightHist(answer)

    def getPosDistWeightHist(self, answer, bins=6):
        posDistWeightHist = {"a":[0]*bins, "n":[0]*bins, "v":[0]*bins, "o":[0]*bins}
        for token in answer["tokens"]:
            if(not(token["lemmas"][0] in self.distWeights)):
                continue
            tokenDist,tokenWeight = self.distWeights[token["lemmas"][0]]
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
            return np.sum(np.array([(simTup[1]>FeatureExtractor.SYN_THRES[simTup[3]]) * self.annWeights[simTup[refIdx]] for simTup in simWordsTuples if simTup[refIdx]]))/len(simWordsTuples)
        else:
            return np.sum(np.array([simTup[1]>FeatureExtractor.SYN_THRES[simTup[3]] for simTup in simWordsTuples]))/len(simWordsTuples)

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

    def getMaxWordsSim(self, dataFrame, sourceTokens, axis=0):
        maxIdc = dataFrame.idxmax(axis=axis)
        sourceWords = dataFrame.columns if(axis==0) else dataFrame.index
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
        return ((word.isdigit() or len(word)>1) and not(word in self.filterWords)) or word in self.question["stopwordExceptions"]

    def annotateAnswer(self, answer):
        answer["tokens"] =  self.tokenAnnotator.annotateText(answer["correctionOrComment"] if("correctionOrComment" in answer) else answer["text"])
        for token in answer["tokens"]:
            token["contentWord"] = self.isContentWord(token["lemmas"][0])
        answer["alignAnn"] = self.tokenAnnotator.getAlignmentAnnotation(answer["correctionOrComment"] if("correctionOrComment" in answer) else answer["text"])

    def updateVocab(self, answer, useAnn=True, synThres=0.5):
        if(not("features" in answer)):
            self.extractFeatures(answer)
        if(not(useAnn) and "pred" in answer):
            binLabel = self.getBinaryAnswerPred(answer, threshold=0.8, updateProcedure=1)
        else:
            binLabel = FeatureExtractor.LABEL_DICT[answer["answerCategory"]] <= 1
        lemmas = [token["lemmas"][0] for token in answer["tokens"] if(token["contentWord"])]
        for lemma in lemmas:
            if(not(lemma in self.question["vocabulary"])):
                self.question["vocabulary"][lemma] = {
                    "predDist": [0,0],
                    "synonyms": {}
                }
            self.question["vocabulary"][lemma]["predDist"][int(binLabel)] += 1
        for refToken in self.refContentToken:
            refLemma = refToken["lemmas"][0]
            synThres = FeatureExtractor.SYN_THRES[refToken["slPos"] if "slPos" in refToken else "o"]
            synWords = self.contentSimDf[self.contentSimDf[refLemma]>synThres].index
            synToken = [token for token in self.pupContentToken if(token["text"] in synWords and not(token["lemmas"][0]==refLemma or token["lemmas"][0] in self.question["vocabulary"][refLemma]["synonyms"]))]
            for synTok in synToken:
                synLemma = synTok["lemmas"][0]
                self.question["vocabulary"][refLemma]["synonyms"][synLemma] = self.contentSimDf.loc[synLemma,refLemma]



    def getBinaryAnswerPred(self, answer, threshold=0.8, updateProcedure=1):
        # updateProcedure
        # 0: get correct prediction if not suffiently well predicted
        # 1: always get correct prediction
        # 2: update only if sufficiently well predicted
        # 3: always use the given prediction
        pred = answer["pred"]
        if(updateProcedure<3):
            sufficientPred = abs(pred-0.5) > (0.5 - (1-threshold))
            if(not(sufficientPred) and updateProcedure==2):
                return
            if(not(sufficientPred) or updateProcedure==1):
                pred = self.getCorrectPred(answer)
                if(pred is None):
                    return
        binLabel = pred >= 0.5
        return binLabel
