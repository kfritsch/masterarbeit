import numpy as np
import re, os, sys, json, csv, math
import pandas as pd
from os.path import join, dirname, realpath
FILE_PATH = dirname(realpath(__file__))
RESULTS_PATH = join(os.path.dirname(os.path.realpath(__file__)), "results")
from TokenAnnotator import TokenAnnotator
from sklearn.feature_extraction.text import CountVectorizer

CATEGORIES = {}
LABEL_DICT = {
    "correct": 0,
    "binary_correct": 1,
    "partially_correct": 2,
    "missconception": 3,
    "concept_mix-up": 4,
    "guessing": 5,
    "none": 6
}

def dataToHtmlTable(data, columnLabels, rowLabels):
    df = pd.DataFrame(data, index=rowLabels, columns=columnLabels)
    html = df.to_html()
    return html

class ASAGClassifier(object):
    def __init__(self, stopwordsFile="stopwords.txt"):
        self.tokenAnnotator = TokenAnnotator()
        with open(stopwordsFile, "r") as txtFile:
            self.stopwords = [line.strip() for line in txtFile]
        self.question = None

    def filterLemmas(self, lemmas):
        return [word for word in lemmas if(len(word)>1 and not(word in self.filterWords) or word in self.question["stopwordExceptions"])]

    def setupQuestion(self, question, resetVocabulary=True):
        self.question = question
        if(not("lemmas" in self.question)):
            questionTokens = self.tokenAnnotator.annotateText(self.question["text"])
            self.question["lemmas"] = self.getLemmasFromTokens(questionTokens)
        if(not("stopwordExceptions" in self.question)):
            self.question["stopwordExceptions"] = []
        if(not("questionStopwords" in self.question)):
            self.question["questionStopwords"] = []
        self.filterWords = self.stopwords + self.question["lemmas"] + self.question["questionStopwords"]
        if(resetVocabulary): self.resetVocabulary()

    def resetVocabulary(self):
        self.question["vocabulary"] = {}
        if(not("tokens" in self.question["referenceAnswer"])):
            self.annotateAnswer(self.question["referenceAnswer"])
        answerLemmas = self.getLemmasFromTokens(self.question["referenceAnswer"]["tokens"])
        for lemma in answerLemmas:
            if(not(lemma in self.question["vocabulary"])):
                self.question["vocabulary"][lemma] = [0,1]
        self.question["classCounts"] = [0,1]

    def getLemmasFromTokens(self, tokens):
        return list(set([lemma for token in tokens for lemma in token["lemmas"]]))

    def getLemmasFromAnswer(self, answer):
        if(not("tokens" in answer)): self.annotateAnswer(answer)
        lemmas = self.getLemmasFromTokens(answer["tokens"])
        answer["lemmas"] = self.filterLemmas(lemmas)
        return answer["lemmas"]

    def annotateAnswer(self, answer):
        answer["tokens"] =  self.tokenAnnotator.annotateText(answer["correctionOrComment"] if("correctionOrComment" in answer) else answer["text"])

    def getProb(self, entry, denomAddend=10, denomFac=1.5):
        [negCount,posCount] = self.question["classCounts"]
        [negOcc,posOcc] = entry
        total = posCount + negCount
        posPerc = self.newTransProb(posOcc,posCount) if(posOcc != 0) else self.newZeroSmoothProb(posCount, total)
        negPerc = self.newTransProb(negOcc,negCount) if(negCount != 0 and negOcc != 0) else self.newZeroSmoothProb(negCount, total)
        return posPerc/(posPerc + negPerc)

    def transProb(self, occCount, classCount, denomAddend=10):
        return (occCount*occCount/(classCount+denomAddend))

    def zeroSmoothProb(self, classCount, totalCount ,denomAddend=10, denomFac=1.5):
        return (1/(denomFac*(totalCount+classCount+denomAddend)))

    def newTransProb(self, occCount, classCount, denomAddend=10):
        return ((occCount/(classCount+denomAddend)) + (1 - (1/occCount)))/2

    def newZeroSmoothProb(self, classCount, totalCount, denomAddend=10, denomFac=2.5):
        return (1/(denomFac*(math.sqrt(classCount)+denomAddend)))

    def classifyAnswer(self, answer):
        lemmas = self.getLemmasFromAnswer(answer)
        posVals =  np.array([self.getProb(self.question["vocabulary"][lemma]) if lemma in self.question["vocabulary"] else 0.5 for lemma in lemmas])
        negVals = 1 - posVals
        posPerc = np.prod(posVals)
        negPerc = np.prod(negVals)
        answer["pred"] = posPerc/(posPerc + negPerc)
        print(list(zip(posVals,lemmas)), answer["pred"])
        return answer["pred"]

    def getCorrectPred(self, answer):
        if(LABEL_DICT[answer["answerCategory"]]<=1):
            return 0.99
        elif(LABEL_DICT[answer["answerCategory"]]>=3):
            return 0.01
        else:
            return None

    def updateVocabulary(self, answer, threshold=0.8, updateProcedure=1):
        # updateProcedure
        # 0: get correct prediction if not suffiently well predicted
        # 1: always get correct prediction
        # 2: update only if sufficiently well predicted
        # 3: always use the given prediction
        pred = answer["pred"] if("pred" in answer) else 0.5
        if(updateProcedure<3):
            sufficientPred = abs(pred-0.5) > (0.5 - (1-threshold))
            if(not(sufficientPred) and updateProcedure==2):
                return
            if(not(sufficientPred) or updateProcedure==1):
                pred = self.getCorrectPred(answer)
                if(pred is None):
                    return
        pred = pred >= 0.5
        self.question["classCounts"][int(pred)] += 1
        for lemma in answer["lemmas"]:
            if(not(lemma in self.question["vocabulary"])):
                self.question["vocabulary"][lemma] = [int(not(pred)),int(pred)]
            else:
                self.question["vocabulary"][lemma][int(pred)] += 1

    def classifyBatch(self, question):
        pupilAnswerLemmas = [self.getLemmasFromAnswer(answer) for answer in question["answersAnnotation"]]
        # take answer with highest information gain first (shares most words with others), update vocabulary, (recompute -> even cheap
        self.vectorizer = CountVectorizer()
        self.vectorizer.fit([" ".join(question["vocabulary"].keys())])
        M = self.vectorizer.transform([" ".join(p) for p in pupilAnswerLemmas]).toarray()
        M = (M>0).astype(int)
        return np.sum(M, axis=1)


def confStats(preds):
    return [len(preds), np.sum(preds), len(preds)-np.sum(preds)]

def printResults(M, labels, filename="q1_results.html"):
    rowLabels = ["correct", "binary_correct", "partially_correct", "missconception", "concept_mix-up", "guessing", "none", "", "false", ">= part_correct", ">= bin_correct", "total"]
    columnLabels = ["label #", "pos", "neg"]
    with open(join(RESULTS_PATH,filename), 'w') as f:
        f.write("<!DOCTYPE html>\n<html>\n<head>\n<title>{}</title>\n</head>\n<body>\n")
        for i in range(1,5):
            preds = M>i
            questionRes = [confStats(preds[labels == i]) for i in range(7)]
            questionRes.append([""] * len(questionRes[0]))
            questionRes.append(confStats(preds[labels >= 3]))
            questionRes.append(confStats(preds[labels <= 2]))
            questionRes.append(confStats(preds[labels <= 1]))
            questionRes.append(confStats(preds))
            html = listsToHtmlTable(questionRes, columnLabels, rowLabels)
            f.write("<h2>Threshold {}</h2>\n".format(i))
            f.write(html)
        f.write("</body>\n</html>\n")

def printPreds(preds, labels, question):
    print(question["text"])
    for i,label in enumerate(["correct", "binary_correct", "partially_correct", "missconception", "concept_mix-up", "guessing", "none"]):
        labelPreds = preds[labels==i]
        labelPreds.sort()
        print("{},{}".format(label, labelPreds))

def printPredTable(preds, labels, question):
    print(question["text"])
    predTules = [(labels[i], list(preds[i,:])) for i in range(len(labels))]
    sortedPredTuples = sorted(predTules, key=lambda k: (k[0], -k[1][0]))
    for tup in sortedPredTuples:
        print(tup)
    vocabs = sorted(question["vocabulary"].items(), key=lambda k: max(k[1][0],k[1][1]), reverse=True)
    for word,counts in vocabs:
        print(word, counts)

ProcedureList = ["GET CORRECT IF NOT SUFFICIENT...", "GET CORRECT...","UPADATE IF SUFFICIENT...", "ALWAYS USE PRED..."]

def classifyQuestionData(dataPath=join(FILE_PATH, "GoldStandards.json"), save=False):
    clf = ASAGClassifier()
    with open(dataPath, "r") as f:
        questionsAnnotation = json.load(f)
    for question in questionsAnnotation["questions"]:
        if(not(clf.question) or clf.question["id"] != question["id"]):
            clf.setupQuestion(question)
        labels = np.array([LABEL_DICT[answer["answerCategory"]] for answer in question["answersAnnotation"]])
        predTable = np.zeros((len(labels),6))
        print("NO VOCAB UPDATE...")
        predTable[:,0] = np.array([clf.classifyAnswer(answer) for answer in question["answersAnnotation"]])
        for i in [0,2,3,1]:
            print("\n\n")
            print(ProcedureList[i])
            preds = []
            clf.resetVocabulary()
            for answer in question["answersAnnotation"]:
                preds.append(clf.classifyAnswer(answer))
                clf.updateVocabulary(answer, threshold=0.8, updateProcedure=i)
            predTable[:,i+1] = preds
        print("USE PERFEKT VOCAB...")
        predTable[:,5] = np.array([clf.classifyAnswer(answer) for answer in question["answersAnnotation"]])
        #printPreds(preds, labels, question)
        printPredTable(predTable, labels, question)
        with open("AnnotatedGoldStandards.json", "w+") as f:
            json.dump(questionsAnnotation, f, indent=4)
        input()
    if(save):
        with open("AnnotatedGoldStandards.json", "w+") as f:
            json.dump(questionsAnnotation, f, indent=4)

if __name__ == "__main__":
    classifyQuestionData(dataPath=join(FILE_PATH, "AnnotatedGoldStandards.json"), save=True)
