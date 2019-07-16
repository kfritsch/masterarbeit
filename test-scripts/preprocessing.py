import numpy as np
import re, os, sys, json, csv, math
from os.path import join, dirname, realpath
FILE_PATH = dirname(realpath(__file__))
from FeatureExtractor import FeatureExtractor

def preprocessQuestionData(questionDataPath, featurePath):
    with open(questionDataPath, "r") as f:
        questionsAnnotation = json.load(f)
    featureExtrator = FeatureExtractor(simMeasure="fasttext", stopwordsFile="stopwords.txt")
    for question in questionsAnnotation["questions"]:
        print(question["title"])
        featureExtrator.setQuestion(question)
        for answer in question["answersAnnotation"]:
            featureExtrator.extractFeatures(answer)
            featureExtrator.updateVocab(answer)
        for answer in question["answersAnnotation"]:
            featureExtrator.addPredDistBasedFeatures(answer)
        saveFeatureCSV(question, featurePath)
    with open("AnnotatedGoldStandards.json", "w+") as f:
        json.dump(questionsAnnotation, f, indent=4)

def generateCSVMapping(answer, questionId):
    fields = ["qID", "aID", "label"]
    values = [questionId,answer["id"], FeatureExtractor.LABEL_DICT[answer["answerCategory"]]]
    for key,val in answer["features"].items():
        if(isinstance(val,dict)):
            for key2,val2 in val.items():
                if(isinstance(val2, list) or isinstance(val2, tuple)):
                    for i in range(len(val2)):
                        fields.append(key+ "_" +key2+ "_"+str(i))
                        values.append(val2[i])
                else:
                    fields.append(key+ "_" +key2)
                    values.append(val2)
        elif(isinstance(val, list) or isinstance(val, tuple)):
            for i in range(len(val)):
                fields.append(key+ "_" +str(i))
                values.append(val[i])
        else:
            fields.append(key)
            values.append(val)
    return values, fields

def saveFeatureCSV(question, featurePath):
    _, fields = generateCSVMapping(question["answersAnnotation"][0],question["id"])
    with open(featurePath, 'w') as csvfile:
        featureWriter = csv.writer(csvfile)
        featureWriter.writerow(fields)
        for answer in question["answersAnnotation"]:
            values, _ = generateCSVMapping(answer, question["id"])
            featureWriter.writerow(values)


def setRefWordCounts(originalAnn, processedAnn):
    for qIdx,question in enumerate(processedAnn["questions"]):
        refAnsAnn = question["referenceAnswer"]
        refAnsTar = originalAnn["questions"][qIdx]["referenceAnswer"]
        if(not("vocabulary" in question) or not("tokens" in refAnsAnn)):
            continue
        lemmas = [token["lemmas"][0] for token in refAnsAnn["tokens"] if(token["contentWord"])]
        refAnsTar["compWeights"] = {}
        totalCount = len(question["answersAnnotation"]) + 1
        for lemma in lemmas:
            if(not(lemma in  question["vocabulary"])):
                refAnsTar["compWeights"][lemma] = 0
                continue
            lemmaDist = question["vocabulary"][lemma]
            refAnsTar["compWeights"][lemma] = lemmaDist


preprocessQuestionData("GoldStandards.json", "features.csv")
