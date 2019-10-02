import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr
import re, os, sys, json, csv, math, codecs
from os.path import join, dirname, realpath, isdir, isfile, exists
from FeatureExtractor import FeatureExtractor
import xml.etree.ElementTree as ET

FILE_PATH = dirname(realpath(__file__))
SEMVAL_PATH = join(FILE_PATH, "..", "question-corpora","SEMVAL")
TEST_PATH = join(SEMVAL_PATH,"test")
TRAIN_PATH = join(SEMVAL_PATH,"training")
SEMVAL_PATHES = {
    "T": [join(TRAIN_PATH, "beetle"),join(TRAIN_PATH, "sciEntsBank")],
    "UA": [join(TEST_PATH,"beetle","test-unseen-answers"), join(TEST_PATH,"sciEntsBank","test-unseen-answers")],
    "UQ": [join(TEST_PATH,"beetle","test-unseen-questions"), join(TEST_PATH,"sciEntsBank","test-unseen-questions")],
    "UD": [join(TEST_PATH,"sciEntsBank","test-unseen-domains")],
}

def generateSemvalFeatureFiles(featureFile, annFile, testSet=False, trainVocab=False):
    if(isFile(annFile)):
        with open(annFile, "r") as f:
            questionData = json.load(f)
    else:
        questionData = getQuestionData(testSet if testSet else "T")

    if(testSet=="UA"):
        with open(join(TRAIN_PATH, trainVocab), "r") as f:
            trainedVocab = json.load(f)
        for qId, question in questionData.items():
            questionVocab = trainedVocab[question["id"]]
            questionVocab["studentAnswers"] = question["studentAnswers"]
            question = questionVocab

    featureExtrator = FeatureExtractor(simMeasure="fasttext", lang="en")
    for qIdx, qId in enumerate(questionData.keys()):
        question = questionData[qId]
        print(qIdx, qId)
        featureExtrator.setQuestion(question, resetVocabulary=not(testSet))
        for aIdx, answer in enumerate(question["studentAnswers"]):
            featureExtrator.updateVocab(answer, allowMerge=not(testSet))
            if(not(testSet)): featureExtrator.updateVocabPredDist(answer, answer["answerCategory"])

        for aIdx, answer in enumerate(question["studentAnswers"]):
            featureExtrator.extractFeatures(answer)
            if(not(testSet=="UQ")):
                featureExtrator.addPredDistBasedFeatures(answer, vocab=True)
            if(not("sulAlign" in answer["features"])):
                if("sulAlign" in answer):
                    answer["features"]["sulAlign"] = answer["sulAlign"]
                else:
                    featureExtrator.computeSulAlign(answer)

    saveFeatureJSON({"questions":[val for key,val in questionData.items()]}, featureFile )
    if(not(testSet)): saveTrainingVocab(trainVocab, questionData)
    if(!isFile(annFile)): saveTokenAnnotation(annFile, questionData)

def getQuestionData(dataSet):
    dataPathes = SEMVAL_PATHES[dataSet]
    questionFiles = []
    for path in dataPathes:
        for root, _, files in os.walk(path):
            for file in files:
                if '.xml' == file[-4:]:
                    questionFiles.append(join(root, file))
    questionsAnnotation = {}
    for qIdx, file in enumerate(questionFiles):
        question = parseSemval(file)
        questionsAnnotation[question["id"]] = question
    return questionsAnnotation

def parseSemval(xmlPath):
    tree = ET.parse(xmlPath)
    root = tree.getroot()
    # go through all pages
    questionDict = root.attrib
    for root_obj in root:
        if(root_obj.tag=="questionText"):
            questionDict["text"] = root_obj.text
        if(root_obj.tag=="referenceAnswers"):
            questionDict["referenceAnswers"] = []
            for refAns in root_obj:
                questionDict["referenceAnswers"].append({"id": refAns.attrib["id"], "text":refAns.text})
        if(root_obj.tag=="studentAnswers"):
            questionDict["studentAnswers"] = []
            for pupAns in root_obj:
                questionDict["studentAnswers"].append({"id": pupAns.attrib["id"], "text":pupAns.text, "answerCategory": pupAns.attrib["accuracy"] if pupAns.attrib["accuracy"]=="correct" else "none"})
    questionDict["referenceAnswer"] = questionDict["referenceAnswers"][0]
    return  questionDict

def saveTokenAnnotation(annFile, questionData):
    for qId in questionData.keys():
        question = questionData[qId]
        del question["semCats"]
        del question["vocabulary"]
        del question["classCounts"]
        refAns = question["referenceAnswer"]
        del refAns["alignAnn"]
        studentAnswers = question["studentAnswers"]
        for studentAnswer in studentAnswers:
            del studentAnswer["alignAnn"]
            studentAnswer["sulAlign"] = studentAnswer["features"]["sulAlign"]
            del studentAnswer["features"]
    with open(annFile, "w+") as f:
        json.dump(questionData, f, indent=4)

def saveTrainingVocab(trainFile, questionData):
    questionData = JSON.parse(JSON.stringify(questionData))
    for key,val in questionData.items():
        del val["studentAnswers"]
        del val["referenceAnswers"]
    with open(join(TRAIN_PATH, trainFile), "w+") as f:
        json.dump(questionData, f, indent=4)

def saveFeatureJSON(questionsAnnotation, featurePath):
    featureData = {}
    for question in questionsAnnotation["questions"]:
        featureData[question["id"]] = {}
        for aIdx, answer in enumerate(question["studentAnswers"]):
            featureData[question["id"]][answer["id"]] = {"label": answer["answerCategory"] == "correct", "features": answer["features"]}
    with open(featurePath, "w+") as f:
        json.dump(featureData, f, indent=4)

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

def saveFeatureCSV(questionsAnnotation, featurePath):
    _, fields = generateCSVMapping(questionsAnnotation["questions"][0]["studentAnswers"][0],questionsAnnotation["questions"][0]["id"])
    with open(featurePath, 'w') as csvfile:
        featureWriter = csv.writer(csvfile)
        featureWriter.writerow(fields)
        for question in questionsAnnotation["questions"]:
            for aIdx, answer in enumerate(question["studentAnswers"]):
                values, _ = generateCSVMapping(answer, question["id"])
                featureWriter.writerow(values)

############################ OLD ##########################################

def analyseVocabDiff():

    with open(join(TRAIN_PATH, "trainingVocabEmb.json"), "r") as f:
        trainVocab = json.load(f)

    with open(join(TRAIN_PATH, "testVocabEmb.json"), "r") as f:
        testVocab = json.load(f)

    with open(join(TRAIN_PATH, "trainingFeaturesEmb.csv"), "r") as f:
        trainFeatures = pd.read_csv(f)
    trainFeatures.insert(3,'bin_label',trainFeatures["label"]<=1)

    with open(join(TRAIN_PATH, "testVFeaturesEmb.csv"), "r") as f:
        testFeatures = pd.read_csv(f)
    testFeatures.insert(3,'bin_label',testFeatures["label"]<=1)

    for qId in trainVocab.keys():

        qTrFeat = trainFeatures.loc[trainFeatures["qID"]==qId]
        trDWCCor = pearsonr(qTrFeat["distWeightContent"],qTrFeat["bin_label"])[0]
        print(qTrFeat["distWeightContent"])
        trDWhCCor = pearsonr(qTrFeat["distWeightHeadContent"],qTrFeat["bin_label"])[0]
        qTeFeat = testFeatures.loc[testFeatures["qID"]==qId]
        teDWCCor = pearsonr(qTeFeat["distWeightContent"],qTeFeat["bin_label"])[0]
        print(qTeFeat["distWeightContent"])
        teDWhCCor = pearsonr(qTeFeat["distWeightHeadContent"],qTeFeat["bin_label"])[0]

        trainQ = trainVocab[qId]
        testQ = testVocab[qId]
        trV = trainQ["vocabulary"]
        teV = testQ["vocabulary"]
        allVocab = list(set(list(trV.keys()) + list(teV.keys())))
        trC = sum(trainQ["classCounts"])
        trNC, trPC = trainQ["classCounts"]
        teC = sum(testQ["classCounts"])
        teNC, tePC = testQ["classCounts"]
        print(trainQ["text"])
        print(trainQ["referenceAnswer"]["text"])
        print("trainSize: {} neg: {} pos: {} distR: {} headR: {}".format(trC, trNC, trPC, trDWCCor, trDWhCCor))
        print("testSize: {} neg: {} pos: {} distR: {} headR: {}".format(teC, teNC, tePC, teDWCCor, teDWhCCor))
        stats = [[voc,
        sum(trV[voc]["predDist"]) if(voc in trV) else None,
        sum(trV[voc]["predDist"])/trC if(voc in trV) else None,
        trV[voc]["predDist"][0]/trNC if(voc in trV) else None,
        trV[voc]["predDist"][1]/trPC if(voc in trV) else None,
        sum(teV[voc]["predDist"]) if(voc in teV) else None,
        sum(teV[voc]["predDist"])/teC if(voc in teV) else None,
        teV[voc]["predDist"][0]/teNC if(voc in teV) else None,
        teV[voc]["predDist"][1]/tePC if(voc in teV) else None] for voc in allVocab]
        statsFrame = pd.DataFrame(stats, columns = ["voc", "tr all", "tr all %", "tr neg", "tr pos", "te all","te all %", "te neg", "te pos"])
        statsFrame = statsFrame.set_index("voc")
        statsFrame.insert(0, "total", statsFrame[["tr all", "te all"]].fillna(0).sum(axis=1))
        statsFrame["diff all"] = statsFrame["tr all %"].fillna(0) - statsFrame["te all %"].fillna(0)
        statsFrame["diff neg"] = statsFrame["tr neg"].fillna(0)  - statsFrame["te neg"].fillna(0)
        statsFrame["diff pos"] = statsFrame["tr pos"].fillna(0)  - statsFrame["te pos"].fillna(0)
        statsFrame = statsFrame.loc[statsFrame["total"]>=3].sort_values('total', ascending=False)
        print(statsFrame)
        input()

def setRefWordCounts(originalAnn, processedAnn):
    for qIdx,question in enumerate(processedAnn["questions"]):
        refAnsAnn = question["referenceAnswer"]
        refAnsTar = originalAnn["questions"][qIdx]["referenceAnswer"]
        if(not("vocabulary" in question) or not("tokens" in refAnsAnn)):
            continue
        lemmas = [token["lemmas"][0] for token in refAnsAnn["tokens"] if(token["contentWord"])]
        refAnsTar["compWeights"] = {}
        totalCount = len(question["studentAnswers"]) + 1
        for lemma in lemmas:
            if(not(lemma in  question["vocabulary"])):
                refAnsTar["compWeights"][lemma] = 0
                continue
            lemmaDist = question["vocabulary"][lemma]
            refAnsTar["compWeights"][lemma] = lemmaDist

if __name__ == "__main__":
    generateSemvalFeatureFiles("trainingFeaturesWithVocab.json", join(TRAIN_PATH, "rawTrainingTokenAnnotation.json"), testSet=False, "trainingVocabWithVocab.json")
    for testSet in ["UA","UQ", "UD"]:
        featurePath = join(TEST_PATH, "test" + testSet + "FeaturesWithVocab.json")
        annPath = join(TEST_PATH, "rawTest" + testSet + "TokenAnnotation.json")
        generateSemvalFeatureFiles(featurePath, annPath, testSet, "trainingVocabWithVocab.json")
