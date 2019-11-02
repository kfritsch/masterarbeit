import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr
import re, os, sys, json, csv, math, codecs
from os.path import join, dirname, realpath, isdir, isfile, exists
from featureExtraction.FeatureExtractor import FeatureExtractor
from featureExtraction.AspectsFeatureExtractor import AspectsFeatureExtractor
import xml.etree.ElementTree as ET
from copy import deepcopy

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
SEM_FEATURES_PATH = join(FILE_PATH, "features","SEMVAL")
SEM_TOKENANN_PATH = join(FILE_PATH, "tokenAnnotations","SEMVAL")
SEM_VOCAB_PATH = join(SEM_FEATURES_PATH, "vocab")

VIPS_PATH = join(FILE_PATH, "..", "question-corpora","VIPS")
VIPS_FEATURES_PATH = join(FILE_PATH, "features","VIPS")
VIPS_TOKENANN_PATH = join(FILE_PATH, "tokenAnnotations","VIPS")
VIPS_VOCAB_PATH = join(VIPS_FEATURES_PATH, "vocab")

def generateSemvalFeatureFiles(featureFile, annFile, testSet=False, vocabFile=None, trainSize=None, useCorrectAsRef=False):
    if(isfile(annFile)):
        with open(annFile, "r") as f:
            questionData = json.load(f)
    else:
        questionData = getQuestionData(testSet if testSet else "T")

    if(testSet=="UA"):
        with open(vocabFile, "r") as f:
            trainedVocab = json.load(f)
        for qId, question in questionData.items():
            question["vocabulary"] = trainedVocab[question["id"]]["vocabulary"]
            question["semCats"] = trainedVocab[question["id"]]["semCats"]
            question["classCounts"] = trainedVocab[question["id"]]["classCounts"]
            if(useCorrectAsRef):
                question["studentReferenceAnswers"] = trainedVocab[question["id"]]["studentReferenceAnswers"]

    featureExtrator = FeatureExtractor(simMeasure="fasttext", lang="en")
    for qIdx, qId in enumerate(questionData.keys()):
        question = questionData[qId]
        unseenQuestion = testSet in ["UQ","UD"]
        if(not(testSet) and useCorrectAsRef):
            question["studentReferenceAnswers"] = []
        print(qIdx, qId)
        featureExtrator.setQuestion(question, resetVocabulary=not(testSet))
        for aIdx, answer in enumerate(question["studentAnswers"]):
            predUpdate = not(testSet) and (not(trainSize) or (trainSize and aIdx<trainSize))
            featureExtrator.updateVocab(answer, allowMerge=predUpdate)
            if(predUpdate):
                featureExtrator.updateVocabPredDist(answer, answer["answerCategory"])
                if(useCorrectAsRef and answer["answerCategory"]=="correct"):
                    question["studentReferenceAnswers"].append(answer)
        featureExtrator.prepareReferenceAnswers(useCorrectAsRef=not(unseenQuestion) and useCorrectAsRef)

        for aIdx, answer in enumerate(question["studentAnswers"]):
            featureExtrator.extractFeatures(answer, predDist=not(unseenQuestion), vocab=True)
            # if(not("sulAlign" in answer["features"])):
            #     if("sulAlign" in answer):
            #         answer["features"]["sulAlign"] = answer["sulAlign"]
            #     else:
            #         featureExtrator.addSulAlign(answer, question["referenceAnswers"][0])
    saveFeatureJSON(questionData, featureFile)
    if(not(testSet)): saveTrainingVocab(vocabFile, questionData)
    if(not(isfile(annFile))): saveTokenAnnotation(annFile, questionData)

def generateVipsFeatureFiles(qFile, featureFile, annFile, testSet=False, vocabFile=None, trainSize=None, useCorrectAsRef=False):
    if(isfile(annFile)):
        with open(annFile, "r") as f:
            questionData = json.load(f)
    else:
        with open(qFile, "r") as f:
            questionsObj = json.load(f)
            questionData = {}
            for question in questionsObj["questions"]:
                questionData[question["id"]] = question

    if(testSet=="UA"):
        with open(vocabFile, "r") as f:
            trainedVocab = json.load(f)
        for qId, question in questionData.items():
            question["vocabulary"] = trainedVocab[question["id"]]["vocabulary"]
            question["semCats"] = trainedVocab[question["id"]]["semCats"]
            question["classCounts"] = trainedVocab[question["id"]]["classCounts"]
            if(useCorrectAsRef):
                question["studentReferenceAnswers"] = trainedVocab[question["id"]]["studentReferenceAnswers"]

    featureExtrator = AspectsFeatureExtractor(simMeasure="fasttext", lang="de")
    for qIdx, qId in enumerate(questionData.keys()):
        question = questionData[qId]
        unseenQuestion = testSet == "UQ"
        if(not(testSet) and useCorrectAsRef):
            question["studentReferenceAnswers"] = []
        print(qIdx, question["id"])
        featureExtrator.setQuestion(question, resetVocabulary=not(testSet))
        for aIdx, answer in enumerate(question["studentAnswers"]):
            predUpdate = not(testSet) and (not(trainSize) or (trainSize and aIdx<trainSize))
            featureExtrator.updateVocab(answer, allowMerge=predUpdate)
            if(predUpdate):
                featureExtrator.updateAspectVocabPredDist(answer)
                if(useCorrectAsRef and len(answer["aspects"])==len(question["aspects"]) and all([aspect["label"] == 0 for aspect in answer["aspects"]])):
                    question["studentReferenceAnswers"].append(answer)
        featureExtrator.prepareReferenceAspects(useCorrectAsRef=not(unseenQuestion) and useCorrectAsRef)

        for aIdx, answer in enumerate(question["studentAnswers"]):
            # featureExtrator.extractAspectFeatures(answer, train=not(testSet), predDist=not(unseenQuestion), vocab=True)
            featureExtrator.extractAspectFeatures(answer, train=True, predDist=not(unseenQuestion), vocab=True)
    saveFeatureJSON(questionData, featureFile, True)
    if(not(testSet)): saveTrainingVocab(vocabFile, questionData)
    if(not(isfile(annFile))): saveTokenAnnotation(annFile, questionData)

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
        question["dataset"] = "beetle" if "beetle" in file else "sciEntsBank"
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
                questionDict["studentAnswers"].append({"id": pupAns.attrib["id"], "text":pupAns.text, "answerCategory": pupAns.attrib["accuracy"]})
    return  questionDict

def saveTokenAnnotation(annFile, questionData):
    questionKeys = ["id","title","text","type","aspects","qtype","stype","module","dataset"]
    generalAnswerKeys = ["id","text","correctionOrComment","answerCategory","tokens","alignAnn"]
    aspectKeys = ["aIdx","text","elements","label"]
    saveData = {}
    for qId in questionData.keys():
        question = questionData[qId]
        saveQuestion = {}
        for key in questionKeys:
            if(key in question): saveQuestion[key] = question[key]

        saveQuestion["referenceAnswers"] = []
        for referenceAnswer in question["referenceAnswers"]:
            saveRefAns = {}
            for key in generalAnswerKeys:
                if(key in referenceAnswer): saveRefAns[key] = referenceAnswer[key]
            if("aspects" in referenceAnswer):
                saveRefAns["aspects"] = []
                for aspect in referenceAnswer["aspects"]:
                    saveAspect = {}
                    for key in aspectKeys:
                        if(key in aspect): saveAspect[key] = aspect[key]
                    saveRefAns["aspects"].append(saveAspect)
            saveQuestion["referenceAnswers"].append(saveRefAns)

        saveQuestion["studentAnswers"] = []
        for studentAnswer in question["studentAnswers"]:
            saveStudAns = {}
            for key in generalAnswerKeys:
                if(key in studentAnswer): saveStudAns[key] = studentAnswer[key]
            if("aspects" in studentAnswer):
                saveStudAns["aspects"] = []
                for aspect in studentAnswer["aspects"]:
                    saveAspect = {}
                    for key in aspectKeys:
                        if(key in aspect): saveAspect[key] = aspect[key]
                    saveStudAns["aspects"].append(saveAspect)
            saveQuestion["studentAnswers"].append(saveStudAns)

        saveData[qId] = saveQuestion

    with open(annFile, "w+") as f:
        json.dump(saveData, f, indent=2)

def saveTrainingVocab(vocabFile, questionData):
    vocabData = {}
    for key,val in questionData.items():
        vocabData[key] = {
            "vocabulary": val["vocabulary"],
            "semCats": val["semCats"],
            "classCounts": val["classCounts"]
        }
        if("studentReferenceAnswers" in val):
            vocabData[key]["studentReferenceAnswers"] = val["studentReferenceAnswers"]
    with open(vocabFile, "w+") as f:
        json.dump(vocabData, f, indent=2)

def saveFeatureJSON(questionsAnnotation, featurePath, aspects=False):
    featureData = {}
    aspectKeys = ["features", "unknownContentWords", "refAnsId"]
    wholeKeys = ["features", "refAnsId"]
    for qId,question in questionsAnnotation.items():
        featureData[qId] = {}
        for aIdx, answer in enumerate(question["studentAnswers"]):
            if(aspects):
                featureData[qId][answer["id"]] = []
                for aspect in answer["aspects"]:
                    featureEntry = {"label": aspect["label"] == 0, "category": aspect["label"]}
                    for key in aspectKeys:
                        if(key in aspect): featureEntry[key] = aspect[key]
                    featureData[qId][answer["id"]].append(featureEntry)
            else:
                featureEntry = {"label": answer["answerCategory"] == "correct", "category": answer["answerCategory"], "dataset": question["dataset"]}
                for key in wholeKeys:
                    if(key in answer): featureEntry[key] = answer[key]
                featureData[qId][answer["id"]] = featureEntry

    with open(featurePath, "w+") as f:
        json.dump(featureData, f, indent=2)

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

def getVipQStats():
    with open(join(VIPS_PATH, "CSSAG.json"), "r") as f:
        data = json.load(f)
    for q in data["questions"]:
        qId = q["id"]
        qType = q["type"]
        aspC = len(q["aspects"])
        ansC = len(q["studentAnswers"])
        ansAspC = [0] * (aspC+1)
        labels = [0] * 4
        corrC = [0] * (aspC+1)
        aspCorr = [0] * aspC
        aspDist = {}
        for ans in q["studentAnswers"]:
            sAC = len(ans["aspects"])
            aD = "".join(sorted([str(asp["aIdx"]) for asp in ans["aspects"] if(asp["label"]==0)]))
            if(aD in aspDist):
                aspDist[aD] += 1
            else:
                aspDist[aD] = 1
            ansAspC[sAC] += 1
            labels[-1] += aspC-sAC
            cC = 0
            aspCorr
            labelsn = [asp["label"] for asp in ans["aspects"]]
            if(1 in labelsn and 2 in labelsn):
                print(q["text"], ans["text"])
                input()
            for asp in ans["aspects"]:
                labels[asp["label"]] += 1
                if(asp["label"]==0):
                    aspCorr[asp["aIdx"]] += 1
                    cC += 1
            corrC[cC] += 1

        print("qID: {}\nqType: {}\nasp#: {}\nans#: {}\naspPAns: {}\nlabels: {}\ncorrC: {}\naspCorr: {}\n".format(qId,qType,aspC,ansC,ansAspC,labels,corrC,aspCorr))
        print(aspDist)
        print("\n")

def splitASMA(dataName):
    with open(join(VIPS_PATH, dataName + ".json"), "r") as f:
        data = json.load(f)
    train = {"questions": []}
    UA = {"questions": []}
    UQ = {"questions": []}
    qtestUASel = {
        "3": {'1': 7, '': 13, '012': 0, '12': 3, '2': 0, '02': 1},
        "4": {'01': 7, '0': 1, '': 7, '1': 4},
        "13": {'0': 3, '01': 3, '': 4},
        "26": {'': 6, '03': 1, '0': 3, '0123': 1, '013': 2, '13': 2, '01': 2, '23': 1, '02': 1, '3': 1, '2': 1, '023': 0, '012':0},
        "50717": {'01': 5, '1': 3, '': 6, '0': 0},
        "50721": {'0': 8, '01': 3, '1': 1, '': 4},
        "50722": {'02': 1, '01': 1, '2': 3, '1': 4, '': 3, '12': 4, '0':0, '012': 0},
        "50726": {'': 5, '0': 5, '1': 0, '01': 0},
        "50731": {'01': 6, "0": 1, '1': 2, '': 3},
        "50809": {'': 3, '02': 1, '01':6, '1': 2, '0': 0, '2': 0, '12': 0, '012': 0},
        "50888": {'0': 2, '01': 4, '': 1, '1': 3},
        "50889": {'123': 1, '2': 1, '0123': 1, "01": 1, '1': 2, '': 4, '0': 0, '3': 0, '12':0, '23':0, '13':0}
    }
    for q in data["questions"]:
        if(q["id"] in ["5","14","31", "50817", "50818", "50820", "50890", "50891", "50892", "50895"]):
            UQ["questions"].append(q)
        else:
            testUASel = qtestUASel[q["id"]]
            trainStudentAnswers = []
            testStudentAnswers = []
            for ans in q["studentAnswers"]:
                aD = "".join(sorted([str(asp["aIdx"]) for asp in ans["aspects"] if(asp["label"]==0)]))
                if(testUASel[aD] > 0):
                    testStudentAnswers.append(ans)
                    testUASel[aD] -= 1
                else:
                    trainStudentAnswers.append(ans)
            train["questions"].append(deepcopy(q))
            train["questions"][-1]["studentAnswers"] = trainStudentAnswers
            UA["questions"].append(deepcopy(q))
            UA["questions"][-1]["studentAnswers"] = testStudentAnswers
    for dataSet, setName in zip([train,UA,UQ],["train", "UA", "UQ"]):
        with open(join(VIPS_PATH, dataName + "_" + setName + ".json"), "w+") as f:
            data = json.dump(dataSet, f, indent=4)


def extractSEMVALFeatureData():
    for dataSet in ["train", "UA","UQ", "UD"]:
        featurePath = join(SEM_FEATURES_PATH, dataSet, "MRA.json")
        annPath = join(SEM_TOKENANN_PATH, dataSet + "_MRA.json")
        vocabPath = join(SEM_VOCAB_PATH, "MRA.json")
        testSet = False if(dataSet=="train") else dataSet
        generateSemvalFeatureFiles(featurePath, annPath, testSet, vocabPath)
    for trainSize in [5,10,15,20,25,30]:
        for dataSet in ["train", "UA"]:
            featurePath = join(SEM_FEATURES_PATH, dataSet, "MRA{}.json".format(trainSize))
            annPath = join(SEM_TOKENANN_PATH, dataSet + "_MRA.json")
            vocabPath = join(SEM_VOCAB_PATH, "MRA{}.json".format(trainSize))
            testSet = False if(dataSet=="train") else dataSet
            generateSemvalFeatureFiles(featurePath, annPath, testSet, vocabPath, trainSize)

    for dataSet in ["train", "UA","UQ", "UD"]:
        featurePath = join(SEM_FEATURES_PATH, dataSet, "PRA.json")
        annPath = join(SEM_TOKENANN_PATH, dataSet + "_MRA.json")
        vocabPath = join(SEM_VOCAB_PATH, "PRA.json")
        testSet = False if(dataSet=="train") else dataSet
        generateSemvalFeatureFiles(featurePath, annPath, testSet, vocabPath, None, True)
    for trainSize in [5,10,15,20,25,30]:
        for dataSet in ["train", "UA"]:
            featurePath = join(SEM_FEATURES_PATH, dataSet, "PRA{}.json".format(trainSize))
            annPath = join(SEM_TOKENANN_PATH, dataSet + "_MRA.json")
            vocabPath = join(SEM_VOCAB_PATH, "PRA{}.json".format(trainSize))
            testSet = False if(dataSet=="train") else dataSet
            generateSemvalFeatureFiles(featurePath, annPath, testSet, vocabPath, trainSize, True)

def extractVIPSFeatureData():
    for dataSet in ["train","UA","UQ"]:
        qDataPath = join(VIPS_PATH, "CSSAG_" + dataSet + ".json")
        featurePath = join(VIPS_FEATURES_PATH, dataSet, "CSSAG_ASP_ANN.json")
        annPath = join(VIPS_TOKENANN_PATH, dataSet + "_CSSAG_ASP.json")
        vocabPath = join(VIPS_VOCAB_PATH, "CSSAG_ASP_ANN.json")
        testSet = False if(dataSet=="train") else dataSet
        generateVipsFeatureFiles(qDataPath, featurePath, annPath, testSet, vocabPath)
    # for trainSize in [5,10,15,20,25,30]:
    #     for dataSet in ["train", "UA"]:
    #         featurePath = join(SEM_FEATURES_PATH, dataSet, "MRA{}.json".format(trainSize))
    #         annPath = join(SEM_TOKENANN_PATH, dataSet + "_MRA.json")
    #         vocabPath = join(SEM_VOCAB_PATH, "MRA{}.json".format(trainSize))
    #         testSet = False if(dataSet=="train") else dataSet
    #         generateVipsFeatureFiles(featurePath, annPath, testSet, vocabPath, trainSize)

    for dataSet in ["train", "UA","UQ"]:
        qDataPath = join(VIPS_PATH, "CSSAG_" + dataSet + ".json")
        featurePath = join(VIPS_FEATURES_PATH, dataSet, "CSSAG_ASP_PRA_ANN.json")
        annPath = join(VIPS_TOKENANN_PATH, dataSet + "_CSSAG_ASP.json")
        vocabPath = join(VIPS_VOCAB_PATH, "CSSAG_ASP_PRA_ANN.json")
        testSet = False if(dataSet=="train") else dataSet
        generateVipsFeatureFiles(qDataPath, featurePath, annPath, testSet, vocabPath, None, True)
    # for trainSize in [5,10,15,20,25,30]:
    #     for dataSet in ["train", "UA"]:
    #         featurePath = join(SEM_FEATURES_PATH, dataSet, "PRA{}.json".format(trainSize))
    #         annPath = join(SEM_TOKENANN_PATH, dataSet + "_MRA.json")
    #         vocabPath = join(SEM_VOCAB_PATH, "PRA{}.json".format(trainSize))
    #         testSet = False if(dataSet=="train") else dataSet
    #         generateVipsFeatureFiles(featurePath, annPath, testSet, vocabPath, trainSize, True)

def getQuestionCSV():
    beetleQSum = []
    sciEntsBankQSum = []
    for dataSet in ["UA","UQ","UD"]:
        questionData = getQuestionData(dataSet)
        for qId, question in questionData.items():
            entry = [dataSet, len(question["studentAnswers"]), question["text"], question["referenceAnswers"][0]["text"]]
            if(question["dataset"]=="beetle"):
                beetleQSum.append(entry)
            else:
                sciEntsBankQSum.append(entry)
    header = ["dataSet", "answer#", "question", "reference answer"]
    for filePath, data in [("beetleQuestions.csv", beetleQSum), ("sciEntsBankQuestions.csv", sciEntsBankQSum)]:
        with open(filePath, 'w') as csvfile:
            questionWriter = csv.writer(csvfile)
            questionWriter.writerow(header)
            questionWriter.writerows(data)

def getASMAStats():
    qTypeAspects = {"UA": {}, "UQ": {}}
    dataSetLabels = {}
    for dName in ["VIPS", "CSSAG"]:
        dataSetLabels[dName] = {}
        for sName in ["train", "UA", "UQ"]:
            dataSetLabels[dName][sName] = [0,0,0,0]
            labels = dataSetLabels[dName][sName]
            with open(join(VIPS_PATH, dName + "_" + sName + ".json"), "r") as f:
                data = json.load(f)
            for q in data["questions"]:
                qAC = len(q["aspects"])
                if(sName in qTypeAspects):
                    setAspects = qTypeAspects[sName]
                    if(q["type"] in setAspects):
                        setAspects[q["type"]] += len(q["aspects"])
                    else:
                        setAspects[q["type"]] = len(q["aspects"])
                for a in q["studentAnswers"]:
                    labels[-1] += (qAC - len(a["aspects"]))
                    for asp in a["aspects"]:
                        labels[asp["label"]] += 1

    for set, labelDict in qTypeAspects.items():
        for label, count in labelDict.items():
            print(set, label, count)

    combined = {}
    for data, dataDict in dataSetLabels.items():
        for set, labels in dataDict.items():
            if(set in combined):
                combined[set] += labels
                print(set, combined[set], sum(combined[set][1:]),sum(combined[set]),   ["{:.2f}".format(val) for val in combined[set]/sum(combined[set])])
            else:
                combined[set] = np.array(labels)
            # print(data, set, labels, sum(labels), sum(labels[1:]), ["{:.2f}".format(label/sum(labels)) for label in labels])





if __name__ == "__main__":
    extractSEMVALFeatureData()
    # getVipQStats()
    # splitASMA("VIPS")
    # splitASMA("CSSAG")
    # getASMAStats()
    # getVipQStats()
    # extractVIPSFeatureData()
    # getQuestionCSV()
