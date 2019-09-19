import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr
import re, os, sys, json, csv, math, codecs
from os.path import join, dirname, realpath, isdir, isfile, exists
FILE_PATH = dirname(realpath(__file__))
from FeatureExtractor import FeatureExtractor
import xml.etree.ElementTree as ET

SEMVAL_PATH = join(FILE_PATH, "..", "question-corpora","SEMVAL")
TEST_PATH = join(SEMVAL_PATH,"test")
TRAIN_PATH = join(SEMVAL_PATH,"training")

def preprocessQuestionData(questionDataPath, featurePath, annotationOut=False):
    with open(questionDataPath, "r") as f:
        questionsAnnotation = json.load(f)
    featureExtrator = FeatureExtractor(simMeasure="fasttext", lang="de")
    for qIdx, question in enumerate(questionsAnnotation["questions"]):
        print(question["title"])
        featureExtrator.setQuestion(question)
        for aIdx, answer in enumerate(question["studentAnswers"]):
            featureExtrator.extractFeatures(answer)
            featureExtrator.updateVocabPredDist(answer)
            print(answer["features"])
            sys.exit(0)
            # if(qIdx>=2 and aIdx < len(question["studentAnswers"])-20): featureExtrator.updateVocabPredDist(answer)
        for answer in question["studentAnswers"]:
            featureExtrator.addPredDistBasedFeatures(answer)
    saveFeatureCSV(questionsAnnotation, featurePath)
    # if(annotationOut):
    #     with open(annotationOut, "w+") as f:
    #         json.dump(questionsAnnotation, f, indent=4)

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

def preprocessTrainingData(questions):
    featureExtrator = FeatureExtractor(simMeasure="fasttext", lang="en")
    for qIdx, question in enumerate(questions):
        print(qIdx, question["id"], question["text"])
        featureExtrator.setQuestion(question)
        for aIdx, answer in enumerate(question["studentAnswers"]):
            featureExtrator.extractFeatures(answer, questionDemotion=False)
            featureExtrator.updateVocabPredDist(answer)
        for answer in question["studentAnswers"]:
            featureExtrator.addPredDistBasedFeatures(answer)
    return questions

def processSemvalTrainingData():
    beetlePath = join(TRAIN_PATH, "beetle")
    sciEntsBankPath = join(TRAIN_PATH, "sciEntsBank")
    annFile = join(TRAIN_PATH, "trainedAnnotationsHalf.json")
    featFile = join(TRAIN_PATH, "trainingFeaturesHalf.csv")

    featureExtrator = FeatureExtractor(simMeasure="fasttext", lang="en")

    questionFiles = []
    for path in [beetlePath, sciEntsBankPath]:
        for root, _, files in os.walk(path):
            for file in files:
                if '.xml' == file[-4:]:
                    questionFiles.append(join(root, file))

    if(isfile(annFile)):
        with open(annFile, "r") as f:
            questionsAnnotation = json.load(f)
    else:
        questionsAnnotation = {}
    print("Already processed: " + str(len(questionsAnnotation.keys())))

    for qIdx, file in enumerate(questionFiles):
        question = parseSemval(file)
        if(question["id"] in questionsAnnotation):
            continue
        print(qIdx, question["id"], question["text"])
        featureExtrator.setQuestion(question)
        for aIdx, answer in enumerate(question["studentAnswers"]):
            featureExtrator.updateVocab(answer)
            if(aIdx < len(question["studentAnswers"])/2):
                featureExtrator.updateVocabPredDist(answer, answer["answerCategory"])

        # print("VOCAB")
        # for key, val in question["vocabulary"].items():
        #     print(key, val)
        # print()
        print("SEMCATs")
        for key, val in question["semCats"].items():
            if(len(val["catGroup"])>1): print(key, val["catGroup"])
        print()
        print()
        for aIdx, answer in enumerate(question["studentAnswers"]):
            if(aIdx >= len(question["studentAnswers"])/2):
                featureExtrator.extractFeatures(answer)
                featureExtrator.addPredDistBasedFeatures(answer)
        questionsAnnotation[question["id"]] = question
        with open(annFile, "w+") as f:
            json.dump(questionsAnnotation, f, indent=4)

    questions = [val for key,val in questionsAnnotation.items()]
    saveFeatureCSV({"questions":questions}, featFile)

def processSemvalUATestData():
    beetlePath = join(TEST_PATH, "beetle")
    sciEntsBankPath = join(TEST_PATH, "sciEntsBank")
    uaPathes = [join(beetlePath,"test-unseen-answers"), join(sciEntsBankPath,"test-unseen-answers")]

    featFile = join(TEST_PATH, "testUAFeaturesHalf.csv")
    annFile = join(TEST_PATH, "testUAAnnotationsHalf.json")
    if(isfile(annFile)):
        with open(annFile, "r") as f:
            questionsAnnotation = json.load(f)
    else:
        questionsAnnotation = {}

    questionFiles = []
    for path in uaPathes:
        for root, _, files in os.walk(path):
            for file in files:
                if '.xml' == file[-4:]:
                    questionFiles.append(join(root, file))

    trainAnnotations= join(TRAIN_PATH, "trainedAnnotationsHalf.json")
    with open(trainAnnotations, "r") as f:
        trainAnnotations = json.load(f)
    # for key,val in trainAnnotations.items():
    #     print(key)
    #     print(val["text"])
    #     print(val["referenceAnswer"]["text"])
    #     for vocab, dist in val["vocabulary"].items():
    #         print(vocab, dist)
    #     input()


    featureExtrator = FeatureExtractor(simMeasure="fasttext", lang="en")

    for qIdx, file in enumerate(questionFiles):
        testQuestion = parseSemval(file)
        if(testQuestion["id"] in questionsAnnotation):
            continue
        if(not(testQuestion["id"] in trainAnnotations)):
            print("missing: " + testQuestion["id"])
            continue
        question = trainAnnotations[testQuestion["id"]]
        question["studentAnswers"] = testQuestion["studentAnswers"]
        print(qIdx, question["id"], question["text"])
        featureExtrator.setQuestion(question, resetVocabulary=False)
        for aIdx, answer in enumerate(question["studentAnswers"]):
            featureExtrator.updateVocab(answer)
            featureExtrator.extractFeatures(answer)
            featureExtrator.addPredDistBasedFeatures(answer)
        questionsAnnotation[question["id"]] = question
        with open(annFile, "w+") as f:
            json.dump(questionsAnnotation, f, indent=4)

    questions = [val for key,val in questionsAnnotation.items()]
    saveFeatureCSV({"questions":questions}, featFile)

def processSemvalUQTestData():
    beetlePath = join(TEST_PATH, "beetle")
    sciEntsBankPath = join(TEST_PATH, "sciEntsBank")
    uQPathes =  [join(beetlePath,"test-unseen-questions"), join(sciEntsBankPath,"test-unseen-questions")]

    featFile = join(TEST_PATH, "testUQFeatures.csv")
    annFile = join(TEST_PATH, "testUQAnnotations.json")


    featureExtrator = FeatureExtractor(simMeasure="fasttext", lang="en")

    questionFiles = []
    for path in uQPathes:
        for root, _, files in os.walk(path):
            for file in files:
                if '.xml' == file[-4:]:
                    questionFiles.append(join(root, file))

    if(isfile(annFile)):
        with open(annFile, "r") as f:
            questionsAnnotation = json.load(f)
    else:
        questionsAnnotation = {}

    for qIdx, file in enumerate(questionFiles):
        question = parseSemval(file)
        if(question["id"] in questionsAnnotation):
            continue
        print(qIdx, question["id"], question["text"])
        featureExtrator.setQuestion(question, resetVocabulary=True)
        for aIdx, answer in enumerate(question["studentAnswers"]):
            featureExtrator.updateVocab(answer)
            featureExtrator.extractFeatures(answer)
        questionsAnnotation[question["id"]] = question
        with open(annFile, "w+") as f:
            json.dump(questionsAnnotation, f, indent=4)

    questions = [val for key,val in questionsAnnotation.items()]
    saveFeatureCSV({"questions":questions}, featFile)

def stripToTokenInfo(fromFile, toFile):
    with open(fromFile, "r") as f:
        questionsAnnotation = json.load(f)
    for qId in questionsAnnotation.keys():
        questionData = questionsAnnotation[qId]
        del questionData["semCats"]
        del questionData["vocabulary"]
        del questionData["classCounts"]
        refAns = questionData["referenceAnswer"]
        del refAns["alignAnn"]
        studentAnswers = questionData["studentAnswers"]
        for studentAnswer in studentAnswers:
            del studentAnswer["alignAnn"]
            studentAnswer["sulAlign"] = studentAnswer["features"]["sulAlign"]
            del studentAnswer["features"]
    with open(toFile, "w+") as f:
        json.dump(questionsAnnotation, f, indent=4)

def featuresFromTokenAnnotation(trainFile, featureFile, testSet=False):
    path = TEST_PATH if testSet else TRAIN_PATH
    featPath = join(path,featureFile)
    simMatDir = join(TRAIN_PATH,"simMatrices")

    featureExtrator = FeatureExtractor(simMeasure="fasttext", lang="en")

    if(testSet):
        with open(join(TEST_PATH, "rawTest" + testSet + "TokenAnnotation.json"), "r") as f:
            inputAnnotation = json.load(f)
        with open(join(TRAIN_PATH, trainFile), "r") as f:
            trainedVocab = json.load(f)
    else:
        with open(join(TRAIN_PATH, "rawTrainingTokenAnnotation.json"), "r") as f:
            inputAnnotation = json.load(f)

    # with open(join(TEST_PATH, "rawTestUATokenAnnotation.json"), "r") as f:
    #     inputAnnotation = json.load(f)

    for qIdx, qId in enumerate(inputAnnotation.keys()):
        simMatPath = join(simMatDir,qId+".csv")
        if(exists(simMatPath)):
            simMatrix = pd.read_csv(simMatPath)
        else:
            simMatrix = pd.DataFrame(columns = ["WORDS"])
        initialSimSolumns = len(simMatrix.columns)
        simMatrix.set_index("WORDS",inplace=True)


        question = inputAnnotation[qId]
        if(testSet=="UA"):
            questionVocab = trainedVocab[question["id"]]
            questionVocab["studentAnswers"] = question["studentAnswers"]
            question = questionVocab

        print(qIdx)#, question["id"], question["text"])
        featureExtrator.setQuestion(question, resetVocabulary=not(testSet), simMatrix=simMatrix)
        for aIdx, answer in enumerate(question["studentAnswers"]):
            featureExtrator.updateVocab(answer)
            if(not(testSet)): featureExtrator.updateVocabPredDist(answer, answer["answerCategory"])

        for aIdx, answer in enumerate(question["studentAnswers"]):
            featureExtrator.extractFeatures(answer)
            if(not(testSet=="UQ")):
                featureExtrator.addPredDistBasedFeatures(answer)
            if("sulAlign" in answer and not("sulAlign" in answer["features"])):
                answer["features"]["sulAlign"] = answer["sulAlign"]

        if(len(simMatrix.columns)>initialSimSolumns):
            simMatrix.to_csv(simMatPath)

    questions = [val for key,val in inputAnnotation.items()]
    saveFeatureCSV({"questions":questions}, featPath)

    if(not(testSet)):
        for key,val in inputAnnotation.items():
            del val["studentAnswers"]
        with open(join(TRAIN_PATH, trainFile), "w+") as f:
            json.dump(inputAnnotation, f, indent=4)

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



# analyseVocabDiff()
# featuresFromTokenAnnotation("trainingVocabEmb.json", "trainingFeaturesEmb.csv", testSet=False)
# featuresFromTokenAnnotation("trainingVocabEmb.json", "testUAFeaturesEmb.csv", testSet="UA")
featuresFromTokenAnnotation("trainingVocabEmb.json", "testUQFeaturesEmb.csv", testSet="UQ")

# stripToTokenInfo(join(TRAIN_PATH, "trainedAnnotations.json"),join(TRAIN_PATH, "rawTrainingTokenAnnotation.json"))
# stripToTokenInfo(join(TEST_PATH, "testUAAnnotations.json"),join(TEST_PATH, "rawTestUATokenAnnotation.json"))
# stripToTokenInfo(join(TEST_PATH, "testUQAnnotations.json"),join(TEST_PATH, "rawTestUQTokenAnnotation.json"))

# preprocessQuestionData("AnnotatedGoldStandards.json", "features_unques.csv", "AnnotatedGoldStandards.json")
