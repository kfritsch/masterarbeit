import numpy as np
import pandas as pd
import sys, json, csv, math

from scipy.stats.stats import pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score,   fbeta_score, roc_auc_score, mean_squared_error, cohen_kappa_score, confusion_matrix, average_precision_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from os.path import join, dirname, realpath
FILE_PATH = dirname(realpath(__file__))
SEMVAL_PATH = join(FILE_PATH, "..", "question-corpora","SEMVAL")
TEST_PATH = join(SEMVAL_PATH,"test")
TRAIN_PATH = join(SEMVAL_PATH,"training")
SEMVAL_FEATURES_PATH = join(FILE_PATH, "features","SEMVAL_OLD")
SEMVAL_RES_PATH = join(FILE_PATH, "results","SEMVAL")

VIPS_FEATURES_PATH = join(FILE_PATH, "features","VIPS")
VIPS_RES_PATH = join(FILE_PATH, "results","VIPS")

FEATURE_SETS = {}
FEATURE_SETS["LO"] = ["lemmaRec", "lemmaHeadRec"]
FEATURE_SETS["SO"] = ["contentRec", "contentHeadRec", "contentPrec", "simHist"]
FEATURE_SETS["SUL"] = ["sulAlign"]
FEATURE_SETS["SE"] = ["embDiff", "embSim", "embMQDiff", "embMQSim"]
FEATURE_SETS["L"] = ["pupAbsLen", "refAbsLen", "pupContentLen", "refContentLen"]
FEATURE_SETS["PS"] = ["pupPosDist", "refPosDist", "posSimHist"]
FEATURE_SETS["QO"] =  ["qRefOverlap", "qRefAnsOverlap"]
FEATURE_SETS["N"] =  ["pupNegWordCount", "pupNegPrefixCount", "refNegWordCount", "refNegPrefixCount"]
FEATURE_SETS["PWH"] = ["posDistWeightHist"]
FEATURE_SETS["WC"] = ["distWeightContent"]
FEATURE_SETS["WHC"] = ["distWeightHeadContent"]
FEATURE_SETS["V"] = ["vocab"]
FEATURE_SETS["SC"] = ["semCats"]
FEATURE_SETS["SCH"] =  ["semCatHeads"]

OLD_FEATURE_SETS = {}
OLD_FEATURE_SETS["BF"] = ["lemmaRec", "lemmaHeadRec", "contentRec", "contentHeadRec", "contentPrec", "embDiff", "embSim", "embMQDiff", "embMQSim"]
OLD_FEATURE_SETS["GF"] = OLD_FEATURE_SETS["BF"] + ["absLenDiff", "contentLenDiff",  "simHist"]
OLD_FEATURE_SETS["QSF"] = OLD_FEATURE_SETS["BF"] + ["pupAbsLen", "refAbsLen", "pupContentLen", "refContentLen", "pupPosDist", "refPosDist", "posSimHist", "qRefOverlap", "qRefAnsOverlap"] #, "pupNegWordCount", "pupNegPrefixCount", "refNegWordCount", "refNegPrefixCount"]
OLD_FEATURE_SETS["PDF"] = OLD_FEATURE_SETS["QSF"] + ["distWeightContent", "distWeightHeadContent", "posDistWeightHist"]
OLD_FEATURE_SETS["VBF"] = OLD_FEATURE_SETS["PDF"] + ["vocab"]
OLD_FEATURE_SETS["SBF"] = OLD_FEATURE_SETS["PDF"] + ["semCats"]
OLD_FEATURE_SETS["SHBF"] = OLD_FEATURE_SETS["SBF"] + ["semCatHeads"]

# FEATURE_SETS["GF"] = FEATURE_SETS["TF"] + ["absLenDiff", "contentLenDiff"]
# FEATURE_SETS["STF"] = FEATURE_SETS["TF"] + FEATURE_SETS["SF"]
# FEATURE_SETS["TF"] = FEATURE_SETS["LO"] + FEATURE_SETS["SO"] + FEATURE_SETS["SUL"]

# FEATURE_SETS["AF"] = FEATURE_SETS["QSF"] + ["annWContentPrec", "annWContentRec", "annWContentHeadRec"]
# FEATURE_SETS["PDF"] = FEATURE_SETS["STFLDON"] + ["distWeightContent", "distWeightHeadContent", "posDistWeightHist"]
# FEATURE_SETS["VBF"] = FEATURE_SETS["PDF"] + ["vocab"]
# FEATURE_SETS["SBF"] = FEATURE_SETS["PDF"] + ["semCats"]
# FEATURE_SETS["SHBF"] = FEATURE_SETS["SBF"] + ["semCatHeads"]

ASPECT_FEATURE_SETS = {}
ASPECT_FEATURE_SETS["BF"] = ["lemmaRec", "lemmaHeadRec", "contentRec", "contentHeadRec", "contentPrec", "embDiff", "embSim", "embMQDiff", "embMQSim", "unknownContentWords"]
ASPECT_FEATURE_SETS["GF"] = ASPECT_FEATURE_SETS["BF"] + ["absLenDiff", "contentLenDiff", "lenRatioDiff",  "simHist"]
ASPECT_FEATURE_SETS["QSF"] = ASPECT_FEATURE_SETS["BF"] + ["pupAbsLen", "refAbsLen", "pupContentLen", "refContentLen", "pupPosDist", "refPosDist", "posSimHist", "qRefOverlap", "qRefAnsOverlap", "pupNegWordCount", "pupNegPrefixCount", "refNegWordCount", "refNegPrefixCount", "pupAnsContentLenRatio", "refAnsContentLenRatio"]
ASPECT_FEATURE_SETS["AF"] = ASPECT_FEATURE_SETS["QSF"] + ["annWContentPrec", "annWContentRec", "annWContentHeadRec"]
ASPECT_FEATURE_SETS["PDF"] = ASPECT_FEATURE_SETS["QSF"] + ["distWeightContent", "distWeightHeadContent", "posDistWeightHist"]
ASPECT_FEATURE_SETS["VBF"] = ASPECT_FEATURE_SETS["PDF"] + ["vocab"]
ASPECT_FEATURE_SETS["SBF"] = ASPECT_FEATURE_SETS["PDF"] + ["semCats"]
ASPECT_FEATURE_SETS["SHBF"] = ASPECT_FEATURE_SETS["SBF"] + ["semCatHeads"]

RF_PARAMS = {"n_estimators":200, "max_depth":25, "min_samples_split":4}
############## SEMVAL PREDICTION ################

def semvalDropEvaluation(featureSetup, testSet, resFile):
    featureSets = ["LO", "SO", "SUL", "SE", "L", "PS", "QO", "N", "PWH", "WC", "WHC"] # ["GF","QSF","PDF","VBF","SBF","SHBF"]
    featureDropSet = [["all", []]]
    for drop in featureSets:
        featureDropSet[0][1] += FEATURE_SETS[drop]
        newSet = []
        for feat in featureSets:
            if(not(drop==feat)):
                newSet += FEATURE_SETS[feat]
        featureDropSet.append([drop,newSet])

    RF_PARAMS.update({"n_estimators":200, "max_depth":27, "min_samples_split":4})

    trainData, trainQIds = loadJsonData(join(SEMVAL_FEATURES_PATH,"train",featureSetup+".json"))
    testData, _ = loadJsonData(join(SEMVAL_FEATURES_PATH,testSet,featureSetup+".json"))

    testLabels = testData[:,2].astype("int")
    res = [([], testLabels, testSet, [])]
    for (fsKey, featureSet) in featureDropSet:
        print(fsKey, featureSet)
        res[0][0].append(getPredictionResults(trainData, testData, featureSet))
        res[0][3].append(fsKey)
    print(RF_PARAMS)
    evalResults(res, resFile)

def semvalEvaluation(featureSetup, resFile):
    featureSets =  ["GF","QSF","PDF","VBF","SBF","SHBF"]
    # featureTuples = [(featureKey, False, False) for featureKey in featureSets]
    # trainSizes = [5,10,15,20,25,30]
    RF_PARAMS.update({"n_estimators":100, "max_depth":27, "min_samples_split":3, "random_state":0})

    trainData, trainQIds = loadJsonData(join(SEMVAL_FEATURES_PATH,"train",featureSetup+".json"))
    testUAData, _ = loadJsonData(join(SEMVAL_FEATURES_PATH,"UA",featureSetup+".json"))
    testUQData, _ = loadJsonData(join(SEMVAL_FEATURES_PATH,"UQ",featureSetup+".json"))
    testUDData, _ = loadJsonData(join(SEMVAL_FEATURES_PATH,"UD",featureSetup+".json"))
    # trainCounts = [len(trainData[trainData[:,0]==qId]) for qId in trainQIds]
    # testCounts = [len(testUAData[testUAData[:,0]==qId]) for qId in trainQIds]
    # print(len(trainData)/len(trainQIds),min(trainCounts),max(trainCounts),len(testUAData)/len(trainQIds),min(testCounts),max(testCounts))

    # iterTrainData = [loadJsonData(join(SEMVAL_FEATURES_PATH,"train",featureSetup+"{}.json".format(trainSize)))[0] for trainSize in trainSizes]
    # iterTestUAData = [loadJsonData(join(SEMVAL_FEATURES_PATH,"UA",featureSetup+"{}.json".format(trainSize)))[0] for trainSize in trainSizes]

    labelsUA = testUAData[:,2].astype("int")
    labelsUQ = testUQData[:,2].astype("int")
    labelsUD = testUDData[:,2].astype("int")
    res = [([], labelsUQ, "UQ", []), ([], labelsUD, "UD", []), ([], labelsUA, "UA", []), ([], labelsUA, "UApQ", [])] # + [([], labelsUA, "S{} FULL".format(trainSize), []) for trainSize in trainSizes] + [([], labelsUA, "S{} T{}".format(trainSize, trainSize), []) for trainSize in trainSizes]
    for fsIdx,fsKey in enumerate(featureSets):
        print(fsKey)
        featureSet = OLD_FEATURE_SETS[fsKey]
        if(fsIdx<2):
            res[0][0].append(getPredictionResults(trainData, testUQData, featureSet))
            res[0][3].append(fsKey)
            res[1][0].append(getPredictionResults(trainData, testUDData, featureSet))
            res[1][3].append(fsKey)
        if(fsIdx<3):
            res[2][0].append(getPredictionResults(trainData, testUAData, featureSet))
            res[2][3].append(fsKey)
        #     for sizeIdx, trainSize in enumerate(trainSizes):
        #         res[10+sizeIdx][0].append(getPredictionResults(np.concatenate([iterTrainData[sizeIdx][iterTrainData[sizeIdx][:,0]==qId][:trainSize] for qId  in trainQIds]), iterTestUAData[sizeIdx], featureSet))
        #         res[10+sizeIdx][3].append(fsKey)
        # if(fsIdx==2):
        #     for sizeIdx, trainSize in enumerate(trainSizes):
        #         res[4+sizeIdx][0].append(getPredictionResults(iterTrainData[sizeIdx], iterTestUAData[sizeIdx], featureSet))
        #         res[4+sizeIdx][3].append(fsKey)
        res[3][0].append(getPredictionResults(trainData, testUAData, featureSet, qIds=trainQIds))
        res[3][3].append(fsKey)
    print(RF_PARAMS)
    evalResults(res, resFile)

def vipsEvaluation(featureSetup, resFile):
    featureSets = ["GF","QSF","PDF"]#,"VBF","SBF","SHBF"]
    # trainSizes = [5,10,15,20,25,30]

    trainData, trainQIds = loadJsonData(join(VIPS_FEATURES_PATH,"train",featureSetup+".json"))
    testUAData, _ = loadJsonData(join(VIPS_FEATURES_PATH,"UA",featureSetup+".json"))
    testUQData, _ = loadJsonData(join(VIPS_FEATURES_PATH,"UQ",featureSetup+".json"))
    # trainCounts = [len(trainData[trainData[:,0]==qId]) for qId in trainQIds]
    # testCounts = [len(testUAData[testUAData[:,0]==qId]) for qId in trainQIds]
    # print(len(trainData)/len(trainQIds),min(trainCounts),max(trainCounts),len(testUAData)/len(trainQIds),min(testCounts),max(testCounts))

    # iterTrainData = [loadJsonData(join(SEMVAL_FEATURES_PATH,"train",featureSetup+"{}.json".format(trainSize)))[0] for trainSize in trainSizes]
    # iterTestUAData = [loadJsonData(join(SEMVAL_FEATURES_PATH,"UA",featureSetup+"{}.json".format(trainSize)))[0] for trainSize in trainSizes]

    labelsUA = testUAData[:,2].astype("int")
    labelsUQ = testUQData[:,2].astype("int")
    res = [([], labelsUQ, "UQ", []), ([], labelsUA, "UA", [])] #, ([], labelsUA, "UApQ", [])] # + [([], labelsUA, "S{} FULL".format(trainSize), []) for trainSize in trainSizes] + [([], labelsUA, "S{} T{}".format(trainSize, trainSize), []) for trainSize in trainSizes]
    for fsIdx,fsKey in enumerate(featureSets):
        print(fsKey)
        featureSet = ASPECT_FEATURE_SETS[fsKey]
        if(fsIdx<2):
            res[0][0].append(getPredictionResults(trainData, testUQData, featureSet))
            res[0][3].append(fsKey)
        if(fsIdx<3):
            res[1][0].append(getPredictionResults(trainData, testUAData, featureSet))
            res[1][3].append(fsKey)
            # for sizeIdx, trainSize in enumerate(trainSizes):
            #     res[10+sizeIdx][0].append(getPredictionResults(np.concatenate([iterTrainData[sizeIdx][iterTrainData[sizeIdx][:,0]==qId][:trainSize] for qId  in trainQIds]), iterTestUAData[sizeIdx], featureSet))
            #     res[10+sizeIdx][3].append(fsKey)
        # if(fsIdx==2):
        #     for sizeIdx, trainSize in enumerate(trainSizes):
        #         res[4+sizeIdx][0].append(getPredictionResults(iterTrainData[sizeIdx], iterTestUAData[sizeIdx], featureSet))
        #         res[4+sizeIdx][3].append(fsKey)
        # res[2][0].append(getPredictionResults(trainData, testUAData, featureSet, qIds=trainQIds))
        # res[2][3].append(fsKey)

    evalResults(res, resFile)

def getPredictionResults(trainData, testData, featureSet, qIds=False):
    if(not(qIds)):
        trainFeatures = np.nan_to_num(np.array([np.array(extractFeatureValues(trainSample, featureSet)) for trainSample in trainData[:,3]]))
        testFeatures = np.nan_to_num(np.array([np.array(extractFeatureValues(testSample, featureSet)) for testSample in testData[:,3]]))
        trainLabels = trainData[:,2].astype('int')
        # clf = XGBClassifier(max_depth=7, learning_rate=0.09, n_estimators=50, objective='binary:logistic', booster='gbtree').fit(trainFeatures, trainLabels)
        clf = RandomForestClassifier(**RF_PARAMS).fit(trainFeatures, trainLabels)
        # clf = LogisticRegression(random_state=0, C=0.001,  solver='liblinear', multi_class='ovr').fit(trainFeatures, trainLabels)
        probs = clf.predict_proba(testFeatures)[:,1]

        # preds = probs>0.5
        # testLabels = testData[:,2].astype('int')
        # for i in range(len(preds)):
        #     if(preds[1] and not(testLabels[i])):
        #         print(testData[i][0],testData[i][1],testData[i][4])
        # input()
        return probs
    else:
        probs = np.zeros(len(testData))
        for qId in qIds:
            trainFeatures = np.nan_to_num(np.array([np.array(extractFeatureValues(trainSample, featureSet)) for trainSample in trainData[trainData[:,0]==qId,3]]))
            testFeatures = np.nan_to_num(np.array([np.array(extractFeatureValues(testSample, featureSet)) for testSample in testData[testData[:,0]==qId,3]]))
            trainLabels = trainData[trainData[:,0]==qId,2].astype('int')
            # clf = XGBClassifier(max_depth=20, learning_rate=0.09, n_estimators=40, min_child_weight=2, objective='binary:logistic', booster='gbtree').fit(trainFeatures, trainLabels)
            # clf = MLPClassifier(hidden_layer_sizes=(50, ), activation="relu", solver="adam", batch_size=16).fit(trainFeatures, trainLabels)
            # clf = LogisticRegression(random_state=0, C=0.001,  solver='liblinear', multi_class='ovr').fit(trainFeatures, trainLabels)
            clf = RandomForestClassifier(n_estimators=20, criterion="gini", max_features="sqrt", max_depth=28, min_samples_split=2, random_state=0).fit(trainFeatures, trainLabels)
            proba = clf.predict_proba(testFeatures)
            probs[testData[:,0]==qId] = proba[:,1]
        return probs

def autonomyAt(probs, labels, recallThres=0.9, precisionThres=0.9):
    posCount = sum(labels)
    negCount = len(labels) - posCount
    sortedTup =  sorted(zip(probs,labels), key=lambda pair: pair[0])

    precAutonomy = 0
    precProb = 0
    fp = negCount
    fpThres = 1 - precisionThres
    if(fp/len(labels)<fpThres):
        print(fp, len(labels))
        return [0, 0 , 0, len(labels)]


    recAutonomy = 0
    recProb = 0
    fn = 0
    fnThres = posCount * (1-recallThres)
    for i, (prob, label) in enumerate(sortedTup):
        if(not(label)):
            fp -= 1
            remain = len(labels)-(i+1)
            if(fp/remain<fpThres):
                precProb = prob
                precAutonomy = remain
                print(fp, remain, precAutonomy, precProb)
                break

        fn += label
        if(recAutonomy==0 and fn>fnThres):
            recAutonomy = i
            recProb = prob
            print(recProb, recAutonomy)

    return [recProb, recAutonomy/len(labels), precProb, precAutonomy/len(labels), (len(labels)-recAutonomy-precAutonomy)/len(labels)]



def evalResults(res, resFile):
    metrics = ["acc", "f1",  "f0.5", "f0.1", "avg_prec", "rsme", "cappa", "tn", "fp", "fn", "tp", "neg_prob", "neg_aut", "pos_prob", "pos_aut", "manual"]
    index = []
    size = 0
    evalRes = []
    for resTup in res:
        size += len(resTup[3])
        labels = resTup[1]
        for (probs, f) in zip(resTup[0],resTup[3]):
            index.append((resTup[2], f))
            preds = probs > 0.5
            evalRes.append(np.array([accuracy_score(labels,preds), f1_score(labels,preds), fbeta_score(labels,preds,0.5), fbeta_score(labels,preds,0.1), average_precision_score(labels,probs), mean_squared_error(labels,probs), cohen_kappa_score(labels,preds), *confusion_matrix(labels,preds).ravel(), *autonomyAt(probs, labels)]))
            # predHist = np.zeros((10,2))
            # for i in range(len(probs)):
            #     bin = min(math.floor(probs[i]*10),9)
            #     predHist[bin][labels[i]] += 1
            # print(resTup[2], f)
            # print(predHist)
            # input()
    index = pd.MultiIndex.from_tuples(index, names=['target', 'features'])
    resFrame = pd.DataFrame(np.array(evalRes), index=index, columns=metrics)
    print(resFrame)
    pd.set_option('display.width', 1000)
    pd.set_option('colheader_justify', 'center')

    html_string = '''
        <html>
          <head><title>HTML Pandas Dataframe with CSS</title></head>
          <link rel="stylesheet" type="text/css" href="df_style.css"/>
          <body>
            {table}
          </body>
        </html>.
    '''
    # with open(resFile, 'w') as f:
    #     f.write(html_string.format(table=resFrame.to_html(classes='mystyle')))

def loadJsonData(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)
    npData = []
    qIds = []
    for qId, questionData in data.items():
        qIds.append(qId)
        for aId, answerData in questionData.items():
            if isinstance(answerData, list):
                for aIdx,aspectData in enumerate(answerData):
                    features = aspectData["features"]
                    features["unknownContentWords"] = 0
                    if("unknownContentWords" in aspectData):
                        features["unknownContentWords"] = aspectData["unknownContentWords"]
                    npData.append(np.array([qId,aId,int(aspectData["label"]), features, aIdx]))
            else:
                npData.append(np.array([qId,aId,int(answerData["label"]), answerData["features"]]))
    return np.array(npData), qIds

def extractFeatureValues(featureData, keys):
    values = []
    for key in keys:
        if(not(key in featureData)):
            print(key,featureData)
        val = featureData[key]
        if(isinstance(val,dict)):
            for key2,val2 in val.items():
                if(isinstance(val2, list) or isinstance(val2, tuple)):
                    for i in range(len(val2)):
                        values.append(val2[i])
                else:
                    values.append(val2)
        elif(isinstance(val, list) or isinstance(val, tuple)):
            for i in range(len(val)):
                values.append(val[i])
        else:
            values.append(val)
    return values

def extractFeatureHeader(featureData, keys):
    fields = []
    for key in keys:
        val = featureData[key]
        if(isinstance(val,dict)):
            for key2,val2 in val.items():
                if(isinstance(val2, list) or isinstance(val2, tuple)):
                    for i in range(len(val2)):
                        fields.append(key+ "_" +key2+ "_"+str(i))
                else:
                    fields.append(key+ "_" +key2)
        elif(isinstance(val, list) or isinstance(val, tuple)):
            for i in range(len(val)):
                fields.append(key+ "_" +str(i))
        else:
            fields.append(key)
    return fields

############## OWN DATA PREDICTION (OLD) ################

def genTrainTestData(featureData):
    qIDData = featureData["qID"]
    testData = pd.concat([featureData.loc[qIDData==qID].iloc[-20:] for qID in qIDData.unique()], axis=0)
    trainData = pd.concat([featureData.loc[qIDData==qID].iloc[:-20] for qID in qIDData.unique()], axis=0)
    return testData, trainData

def genUnknownQuestionDataSplit(featureData):
    qIDData = featureData["qID"]
    trainQIDs = qIDData.unique()[2:]
    testData = pd.concat([featureData.loc[qIDData==qID].iloc[-20:] if(qID in trainQIDs) else featureData.loc[qIDData==qID].iloc[20:] for qID in qIDData.unique()], axis=0)
    trainData = pd.concat([featureData.loc[qIDData==qID].iloc[:-20] for qID in trainQIDs], axis=0)
    return testData, trainData

def checkPrediction(featurePath):
    featureData = pd.read_csv(featurePath)
    featureData.insert(3,'bin_label',featureData["label"]<=1)
    testData, trainData = genUnknownQuestionDataSplit(featureData)
    nonFeatures = ["aID", "label", "bin_label", "pupPosDist", "refPosDist", "distWeightContent", "posDistWeightHist", "posSimHist"]
    allFeatures = list(set([feat for feat in featureData.columns if not(feat.split("_")[0] in nonFeatures or feat=="bin_label")]))
    print(allFeatures)

    binTrainLabels = trainData["bin_label"]
    binTestLabels = testData["bin_label"]
    trainLabels = trainData["label"]
    testLabels = testData["label"]

    qIDData = testData["qID"]
    qIDs = list(qIDData.unique())
    binQuestionsIdc = [qIDData==qID for qID in qIDs]
    uQIdc = np.logical_or(qIDData==qIDs[0], qIDData==qIDs[1])

    testDataDrop = testData[allFeatures]
    trainDataDrop = trainData[allFeatures]
    # check out parameter
    # clf = LogisticRegression(random_state=0, solver='sag', multi_class='ovr').fit(trainDataDrop, binTrainLabels)
    # clf = MLPClassifier(hidden_layer_sizes=(100, ), activation="tanh", solver="adam", batch_size=16).fit(trainDataDrop, binTrainLabels)
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=8, random_state=0).fit(trainDataDrop, binTrainLabels)
    preds = clf.predict(testDataDrop)
    print([accuracy_score(binTestLabels, preds)] + [accuracy_score(binTestLabels[bQI], preds[bQI]) for bQI in binQuestionsIdc] + [accuracy_score(binTestLabels[uQIdc], preds[uQIdc])])
    print([np.sum(binTestLabels==1)/len(binTestLabels)] + [np.sum(binTestLabels[bQI]==1)/len(binTestLabels[bQI]) for bQI in binQuestionsIdc] + [np.sum(binTestLabels[uQIdc]==1)/len(binTestLabels[uQIdc])])

############## FEATURE/DATA ANALYSIS (OLD) ################

def singleValueCorrelation(featureData):
    qIDData = featureData["qID"]
    labelData = featureData["bin_label"]
    qIDs = list(qIDData.unique())

    resPD = pd.DataFrame(columns= (["feature", "allQ", "rmAllQ"] + qIDs))
    for feature in ["pupAbsLen","pupContentLen","lemmaRecall","sulAlign_0","sulAlign_1", "contentRec_0", "annWContentRec_0", "contentPrec_0","annWContentPrec_0", "embDiff", "embCorr", "embProd", "embSim", "embMQDiff", "embMQCorr", "embMQProd", "embMQSim"]:
        columnData = featureData[feature]
        # meanColumnData = columnData.copy()
        # meanLabelData = labelData.copy()
        pearsonRes = [pearsonr(columnData,labelData)[0]]
        for qID in qIDs:
            qIdc = qIDData==qID
            meanColumnData.loc[qIdc] -= np.mean(columnData.loc[qIdc])
            meanLabelData.loc[qIdc] -= np.mean(labelData.loc[qIdc])
            pearsonRes.append(pearsonr(columnData.loc[qIdc],labelData.loc[qIdc])[0])
        resPD.loc[len(resPD)] = [feature] + [pearsonRes[0], pearsonr(meanColumnData,meanLabelData)[0]] + pearsonRes[1:]
    print(resPD)

def featureDropoutEvaluation(featureData):
    testData, trainData = genUnknownQuestionDataSplit(featureData)
    nonFeatures = ["aID", "label", "bin_label"]
    allFeatures = list(set([feat.split("_")[0] for feat in featureData.columns if not(feat in nonFeatures)]))

    binTrainLabels = trainData["bin_label"]
    binTestLabels = testData["bin_label"]
    trainLabels = trainData["label"]
    testLabels = testData["label"]

    qIDData = testData["qID"]
    qIDs = list(qIDData.unique())
    binQuestionsIdc = [qIDData==qID for qID in qIDs]

    nonFeatures = ["aID", "label", "bin_label"]
    print(nonFeatures)

    resPD = pd.DataFrame(columns= (["dropout", "overall"] + qIDs))

    for idx,dropout in enumerate(allFeatures):
        featureSet = [feat for feat in featureData.columns if(feat.split("_")[0] in [dropout,"posSimHist","sulAlign"])]
        testDataDrop = testData[featureSet]
        trainDataDrop = trainData[featureSet]
        # check out parameter
        # clf = LogisticRegression(random_state=0, solver='sag', multi_class='ovr').fit(trainDataDrop, binTrainLabels)
        # clf = MLPClassifier(hidden_layer_sizes=(100, ), activation="tanh", solver="adam", batch_size=16).fit(trainDataDrop, binTrainLabels)
        clf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=8, random_state=0).fit(trainDataDrop, binTrainLabels)
        preds = clf.predict(testDataDrop)
        resPD.loc[len(resPD)] = [dropout, accuracy_score(binTestLabels, preds)] + [accuracy_score(binTestLabels[bQI], preds[bQI]) for bQI in binQuestionsIdc]

    # for idx,dropout in enumerate(allFeatures):
    #     featureSet = [feat for feat in featureData.columns if(not(feat in nonFeatures or dropout in feat))]
    #     testDataDrop = testData[featureSet]
    #     trainDataDrop = trainData[featureSet]
    #     # check out parameter
    #     # clf = LogisticRegression(random_state=0, solver='sag', multi_class='ovr').fit(trainDataDrop, binTrainLabels)
    #     # clf = MLPClassifier(hidden_layer_sizes=(100, ), activation="tanh", solver="adam", batch_size=16).fit(trainDataDrop, binTrainLabels)
    #     clf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=4, random_state=0).fit(trainDataDrop, binTrainLabels)
    #     preds = clf.predict(testDataDrop)
    #     resPD.loc[len(resPD)] = [dropout, accuracy_score(binTestLabels, preds)] + [accuracy_score(binTestLabels[bQI], preds[bQI]) for bQI in binQuestionsIdc]
    resPD.loc[len(resPD)] = ["dropout", np.sum(binTestLabels==1)/len(binTestLabels)] + [np.sum(binTestLabels[bQI]==1)/len(binTestLabels[bQI]) for bQI in binQuestionsIdc]
    print(resPD)

def singleValueCorrelation():
    semvalPath = join(FILE_PATH, "..", "question-corpora","SEMVAL")
    trainFile = join(semvalPath,"test","testUAFeaturesEmb.csv")
    trainData = pd.read_csv(trainFile)
    trainData.insert(3,'bin_label',trainData["label"]<=1)
    trainData = trainData.fillna(0)
    labelData = trainData["bin_label"]

    a = ["pupAbsLen","pupContentLen","lemmaRecall","sulAlign_0","sulAlign_1", "contentRec_0", "annWContentRec_0", "contentPrec_0","annWContentPrec_0", "embDiff", "embCorr", "embProd", "embSim", "embMQDiff", "embMQCorr", "embMQProd", "embMQSim"]

    for feature in [feat for feat in trainData.columns if(not(len(feat.split("_"))>1) and not feat in ["qID", "aID", "label"])]:
        columnData = trainData[feature]
        # meanColumnData = columnData.copy()
        # meanLabelData = labelData.copy()
        print(feature, pearsonr(columnData,labelData)[0])

if __name__ == "__main__":
    semvalEvaluation("Vocab",join(SEMVAL_RES_PATH,"Vocab.html"))
    # semvalDropEvaluation("MRA","UA",join(SEMVAL_RES_PATH,"drop_MRA_UA.html"))
    # semvalEvaluation("PRA",join(SEMVAL_RES_PATH,"PRA.html"))
    # checkPrediction("features_unques.csv")

    # vipsEvaluation("CSSAG_ASP",join(VIPS_RES_PATH,"CSSAG_ASP.html"))
    # vipsEvaluation("CSSAG_ASP_ANN",join(VIPS_RES_PATH,"CSSAG_ASP_ANN.html"))
