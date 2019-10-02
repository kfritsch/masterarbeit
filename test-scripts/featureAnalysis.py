import numpy as np
import pandas as pd
import sys, json, csv, math
from scipy.stats.stats import pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score,   fbeta_score, roc_auc_score, mean_squared_error, cohen_kappa_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from os.path import join, dirname, realpath
FILE_PATH = dirname(realpath(__file__))
SEMVAL_PATH = join(FILE_PATH, "..", "question-corpora","SEMVAL")
TEST_PATH = join(SEMVAL_PATH,"test")
TRAIN_PATH = join(SEMVAL_PATH,"training")

FEATURE_SETS = {}
FEATURE_SETS["BF"] = ["lemmaRec", "lemmaHeadRec", "contentRec", "contentHeadRec", "contentPrec", "embDiff", "embSim", "embMQDiff", "embMQSim", "sulAlign"]
FEATURE_SETS["GF"] = FEATURE_SETS["BF"] + ["absLenDiff", "contentLenDiff", "simHist"]
FEATURE_SETS["QSF"] = FEATURE_SETS["BF"] + ["pupAbsLen", "refAbsLen", "pupContentLen", "refContentLen", "pupPosDist", "refPosDist", "posSimHist", "qRefOverlap", "qRefAnsOverlap", "pupNegWordCount", "pupNegPrefixCount", "refNegWordCount", "refNegPrefixCount"]
FEATURE_SETS["PDF"] = FEATURE_SETS["QSF"] + ["distWeightContent", "distWeightHeadContent", "posDistWeightHist"]
FEATURE_SETS["VBF"] = FEATURE_SETS["PDF"] + ["vocab"]
FEATURE_SETS["SBF"] = FEATURE_SETS["PDF"] + ["semCats"]
FEATURE_SETS["SHBF"] = FEATURE_SETS["SBF"] + ["semCatHeads"]

############## SEMVAL PREDICTION ################

def semvalEvaluation():
    featureSets = ["GF","QSF","PDF","VBF","SBF","SHBF"]

    trainData, trainQIds = loadJsonData(join(TRAIN_PATH,"trainingFeaturesWithVocab.json"))
    testUAData, _ = loadJsonData(join(TEST_PATH,"testUAFeaturesWithVocab.json"))
    testUQData, _ = loadJsonData(join(TEST_PATH,"testUQFeaturesWithVocab.json"))
    testUDData, _ = loadJsonData(join(TEST_PATH,"testUDFeaturesWithVocab.json"))

    labelsUA = testUAData[:,2].astype("int")
    labelsUQ = testUQData[:,2].astype("int")
    labelsUD = testUDData[:,2].astype("int")
    res = [([], labelsUQ, "UQ", []), ([], labelsUD, "UD", []), ([], labelsUA, "UA", []), ([], labelsUA, "UApQ", [])]
    for fsIdx,fsKey in enumerate(featureSets):
        print(fsKey)
        featureSet = FEATURE_SETS[fsKey]
        if(fsIdx<2):
            res[0][0].append(getPredictionResults(trainData, testUQData, featureSet))
            res[0][3].append(fsKey)
            res[1][0].append(getPredictionResults(trainData, testUDData, featureSet))
            res[1][3].append(fsKey)
        if(fsIdx<3):
            res[2][0].append(getPredictionResults(trainData, testUAData, featureSet))
            res[2][3].append(fsKey)
        res[3][0].append(getPredictionResults(trainData, testUAData, featureSet, qIds=trainQIds))
        res[3][3].append(fsKey)
    evalResults(res)

def getPredictionResults(trainData, testData, featureSet, qIds=False):
    if(not(qIds)):
        extractFeatureHeader
        trainFeatures = np.nan_to_num(np.array([np.array(extractFeatureValues(trainSample, featureSet)) for trainSample in trainData[:,3]]))
        testFeatures = np.nan_to_num(np.array([np.array(extractFeatureValues(testSample, featureSet)) for testSample in testData[:,3]]))
        trainLabels = trainData[:,2].astype('int')
        # clf = XGBClassifier(max_depth=7, learning_rate=0.09, n_estimators=50, objective='binary:logistic', booster='gbtree').fit(trainFeatures, trainLabels)
        clf = RandomForestClassifier(n_estimators=50, criterion="gini", max_features="sqrt", max_depth=28, min_samples_split=2, random_state=0).fit(trainFeatures, trainLabels)
        probs = clf.predict_proba(testFeatures)[:,1]
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

def evalResults(res):
    metrics = ["acc", "f1",  "f0.5", "roc_auc", "rsme", "cappa", "tn", "fp", "fn", "tp", "acc_0.7", "%_0.7", "acc_0.9", "%_0.9"]
    index = []
    size = 0
    evalRes = []
    for resTup in res:
        size += len(resTup[3])
        labels = resTup[1]
        for (probs, f) in zip(resTup[0],resTup[3]):
            index.append((resTup[2], f))
            preds = probs > 0.5
            saveIdc = np.logical_or(probs>0.7, probs<0.3)
            saveIdc2 = np.logical_or(probs>0.9, probs<0.2)
            evalRes.append(np.array([accuracy_score(labels,preds), f1_score(labels,preds), fbeta_score(labels,preds,0.5), roc_auc_score(labels,probs), mean_squared_error(labels,probs), cohen_kappa_score(labels,preds), *confusion_matrix(labels,preds).ravel(), accuracy_score(labels[saveIdc],preds[saveIdc]),sum(saveIdc)/len(probs),accuracy_score(labels[saveIdc2],preds[saveIdc2]),sum(saveIdc2)/len(probs)]))
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
    with open('resTable.html', 'w') as f:
        f.write(html_string.format(table=resFrame.to_html(classes='mystyle')))

def loadJsonData(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)
    npData = []
    qIds = []
    for qId, qData in data.items():
        qIds.append(qId)
        for aId, aData in qData.items():
            npData.append(np.array([qId,aId,int(aData["label"]), aData["features"]]))
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

def getFeatureData(featurePath):

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
    semvalEvaluation()
    # checkPrediction("features_unques.csv")
