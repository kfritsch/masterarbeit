import numpy as np
import pandas as pd
import sys, json, csv, math
from os.path import join, dirname, realpath

from sklearn.metrics import accuracy_score, f1_score, fbeta_score, roc_auc_score, mean_squared_error, cohen_kappa_score, confusion_matrix, average_precision_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing

FILE_PATH = dirname(realpath(__file__))
SEMVAL_FEATURES_PATH = join(FILE_PATH, "features","SEMVAL")
SEMVAL_RES_PATH = join(FILE_PATH, "results","SEMVAL")

VIPS_FEATURES_PATH = join(FILE_PATH, "features","VIPS")
VIPS_RES_PATH = join(FILE_PATH, "results","VIPS")

FEATURE_SETS = {}
FEATURE_SETS["D"] =  ["dataset"]
FEATURE_SETS["LO"] = ["lemmaRec", "lemmaHeadRec"]
FEATURE_SETS["SO"] = ["contentRec", "contentHeadRec", "contentPrec"]
FEATURE_SETS["SH"] = ["simHist"]
FEATURE_SETS["SUL"] = ["sulAlign"]
FEATURE_SETS["SE"] = ["embDiff", "embSim", "embMQDiff", "embMQSim"]
FEATURE_SETS["L"] = ["pupAbsLen", "refAbsLen", "pupContentLen", "refContentLen"]
FEATURE_SETS["LD"] = ["absLenDiff", "contentLenDiff"]
FEATURE_SETS["PD"] = ["pupPosDist", "refPosDist"]
FEATURE_SETS["PS"] = ["posSimHist"]
FEATURE_SETS["QO"] =  ["qRefOverlap", "qRefAnsOverlap"]
FEATURE_SETS["N"] =  ["pupNegWordCount", "pupNegPrefixCount", "refNegWordCount", "refNegPrefixCount"]
FEATURE_SETS["PWH"] = ["posDistWeightHist"]
FEATURE_SETS["WC"] = ["distWeightContent"]
FEATURE_SETS["WHC"] = ["distWeightHeadContent"]
FEATURE_SETS["V"] = ["vocab", "semCats","semCatHeads" ]
FEATURE_SETS["UQ"] =  FEATURE_SETS["LO"] + FEATURE_SETS["SO"] + FEATURE_SETS["SH"] + FEATURE_SETS["SUL"] + FEATURE_SETS["SE"] + FEATURE_SETS["PS"] + FEATURE_SETS["QO"] + FEATURE_SETS["N"]
FEATURE_SETS["UA"] = FEATURE_SETS["UQ"] + FEATURE_SETS["PWH"] + FEATURE_SETS["WC"] + FEATURE_SETS["PD"]
FEATURE_SETS["UApQ"] = FEATURE_SETS["UA"] + FEATURE_SETS["V"]
FEATURE_SETS["ANN"] = ["annWContentPrec", "annWContentRec", "annWContentHeadRec"]
FEATURE_SETS["AL"] = ["pupAnsContentLenRatio", "refAnsContentLenRatio", "pupAbsLen", "refAbsLen", "pupContentLen", "refContentLen"]
FEATURE_SETS["ALD"] = ["absLenDiff", "contentLenDiff","lenRatioDiff"]

# RF_PARAMS = {"n_estimators":50, "max_depth":28, "min_samples_leaf":1, "max_features": 1.0}
RF_PARAMS = {"n_estimators":100, "max_depth":11, "min_samples_leaf":10, "max_features": 1.0}
#GBC_PARAMS = {"learning_rate":0.08, "max_depth":4, "n_estimators":100}
GBC_PARAMS = {"learning_rate":0.3, "max_depth":7, "n_estimators":200} # UA

VIPS_GBC_PARAMS = {"learning_rate":0.01, "max_depth":2, "n_estimators":300} # UA MRA_AM
VIPS_GBC_PARAMS = {"learning_rate":0.01, "max_depth":1, "n_estimators":50} # UA MRA
VIPS_GBC_PARAMS = {"learning_rate":0.1, "max_depth":1, "n_estimators":100} # UQ MRA_AM
VIPS_GBC_PARAMS = {"learning_rate":1.5, "max_depth":10, "n_estimators":200} # UQ MRA
############## SEMVAL PREDICTION ################

def vipsGridSearch():
    featureSetup = "CM_WA"
    resFile = join(VIPS_RES_PATH,"gs_CM_WA_GBC_UA1.csv")
    # featureSets = ["LO", "SO", "SH","SUL", "SE", "L", "PS", "QO", "N", "PWH", "WC", "WHC"]
    featureSets = [ "LO", "SO", "SH", "SE", "L", "LD", "PD", "PS", "QO", "N", "PWH", "WC", "WHC"]
    # featureSets = ["LO", "SO","SUL", "SE", "PS", "PD", "QO"]
    featureSet = []
    for fsKey in featureSets:
        featureSet += FEATURE_SETS[fsKey]

    allTrainData, trainQIds = loadJsonData(join(VIPS_FEATURES_PATH,"train",featureSetup+".json"))
    qTrainData = allTrainData[allTrainData[:,0]==trainQIds[0]]
    trainSplit = int(len(qTrainData)*0.8)
    trainData = qTrainData[:trainSplit]
    testDataTrain = qTrainData[trainSplit:]
    for qId in trainQIds[1:]:
        qTrainData = allTrainData[allTrainData[:,0]==qId]
        trainSplit = int(len(qTrainData)*0.8)
        trainData = np.concatenate((trainData, qTrainData[:trainSplit]), axis=0)
        testDataTrain = np.concatenate((testDataTrain, qTrainData[trainSplit:]), axis=0)
    testDataSets = [testDataTrain,loadJsonData(join(VIPS_FEATURES_PATH,"UA",featureSetup+".json"))[0]]
    testDataNames = ["UAT", "UA"]

    # trainData, trainQIds = loadJsonData(join(VIPS_FEATURES_PATH,"train",featureSetup+".json"))
    # testUQ = loadJsonData(join(VIPS_FEATURES_PATH,"UQ",featureSetup+".json"))[0]
    # uQ1 = int(len(testUQ)*0.25)
    # uQ3 = int(len(testUQ)*0.75)
    # testDataSets = [testUQ[uQ1:uQ3], np.concatenate((testUQ[:uQ1], testUQ[uQ3:]), axis=0)]
    # testDataNames = ["UQ1", "UQ2"]

    res = [([], testDataSet, testSet, []) for (testDataSet, testSet) in zip(testDataSets, testDataNames)]
    trainFeatures = np.nan_to_num(np.array([np.array(extractFeatureValues(trainSample, featureSet)) for trainSample in trainData[:,-1]]))
    trainLabels = trainData[:,-2].astype('int')

    gbcGrid = {"loss":["deviance"], "learning_rate": [0.01,0.1,1], "max_depth":[1,2,5,10,15], "n_estimators":[50, 100,200]}
    rfGrid = {"n_estimators":[100,50], "max_depth": [5,15,30], "min_samples_leaf":[1,10]}
    dtGrid = {"max_depth": [18, 20, 25], "min_samples_leaf": [5, 10]}
    lrGrid = {"solver": ["liblinear"], "C":[100,10000]}
    adaGrid = {"learning_rate": [0.01, 0.1, 1], "n_estimators":[50,100]}
    bagGrid = {"n_estimators":[ 100], "max_samples": [1.0, 0.8, 0.6, 0.1, 0.05], "max_features": [1.0, 0.8, 0.6, 0.1] }
    baseEstimatorGrid = list(ParameterGrid(dtGrid))
    ensembleGrid = list(ParameterGrid(gbcGrid))
    # for baseParams in baseEstimatorGrid:
    #     baseEstimator = DecisionTreeClassifier(**baseParams)
    #     # baseEstimator = LogisticRegression(**baseParams)
    for ensembleParams in ensembleGrid:
        # baseParamName = ["".join([kP[0] for kP in key.split("_")] + [str(val)]) for key,val in baseParams.items()]
        ensembleParamName = ["".join([kP[0] for kP in key.split("_")] + [str(val)]) for key,val in ensembleParams.items()]
        paramName = "_".join(ensembleParamName)# + baseParamName)
        print(paramName)
        # clf = BaggingClassifier(**ensembleParams, base_estimator=baseEstimator)
        # clf = AdaBoostClassifier(**ensembleParams, base_estimator=baseEstimator)
        clf = GradientBoostingClassifier(**ensembleParams, subsample=1)
        # clf = RandomForestClassifier(**ensembleParams)
        clf = clf.fit(trainFeatures, trainLabels)
        # for (feat, thres) in zip(clf.tree_.feature,clf.tree_.threshold):
        #     print(header[feat],thres)
        for setIdx, testData in enumerate(testDataSets):
            testFeatures = np.nan_to_num(np.array([np.array(extractFeatureValues(testSample, featureSet)) for testSample in testData[:,-1]]))
            probs = clf.predict_proba(testFeatures)[:,1]
            res[setIdx][0].append(probs)
            res[setIdx][3].append(paramName)
    evalResults(res, resFile)


def semvalGridSearch():
    featureSetup = "MRA"
    resFile = join(SEMVAL_RES_PATH,"gs_GBC_UQ_final4.csv")
    # featureSets = ["LO", "SO", "SH","SUL", "SE", "L", "PS", "QO", "N", "PWH", "WC", "WHC"]
    featureSets = ["LO","SUL", "SE", "L", "PD", "PS", "SO", "N"]
    # featureSets = ["LO", "SO","SUL", "SE", "PS", "PD", "QO"]
    featureSet = []
    for fsKey in featureSets:
        featureSet += FEATURE_SETS[fsKey]

    allTrainData, trainQIds = loadJsonData(join(SEMVAL_FEATURES_PATH,"train",featureSetup+".json"))
    qTrainData = allTrainData[allTrainData[:,0]==trainQIds[0]]
    trainSplit = int(len(qTrainData)*0.8)
    trainData = qTrainData[:trainSplit]
    testDataTrain = qTrainData[trainSplit:]
    for qId in trainQIds[1:]:
        qTrainData = allTrainData[allTrainData[:,0]==qId]
        trainSplit = int(len(qTrainData)*0.8)
        trainData = np.concatenate((trainData, qTrainData[:trainSplit]), axis=0)
        testDataTrain = np.concatenate((testDataTrain, qTrainData[trainSplit:]), axis=0)
    testDataSets = [testDataTrain,loadJsonData(join(SEMVAL_FEATURES_PATH,"UA",featureSetup+".json"))[0]]
    testDataNames = ["UAT", "UA"]

    # trainData, trainQIds = loadJsonData(join(SEMVAL_FEATURES_PATH,"train",featureSetup+".json"))
    # testUQ = loadJsonData(join(SEMVAL_FEATURES_PATH,"UQ",featureSetup+".json"))[0]
    # testUD = loadJsonData(join(SEMVAL_FEATURES_PATH,"UD",featureSetup+".json"))[0]
    # uQ1 = int(len(testUQ)*0.25)
    # uQ3 = int(len(testUQ)*0.75)
    # uDH = int(len(testUD)*0.5)
    # testDataSets = [testUQ[uQ1:uQ3], np.concatenate((testUQ[:uQ1], testUQ[uQ3:]), axis=0), testUD[:uDH], testUD[uDH:]]
    # testDataNames = ["UQ1", "UQ2", "UD1", "UD2"]

    res = [([], testDataSet, testSet, []) for (testDataSet, testSet) in zip(testDataSets, testDataNames)]
    trainFeatures = np.nan_to_num(np.array([np.array(extractFeatureValues(trainSample, featureSet)) for trainSample in trainData[:,-1]]))
    trainLabels = trainData[:,-2].astype('int')

    gbcGrid = {"loss":["deviance"], "learning_rate": [0.06,0.08,0.1], "max_depth":[1,2,3,4,5,6,7], "n_estimators":[30,50,100]}
    rfGrid = {"n_estimators":[100,50], "max_depth": [5,11,21,26], "min_samples_leaf":[1,10], "max_features": [1.0]}
    dtGrid = {"max_depth": [18, 20, 25], "min_samples_leaf": [5, 10]}
    lrGrid = {"solver": ["liblinear"], "C":[100,10000]}
    adaGrid = {"learning_rate": [0.01, 0.1, 1], "n_estimators":[50,100]}
    bagGrid = {"n_estimators":[ 100], "max_samples": [1.0, 0.8, 0.6, 0.1, 0.05], "max_features": [1.0, 0.8, 0.6, 0.1] }
    baseEstimatorGrid = list(ParameterGrid(dtGrid))
    ensembleGrid = list(ParameterGrid(gbcGrid))
    # for baseParams in baseEstimatorGrid:
    #     baseEstimator = DecisionTreeClassifier(**baseParams)
    #     # baseEstimator = LogisticRegression(**baseParams)
    for ensembleParams in ensembleGrid:
        # baseParamName = ["".join([kP[0] for kP in key.split("_")] + [str(val)]) for key,val in baseParams.items()]
        ensembleParamName = ["".join([kP[0] for kP in key.split("_")] + [str(val)]) for key,val in ensembleParams.items()]
        paramName = "_".join(ensembleParamName)# + baseParamName)
        print(paramName)
        # clf = BaggingClassifier(**ensembleParams, base_estimator=baseEstimator)
        # clf = AdaBoostClassifier(**ensembleParams, base_estimator=baseEstimator)
        clf = GradientBoostingClassifier(**ensembleParams, subsample=1)
        # clf = RandomForestClassifier(**ensembleParams)
        clf = clf.fit(trainFeatures, trainLabels)
        # for (feat, thres) in zip(clf.tree_.feature,clf.tree_.threshold):
        #     print(header[feat],thres)
        for setIdx, testData in enumerate(testDataSets):
            testFeatures = np.nan_to_num(np.array([np.array(extractFeatureValues(testSample, featureSet)) for testSample in testData[:,-1]]))
            probs = clf.predict_proba(testFeatures)[:,1]
            res[setIdx][0].append(probs)
            res[setIdx][3].append(paramName)
    benchResults(res, resFile)


def dropEvaluation(dataSet, featureSetup, testSets, featureSets, resFileName):
    featureDropSet = [["all", []]]
    for drop in featureSets:
        featureDropSet[0][1] += FEATURE_SETS[drop]
        newSet = []
        for feat in featureSets:
            if(not(drop==feat)):
                newSet += FEATURE_SETS[feat]
        featureDropSet.append([drop,newSet])

    if(dataSet=="vips"):
        featurePath = VIPS_FEATURES_PATH
        resPath = VIPS_RES_PATH
    else:
        featurePath = SEMVAL_FEATURES_PATH
        resPath = SEMVAL_RES_PATH

    trainData, trainQIds = loadJsonData(join(featurePath,"train",featureSetup+".json"))
    testDataSets = [loadJsonData(join(featurePath,testSet,featureSetup+".json"))[0] for testSet in testSets]

    res = [([], testData, testSet, []) for (testData, testSet) in zip(testDataSets, testSets)]
    for (fsKey, featureSet) in featureDropSet:
        if(fsKey in FEATURE_SETS):
            print(fsKey, FEATURE_SETS[fsKey])
        trainFeatures = np.nan_to_num(np.array([np.array(extractFeatureValues(trainSample, featureSet)) for trainSample in trainData[:,-1]]))
        trainLabels = trainData[:,-2].astype('int')
        base = GradientBoostingClassifier(**GBC_PARAMS, loss="deviance", subsample=1)
        clf = BaggingClassifier(base_estimator=base, n_jobs=8, n_estimators=8, bootstrap=False).fit(trainFeatures, trainLabels)
        # clf = DecisionTreeClassifier(max_depth=1).fit(trainFeatures, trainLabels)
        # clf = RandomForestClassifier(**RF_PARAMS).fit(trainFeatures, trainLabels)
        # clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1),learning_rate=1,n_estimators=200).fit(trainFeatures, trainLabels)
        # clf = SVC(**SVM_PARAMS).fit(preprocessing.scale(trainFeatures), trainLabels)
        for setIdx, testData in enumerate(testDataSets):
            testFeatures = np.nan_to_num(np.array([np.array(extractFeatureValues(testSample, featureSet)) for testSample in testData[:,-1]]))
            probs = clf.predict_proba(testFeatures)[:,1]
            # probs = clf.predict_proba(preprocessing.scale(testFeatures))[:,1]
            res[setIdx][0].append(probs)
            res[setIdx][3].append(fsKey)
    resFile = join(resPath,resFileName + "_eval.csv")
    evalResults(res, resFile)
    resFile = join(resPath,resFileName + "_bench.csv")
    benchResults(res, dataSet, resFile)

def semvalEvaluation(featureSetup, resFileName):
    featureSets = ["UQ", "UA", "UApQ"]
    trainSizes = [5,10,15,20,25,30]

    trainData, trainQIds = loadJsonData(join(SEMVAL_FEATURES_PATH,"train",featureSetup+".json"))
    testUAData, _ = loadJsonData(join(SEMVAL_FEATURES_PATH,"UA",featureSetup+".json"))
    testUQData, _ = loadJsonData(join(SEMVAL_FEATURES_PATH,"UQ",featureSetup+".json"))
    testUDData, _ = loadJsonData(join(SEMVAL_FEATURES_PATH,"UD",featureSetup+".json"))

    iterTrainData = [loadJsonData(join(SEMVAL_FEATURES_PATH,"train",featureSetup+"{}.json".format(trainSize)))[0] for trainSize in trainSizes]
    iterTestUAData = [loadJsonData(join(SEMVAL_FEATURES_PATH,"UA",featureSetup+"{}.json".format(trainSize)))[0] for trainSize in trainSizes]

    res = [([], testUQData, "UQ", []), ([], testUDData, "UD", []), ([], testUAData, "UA", []), ([], testUAData, "UApQ", [])] + [([], testUAData, "S{} FULL".format(trainSize), []) for trainSize in trainSizes] + [([], testUAData, "S{} T{}".format(trainSize, trainSize), []) for trainSize in trainSizes]
    for fsIdx,fsKey in enumerate(featureSets):
        print(fsKey)
        featureSet = FEATURE_SETS[fsKey]
        if(fsIdx==0):
            GBC_PARAMS = {"learning_rate":0.08, "max_depth":4, "n_estimators":100}
            res[0][0].append(getPredictionResults(trainData, testUQData, featureSet))
            res[0][3].append(fsKey)
            res[1][0].append(getPredictionResults(trainData, testUDData, featureSet))
            res[1][3].append(fsKey)
        if(fsIdx<=1):
            GBC_PARAMS = {"learning_rate":0.3, "max_depth":7, "n_estimators":200} # UA
            res[2][0].append(getPredictionResults(trainData, testUAData, featureSet))
            res[2][3].append(fsKey)
            for sizeIdx, trainSize in enumerate(trainSizes):
                tf = trainSize/5
                GBC_PARAMS = {"learning_rate":1./tf, "max_depth":1*tf, "n_estimators":30*tf}
                res[10+sizeIdx][0].append(getPredictionResults(np.concatenate([iterTrainData[sizeIdx][iterTrainData[sizeIdx][:,0]==qId][:trainSize] for qId  in trainQIds]), iterTestUAData[sizeIdx], featureSet))
                res[10+sizeIdx][3].append(fsKey)
        if(fsIdx==1):
            GBC_PARAMS = {"learning_rate":0.3, "max_depth":7, "n_estimators":200} # UA
            for sizeIdx, trainSize in enumerate(trainSizes):
                res[4+sizeIdx][0].append(getPredictionResults(iterTrainData[sizeIdx], iterTestUAData[sizeIdx], featureSet))
                res[4+sizeIdx][3].append(fsKey)
        GBC_PARAMS = {"learning_rate":1., "max_depth":1, "n_estimators":50}
        res[3][0].append(getPredictionResults(trainData, testUAData, featureSet, qIds=trainQIds))
        res[3][3].append(fsKey)
    resFile = join(SEMVAL_RES_PATH,"eval_" + resFileName)
    evalResults(res, resFile)
    resFile = join(SEMVAL_RES_PATH,"bench_" + resFileName)
    benchResults(res, resFile)

def getPredictionResults(trainData, testData, featureSet, qIds=False):
    if(not(qIds)):
        trainFeatures = np.nan_to_num(np.array([np.array(extractFeatureValues(trainSample, featureSet)) for trainSample in trainData[:,-1]]))
        testFeatures = np.nan_to_num(np.array([np.array(extractFeatureValues(testSample, featureSet)) for testSample in testData[:,-1]]))
        # scaler = preprocessing.StandardScaler().fit(trainFeatures)

        trainLabels = trainData[:,-2].astype('int')
        # clf = RandomForestClassifier(**RF_PARAMS).fit(trainFeatures, trainLabels)
        base = GradientBoostingClassifier(**GBC_PARAMS, loss="deviance")
        clf = BaggingClassifier(base_estimator=base, n_estimators=10, bootstrap=False).fit(trainFeatures, trainLabels).fit(trainFeatures, trainLabels)
        probs = clf.predict_proba(testFeatures)[:,1]
        # clf = SVC(**SVM_PARAMS).fit(preprocessing.scale(trainFeatures), trainLabels)
        # probs = clf.predict_proba(preprocessing.scale(testFeatures))[:,1]

        return probs
    else:
        probs = np.zeros(len(testData))
        for qId in qIds:
            trainFeatures = np.nan_to_num(np.array([np.array(extractFeatureValues(trainSample, featureSet)) for trainSample in trainData[trainData[:,0]==qId,-1]]))
            testFeatures = np.nan_to_num(np.array([np.array(extractFeatureValues(testSample, featureSet)) for testSample in testData[testData[:,0]==qId,-1]]))
            # scaler = preprocessing.StandardScaler().fit(trainFeatures)

            trainLabels = trainData[trainData[:,0]==qId,-2].astype('int')
            # clf = RandomForestClassifier(**RF_PARAMS).fit(trainFeatures, trainLabels)
            base = GradientBoostingClassifier(**GBC_PARAMS, loss="deviance")
            clf = BaggingClassifier(base_estimator=base, n_estimators=10, bootstrap=False).fit(trainFeatures, trainLabels).fit(trainFeatures, trainLabels)
            proba = clf.predict_proba(testFeatures)
            # clf = SVC(**SVM_PARAMS).fit(preprocessing.scale(trainFeatures), trainLabels)
            # proba = clf.predict_proba(preprocessing.scale(testFeatures))
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
        return [0, 0 , 0, len(labels)]


    recAutonomy = 0
    recProb = 0
    fn = 0
    fnThres = posCount * (1-recallThres)
    for i, (prob, label) in enumerate(sortedTup):
        if(not(label)):
            fp -= 1
            remain = len(labels)-(i+1)
            if(remain==0 or fp/remain<fpThres):
                precProb = prob
                precAutonomy = remain
                break

        fn += label
        if(recAutonomy==0 and fn>fnThres):
            recAutonomy = i
            recProb = prob

    return [recProb, recAutonomy/len(labels), precProb, precAutonomy/len(labels), (recAutonomy+precAutonomy)/len(labels)]


def benchResults(res, dataset, resFile):
    pd.options.display.float_format = '{:,.3f}'.format
    if(dataset=="semval"):
        splits = ["beetle", "sciEntsBank", "combined"]
        metrics = ["acc", "cor", "par", "con", "irr", "non"]
        catNames = ["correct", "partially_correct_incomplete", "contradictory", "irrelevant", "non_domain"]
    else:
        splits = ["cssag", "vips", "combined"]
        # metrics = ["acc", "cor", "imp", "mcon", "miss"]
        # catNames = ["correct", "unprecise", "missconception", "missing"]
        metrics = ["acc", "cor", "none"]
        catNames = ["correct", "none"]
    columnTuples = []
    index = []
    evalRes = []
    for resTup in res:
        testData = resTup[1]
        labels = testData[:,-2].astype("int")
        categories = testData[:,-3]
        corpus = testData[:,-4]
        if(dataset=="semval"):
            splitIdcs = [corpus=="beetle", corpus=="sciEntsBank", np.ones(len(labels), dtype=bool)]
        else:
            splitIdcs = [corpus=="CSSAG", corpus=="VIPS", np.ones(len(labels), dtype=bool)]
        counts = []
        for sIdx, splitIdc in enumerate(splitIdcs):
            splitCategories = categories[splitIdc]
            counts += [len(splitCategories)] + [sum(splitCategories==cat) for cat in catNames]
        index.append((resTup[2], "#"))
        evalRes.append(np.array(counts))
        for (probs, f) in zip(resTup[0],resTup[3]):
            index.append((resTup[2], f))
            scores = []
            for sIdx, splitIdc in enumerate(splitIdcs):
                splitProbs = probs[splitIdc]
                splitLabels =labels[splitIdc]
                splitCategories = categories[splitIdc]
                splitPreds = splitProbs > 0.5
                scores += [accuracy_score(splitLabels,splitPreds)] + [accuracy_score(splitLabels[splitCategories==cat],splitPreds[splitCategories==cat]) for cat in catNames]
                if(len(columnTuples)<18):
                    counts = [len(splitCategories)] + [sum(splitCategories==cat) for cat in catNames]
                    columnTuples += [(splits[sIdx], met, catC) for met, catC in zip(metrics, counts)]
            evalRes.append(np.array(scores))
    index = pd.MultiIndex.from_tuples(index, names=['target', 'features'])
    columns = pd.MultiIndex.from_product([splits, metrics], names=['split', 'metric'])
    resFrame = pd.DataFrame(np.array(evalRes), index=index, columns=columns)
    resFrame.to_csv(resFile, float_format='%.3f')
    print(resFrame)

def evalResults(res, resFile):
    pd.options.display.float_format = '{:,.3f}'.format
    metrics = ["acc", "f0.1", "tn", "fp", "fn", "tp", "autonomy1", "autonomy2"]
    index = []
    evalRes = []
    for resTup in res:
        testData = resTup[1]
        labels = testData[:,-2].astype("int")
        for (probs, f) in zip(resTup[0],resTup[3]):
            index.append((resTup[2], f))
            preds = probs > 0.5
            evalRes.append(np.array([accuracy_score(labels,preds), fbeta_score(labels,preds,0.1), *confusion_matrix(labels,preds).ravel(), autonomyAt(probs, labels,recallThres=0.8, precisionThres=0.9)[-1], autonomyAt(probs, labels,recallThres=0.9, precisionThres=0.95)[-1]]))
    index = pd.MultiIndex.from_tuples(index, names=['target', 'features'])
    resFrame = pd.DataFrame(np.array(evalRes), index=index, columns=metrics)
    print(resFrame)
    resFrame.to_csv(resFile, float_format='%.3f')

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
                    npData.append(np.array([qId,aId,aIdx,aspectData["dataset"],aspectData["category"],int(aspectData["label"]), features]))
            else:
                answerData["features"]["dataset"] = answerData["dataset"] == "beetle"
                npData.append(np.array([qId,aId,answerData["dataset"],answerData["category"],int(answerData["label"]), answerData["features"]]))
    return np.array(npData), qIds

def extractFeatureValues(featureData, keys):
    values = []
    for key in keys:
        if(not(key in featureData)):
            # print(key,featureData)
            if(key=="annWContentHeadRec"):
                values.append(0)
                continue
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

if __name__ == "__main__":
    vipsGridSearch()
    # SEMVAL
    # GBC_PARAMS = {"learning_rate":0.3, "max_depth":7, "n_estimators":200} # UA
    # featureSets = [ "LO", "SO", "SH","SUL", "SE", "L", "LD", "PD", "PS", "QO", "N", "PWH", "WC"] # ["WHC"]
    # dropEvaluation("semval","MRA",["UA"],featureSets,"drop_MRA_GBCB_UA2")
    #
    # GBC_PARAMS = {"learning_rate":0.08, "max_depth":4, "n_estimators":100}
    # featureSets = ["LO", "SO", "SH","SUL", "SE", "L", "LD", "PD", "QO", "N"]# ["PS"]
    # dropEvaluation("semval","MRA",["UQ", "UD"],featureSets,"drop_MRA_GBCB_UQ2")
    #
    # # VIPS MRA
    # GBC_PARAMS = {"learning_rate":0.3, "max_depth":7, "n_estimators":200} # UA
    # featureSets = [ "LO", "SO", "SH", "SE", "AL", "ALD", "PD", "PS", "QO", "N", "ANN", "PWH", "WC", "WHC"]
    # dropEvaluation("vips","MRA",["UA"],featureSets,"drop_MRA_GBCB_UA1")
    #
    # GBC_PARAMS = {"learning_rate":0.08, "max_depth":4, "n_estimators":100}
    # featureSets = ["LO", "SO", "SH", "SE", "AL", "ALD", "PD", "PS", "QO", "N", "ANN"]
    # dropEvaluation("vips","MRA",["UQ"],featureSets,"drop_MRA_GBCB_UQ1")
    #
    # # VIPS MRA AM
    # GBC_PARAMS = {"learning_rate":0.3, "max_depth":7, "n_estimators":200} # UA
    # featureSets = [ "LO", "SO", "SH", "SE", "AL", "ALD", "PD", "PS", "QO", "N", "ANN", "PWH", "WC", "WHC"]
    # dropEvaluation("vips","MRA_AM",["UA"],featureSets,"drop_MRA_AM_GBCB_UA1")
    #
    # GBC_PARAMS = {"learning_rate":0.08, "max_depth":4, "n_estimators":100}
    # featureSets = ["LO", "SO", "SH", "SE", "AL", "ALD", "PD", "PS", "QO", "N", "ANN"]
    # dropEvaluation("vips","MRA_AM",["UQ"],featureSets,"drop_MRA_AM_GBCB_UQ1")

    # SEMVAL
    # semvalEvaluation("MRA", "semval_Eval_GBCB.csv")
    # semvalEvaluation("PRA", "semval_Eval_GBCB.csv")

    # VIPS MRA
    # semvalEvaluation("MRA", "vips_Eval_GBCB.csv")
    # semvalEvaluation("MRA_AM", "vips_Eval_GBCB.csv")
    # semvalEvaluation("PRA", "vips_Eval_GBCB.csv")
