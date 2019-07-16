import numpy as np
import math
import re, os, sys, json, csv
import pandas as pd
from os.path import join
FILE_PATH = os.path.dirname(os.path.realpath(__file__))
RESULTS_PATH = join(os.path.dirname(os.path.realpath(__file__)), "results")
from time import time
import hunspell
from hunspell_add import addWords
from sklearn.feature_extraction.text import CountVectorizer

LABEL_DICT = {
    "correct": 0,
    "binary_correct": 1,
    "partially_correct": 2,
    "wrong": 3,
    "concept_mix-up": 4,
    "guessing": 5,
    "none": 6
}

STOP_WORDS = []
with open("stopwords.txt", "r") as txtFile:
    for line in txtFile:
        STOP_WORDS.append(line.strip())
STOP_WORDS = set(STOP_WORDS)

spellchecker = hunspell.HunSpell('/usr/share/hunspell/de_DE.dic', '/usr/share/hunspell/de_DE.aff')
addWords(spellchecker)

import spacy
from spacy_iwnlp import spaCyIWNLP
nlp = spacy.load('de', disable=['parser', "ner"])
iwnlp = spaCyIWNLP(lemmatizer_path='lib/IWNLP.Lemmatizer_20170501.json')
nlp.add_pipe(iwnlp)

SUGGEST_COUNTER = 0
IS_WORD_COUNTER = 0

# from stanfordcorenlp import StanfordCoreNLP
# coreNLP = StanfordCoreNLP(r'./stanford-corenlp-full-2018-02-27', lang="de")

# manually spellcheck sentence using hunspell
def spellCheck(sentence):
    global SUGGEST_COUNTER
    global IS_WORD_COUNTER
    # with nlp.disable_pipes('tagger'):
    matchIterators = {}
    doc = nlp(sentence)
    for token in doc:
        word = token.text
        if(len(word)>1):
            if(not spellchecker.spell(word)):
                SUGGEST_COUNTER += 1
                try:
                    sug = spellchecker.suggest(word)
                    if(sug):
                        print(sentence)
                        for i,s in enumerate(sug):
                            print(i,s)
                        take = input("Which suggestion?")
                        if(int(take) in range(len(sug))):
                            sug = sug[0]
                        elif(int(take)==-1):
                            sug = word
                            IS_WORD_COUNTER += 1
                        else:
                            word = take
                        # print("Replacing '{}' with '{}'".format(word,sug))
                    else:
                        # print("Junk found: {}".format(word))
                        continue
                except:
                    # print("Junk found: {}".format(word))
                    continue
                # Create a regex from the word
                try:
                    b = re.compile(r"(^|[^a-zA-Z_]){}($|[^a-zA-Z_])".format(word))
                except:
                    # Replace special regex characters with .
                    try:
                        regex = re.compile('[^a-zA-Z]')
                        word = regex.sub('.', word)
                        b = re.compile(r"(^|[^a-zA-Z_]){}($|[^a-zA-Z_])".format(word))
                    except:
                        # print("Regex Problem: {}".format(word))
                        continue
                # Check if the word was already matched
                if(word in matchIterators):
                    # take the next match
                    match = next(matchIterators[word])
                else:
                    # find all matches, take the first one, add remaining matches to the dict
                    allMatch = re.finditer(b, sentence)
                    match = next(allMatch)
                    matchIterators[word] = allMatch
                # replace the spelling error in the sentence
                if(match):
                    if(match.start() != 0):
                        sug = match[0][0] + sug
                    if(match.end() != len(sentence)):
                        sug = sug + match[0][-1]
                    sentence = re.sub(b, sug, sentence, 1)
    return sentence

def getLemmas(sentence):
    # sentence = spellCheck(sentence)
    doc = nlp(sentence)
    lemmas = []
    for token in doc:
        # if(not(len(token.text)==1) and not(token.pos_ in ["DET", "PRON"])):
        if(not(len(token.text)==1)):
            lemma = token.text if token.lemma_ is None else token.lemma_
            lemmas.append(lemma.lower())
    return lemmas

def getTokens(sentence):
    # sentence = spellCheck(sentence)
    doc = nlp(sentence)
    return [token for token in doc if not(len(token.text)==1) and not(token.pos_ in ["DET", "PRON"])]

def countStats(M, wordsPerAnswer, features, total):
    if(np.shape(M)[0]==0):
        return [None]*5
    return [np.shape(M)[0], np.shape(M)[0]/total, np.sum(np.sum(M))/np.shape(M)[0], np.mean(wordsPerAnswer), str(sorted(list(zip(features, list(np.sum(M, axis=0)))), key=lambda x: x[1], reverse=True)[0:3])]

# group words into similar menaing categories
def tokenInvestigation(annotationPath):
    import bisect
    words = []
    tokens = []
    iwnlp_lemmas = None
    with open(annotationPath) as f:
        annotations = json.load(f)
    wordGroups = {}
    group = "-"
    with open(join(FILE_PATH,'inf-gurxx.csv'), 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        for row in csvreader:
            for word in row:
                wordGroups[word] = row[0]
    print(wordGroups)
    with open(join(FILE_PATH,"advancedTokenInvestigation.html"), 'w') as f:
        for questionAnnotation in annotations["questions"]:
            f.write("<!DOCTYPE html>\n<html>\n<head>\n<title>{}</title>\n</head>\n<body>\n")
            labels = np.array([LABEL_DICT[answer["answerCategory"]] for answer in questionAnnotation["answersAnnotation"]])
            pupilAnswers = [getTokens(answer["correctionOrComment"]) for answer in questionAnnotation["answersAnnotation"]]
            for answer in pupilAnswers:
                for token in answer:
                    # if(token._.iwnlp_lemmas and len(token._.iwnlp_lemmas)>1):
                    #     print(token._.iwnlp_lemmas)
                    #     input()
                    # if(token._.iwnlp_lemmas and token.lemma_!=token._.iwnlp_lemmas[0]):
                    #     print(token.text, token.lemma_, token._.iwnlp_lemmas[0])
                        # print(answer)
                    tokenText = token.text if token._.iwnlp_lemmas is None else token._.iwnlp_lemmas[0]
                    pos = bisect.bisect(words, tokenText)
                    if(not(token.pos_ in ("NOUN","ADJ","VERB"))):
                        continue
                    if(pos>0 and words[pos-1] == tokenText):
                        tokens[pos-1][1] += 1
                    else:
                        group = "-"
                        if(tokenText in wordGroups):
                            group = wordGroups[tokenText]
                        words.insert(pos,tokenText)
                        tokens.insert(pos,[tokenText, 1, group, token.pos_, bool(token._.iwnlp_lemmas), bool(GERMANET.synsets(tokenText))])
            break

        tokens.sort(key = lambda token: (token[3], token[2], token[1], token[0]), reverse=True)
        html = listsToHtmlTable([token[1:] for token in tokens], ["count", "group", "pos", "hasLemma", "inGN"], [token[0] for token in tokens])
        f.write(html)
        f.write("</body>\n</html>\n")

# # positive and negative word lists learned semi supervised
# def removeStopwords(words, stopwords, lemmas):
#     for lemma in words:
#         if(lemma in stopwords or lemma in lemmas): continue
#         print(lemma)
#         lemmas.append(lemma) if(input()) else stopwords.append(lemma)

def filterLemmas(lemmas, filterLemmas=STOP_WORDS):
    return [word for word in lemmas if(not(word in filterLemmas))]

questionStopword = ["oder", "egal", "dann", "welche", "wie", "beliebig", "da", "drin"]

def lemmaOverlap(annotationPath):
    from TokenAnnotator import TokenAnnotator
    tokenAnnotator = TokenAnnotator()
    def lemmasFromText(text):
        tokens =  tokenAnnotator.annotateText(text)
        lemmas = [lemma for token in tokens for lemma in token["lemmas"]]
        cleanLemmas = filterLemmas(list(set(lemmas)))
        cleanLemmas = filterLemmas(cleanLemmas, questionStopword)
        return cleanLemmas

    # lemmasFromText("Das ist Satz eins. Das ist Satz zwei.")

    with open(annotationPath) as f:
        annotations = json.load(f)
    vectorizer = CountVectorizer()
    rowLabels = ["correct", "binary_correct", "partially_correct", "wrong", "concept_mix-up", "guessing", "none", "", "false", ">= part_correct", ">= bin_correct", "total"]
    columnLabels = ["label #", "label %", "avg match #", "avg answer length", "most frequent matchs"]
    M = None
    questionRes = None
    featureNames = None
    additionalPupilAnswers = [0, 1, 1, 2, 3, 5]
    headers = ["Reference Answer Only"]  + ["{} additional Pupil Answers".format(np.sum(additionalPupilAnswers[1:i])) for i in range(2,len(additionalPupilAnswers)+1)]

    for questionAnnotation in annotations["questions"]:
        with open(join(RESULTS_PATH,"results.html"), 'w') as f:
            f.write("<!DOCTYPE html>\n<html>\n<head>\n<title>{}</title>\n</head>\n<body>\n")
            questionLemmas = lemmasFromText("Was ist eine überladene Methode?")
            referenceAnswer =  lemmasFromText(questionAnnotation["referenceAnswer"]["text"])
            referenceAnswer = filterLemmas(referenceAnswer,questionLemmas)
            trainingSentences = [referenceAnswer]
            labels = np.array([LABEL_DICT[answer["answerCategory"]] for answer in questionAnnotation["answersAnnotation"]])
            pupilAnswers = [filterLemmas(lemmasFromText(answer["correctionOrComment"]), questionLemmas) for answer in questionAnnotation["answersAnnotation"]]
            for (add,header) in list(zip(additionalPupilAnswers,headers)):
                binPupilIdc = [i for i in range(len(labels)) if labels[i]<2]
                addIdc = np.random.permutation(binPupilIdc)[0:add]
                trainingSentences += [pupilAnswers[i] for i in addIdc]
                pupilAnswers =  [pupilAnswers[i] for i in range(len(pupilAnswers)) if not(i in addIdc)]
                labels = np.array([labels[i] for i in range(len(labels)) if not(i in addIdc)])
                wordsPerAnswer = np.array([len(ans) for ans in pupilAnswers])
                wordsPerAnswer[wordsPerAnswer==0] = 1
                vectorizer.fit([" ".join(t) for t in trainingSentences])
                featureNames = vectorizer.get_feature_names()
                M = vectorizer.transform([" ".join(p) for p in pupilAnswers]).toarray()
                M = (M>0).astype(int)
                questionRes = [countStats(M[labels == i], wordsPerAnswer[labels == i], featureNames, np.shape(M)[0]) for i in range(7)]
                questionRes.append([""] * len(questionRes[0]))
                questionRes.append(countStats(M[labels > 3], wordsPerAnswer[labels > 3], featureNames, np.shape(M)[0]))
                questionRes.append(countStats(M[labels <= 2], wordsPerAnswer[labels <= 2], featureNames, np.shape(M)[0]))
                questionRes.append(countStats(M[labels <= 1], wordsPerAnswer[labels <= 1], featureNames, np.shape(M)[0]))
                questionRes.append(countStats(M, wordsPerAnswer, featureNames, np.shape(M)[0]))
                html = listsToHtmlTable(questionRes, columnLabels, rowLabels)
                f.write("<h2>{}</h2>\n".format(header))
                f.write(html)
                f.write(str(sorted(list(zip(featureNames, list(np.sum(M, axis=0)))), key=lambda x: x[1], reverse=True)))
            f.write("</body>\n</html>\n")
        break

def listsToHtmlTable(resultRows, columnLabels, rowLabels):
    df = pd.DataFrame(resultRows, index=rowLabels, columns=columnLabels)
    html = df.to_html()
    return html

def saveResultsAsHtml(filename, title, header, resultRows, columnLabels, rowLabels, openFile=False):
    with open(filename, 'w') as f:
        f.write("<!DOCTYPE html>\n<html>\n<head>\n<title>{}</title>\n</head>\n<body>\n")
        for i in range(len(resultRows)):
            df = pd.DataFrame(resultRows[i], index=rowLabels[i], columns=columnLabels[i])
            html = df.to_html()
            f.write("<h2>{}</h2>\n".format(header[i]))
            f.write(html)
        f.write("\n")
        f.write("</body>\n</html>\n")
    if(openFile):
        webbrowser.open('file://' + os.path.realpath(filename))


if __name__ == "__main__":
    # annotationPath = join(FILE_PATH, "../question-annotation/server/data/annotations/GoldStandards.json")
    # # tokenInvestigation(annotationPath)
    # # print(SUGGEST_COUNTER)
    # # print(IS_WORD_COUNTER)
    # # print(spellchecker.spell("unterscheidenn"))
    # # print(spellchecker.suggest("unterscheidenn"))
    # lemmaOverlap(annotationPath)

    # lemmas = getLemmas("Ich guing und guing und kam doch nicht weiter!")
    # print(lemmas)
