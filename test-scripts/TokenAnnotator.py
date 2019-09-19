# Germalemmatizer = GermaLemma(tiger_corpus="tiger/tiger_release_aug07.corrected.16012013.conll09")
# Germalemmatizer.save_to_pickle("tiger/tiger_lemmas.pkl")
from customIWNLPLemmatizer import CustomIWNLPLemmatizer
from customGermaLemma import CustomGermaLemma
from subprocess import check_output
import spacy
from coreNlp import StanfordCoreNLP
import re
import json

RFTAGGER_UD_MAP = {
    "ADJA":"ADJ",
    "ADJD":"ADJ",
    "ADV":"ADV",
    "APPO":"ADP",
    "APPR":"ADP",
    "APPRART":"ADP",
    "APZR":"ADP",
    "ART":"DET",
    "CARD":"NUM",
    "CONJ":{
        "Comp":"CCONJ",
        "Coord":"CONJ",
        "SubFin":"SCONJ",
        "SubInf":"SCONJ"
    },
    "FM":"X",
    "ITJ":"INTJ",
    "N":{
        "Reg":"NOUN",
        "Name":"PROPN"
    },
    "PART":"PART",
    "PRO":"PRON",
    "PROADV":"ADV",
    "SYM":"PUNCT",
    "TRUNC":"X",
    "VFIN":{
        "Aux":"AUX",
        "Mod":"AUX",
        "Full":"VERB"
    },
    "VIMP":{
        "Aux":"AUX",
        "Mod":"VERB",
        "Full":"VERB"
    },
    "VINF":{
        "Aux":"AUX",
        "Mod":"VERB",
        "Full":"VERB"
    },
    "VPP":{
        "Aux":"AUX",
        "Mod":"VERB",
        "Full":"VERB"
    },
}

STTS_UD_MAP = {
    "ADJA":"ADJ",
    "ADJD":"ADJ",
    "ADV":"ADV",
    "APPO":"ADP",
    "APPR":"ADP",
    "APPRART":"ADP",
    "APZR":"ADP",
    "ART":"DET",
    "CARD":"NUM",
    "FM":"X",
    "ITJ":"INTJ",
    "KOUI":"SCONJ",
    "KOUS":"SCONJ",
    "KON":"CONJ",
    "KOKOM":"CCONJ",
    "NN":"NOUN",
    "NE":"PROPN",
    "ORD":"NUM",
    "PDS":"PRON",
    "PDAT":"PRON",
    "PIS":"PRON",
    "PIAT":"PRON",
    "PIDAT":"PRON",
    "PPER":"PRON",
    "PPOSS":"PRON",
    "PPOSAT":"PRON",
    "PRELS":"PRON",
    "PRELAT":"PRON",
    "PRF":"PRON",
    "PWS":"PRON",
    "PWAT":"PRON",
    "PWAV":"PRON",
    "PAV":"ADV",
    "PROAV":"ADV",
    "PTKZU":"PART",
    "PTKNEG":"PART",
    "PTKVZ":"PART",
    "PTKANT":"PART",
    "PTKA":"PART",
    "SGML":"X",
    "SPELL":"X",
    "TRUNC":"X",
    "VVFIN":"VERB",
    "VVIMP":"VERB",
    "VVINF":"VERB",
    "VVIZU":"VERB",
    "VVPP":"VERB",
    "VAFIN":"VERB",
    "VAIMP":"AUX",
    "VAINF":"AUX",
    "VAPP":"AUX",
    "VMFIN":"VERB",
    "VMINF":"VERB",
    "VMPP":"VERB",
    #verb
    "XY":"X",
    "$,":"PUNCT",
    "$.":"PUNCT",
    "$(":"PUNCT",
    "$[":"PUNCT"
}

RFTAGGER_STTS_MAP = {
    "ADJA":"ADJA",
    "ADJD":"ADJD",
    "ADV":"ADV",
    "APPO":"APPO",
    "APPR":"APPR",
    "APPRART":"APPRART",
    "APZR":"APZR",
    "ART":"ART",
    "CARD":"CARD", # also ORD
    "CONJ":{
        "Comp":"KOKOM",
        "Coord":"KON",
        "SubFin":"KOUS",
        "SubInf":"KOUI"
    },
    "FM":"FM",
    "ITJ":"ITJ",
    "N":{
        "Reg":"NN",
        "Name":"NE"
    },
    "PART":{
        "Ans":"PTKANT",
        "Deg":"PTKA",
        "Neg":"PTKNEG",
        "Zu":"PTKZU",
        "Verb":"PTKVZ"
    },
    "PRO":{
        "Dem":{
            "Attr":"PDAT",
            "Subst":"PDS"
        },
        "Indef":{
            "Attr":"PIAT",
            "Subst":"PIS"
        },
        "Inter":{
            "Attr":"PWAT",
            "Subst":"PWS"
        },
        "Pers":"PPER",
        "Refl":"PRF",
        "Rel":{
            "Attr":"PRELAT",
            "Subst":"PRELS"
        },
        "Poss":{
            "Attr":"PPOSAT",
            "Subst":"PPOSS"
        }
    },
    "PROADV":"PAV",
    "SYM":{
        "Pun":"$.",
        "Quot":"$(",
        "Paren":"$(",
        "Other":"$("
    },
    "TRUNC":"TRUNC",
    "VFIN":{
        "Aux":"VAFIN",
        "Mod":"VMFIN",
        "Full":"VVFIN"
    },
    "VIMP":{
        "Aux":"VAIMP",
        "Mod":"VMIMP",
        "Full":"VVIMP"
    },
    "VINF":{
        "Aux":"VAINF",
        "Mod":"VMINF",
        "Full":"VVINF"
    },
    "VPP":{
        "Aux":"VAPP",
        "Mod":"VMPP",
        "Full":"VVPP"
    }, # also VVIZU   Infinitiv mit ``zu'', anzukommen, loszulassen
}

RF_POS_PROPS_MAP = {
    "N": ["type","case","number","gender"],
    "ART": ["type","case","number","gender"],
    "PRO": ["type","usage","person","case","number","gender"],
    "VFIN": ["type","person","number","tense","mood"],
    "VIMP": ["type","person","number"],
    "VINF": ["type","subtype"],
    "VPP": ["type","subtype"],
    "PART":["type"],
    "ADJA": ["degree","case","number","gender"],
    "ADJD": ["degree"],
    "APPO": ["case"],
    "APPR": ["case"],
    "APPRART": ["case","number","gender"],
}

POS_CORRECTIONS = {
    "sich": "PRON"
}

SHORT_LEMMA_POS = ["n", "a", "v"]

class TokenAnnotator(object):
    def __init__(self):
        self.stanfordCoreNLP = StanfordCoreNLP('http://localhost', port=9000)
        self.spacyNLP = spacy.load('de')
        with open("infVocab.json", "r") as f:
            self.infoVocab = json.load(f)
        # self.iwnlpLemmatizer = CustomIWNLPLemmatizer('lib/IWNLP.Lemmatizer_20170501.json')
        self.germaLemmatizer = CustomGermaLemma(pickle="tiger/tiger_lemmas.pkl")

    def checkPos(self, word, pos):
        if(word in POS_CORRECTIONS and POS_CORRECTIONS[word]!=pos):
            return POS_CORRECTIONS[word]
        else:
            return pos

    def getPossibleLemmas(self, word, pos, props=None):
        composita = None
        if(pos in ["NOUN","ADJ","ADV","VERB","AUX"]):
            wordLemmas, composita = self.germaLemmatizer.find_lemma(word, pos, props)
            if(wordLemmas):
                return wordLemmas, composita
        return None, None

    def getAlignmentLemma(self, word, pos):
        lemmaPos = STTS_UD_MAP[pos]
        if(lemmaPos == "AUX"): lemmaPos = "VERB"
        lemmaPos = self.checkPos(word, lemmaPos)
        lemmas, _ = self.getPossibleLemmas(word, lemmaPos)
        if(lemmas):
            return lemmas[0]
        else:
            return word

    def getAlignmentAnnotation(self, text):
        annotation = self.stanfordCoreNLP.annotate(text, properties={
            'annotators': 'ssplit, depparse, ner',
            'pipelineLanguage':'de',
            'outputFormat':'json'
        })
        annotation = json.loads(annotation)
        return {"sentences":[
            {
                "text": "".join([token["originalText"] + " " if((i+1)<(len(sent["tokens"])) and sent["tokens"][i+1]["characterOffsetBegin"]>token["characterOffsetEnd"]) else token["originalText"] for i,token in enumerate(sent["tokens"])]),
                "dependencies": [
                    ["root" if(dep["dep"]=="ROOT") else dep["dep"], dep["governorGloss"]+"-"+str(dep["governor"]), dep["dependentGloss"]+"-"+str(dep["dependent"])] for dep in sent["basicDependencies"] if dep["dep"]!="punct"
                ],
                "words": [[token["originalText"],{
                    "CharacterOffsetBegin": str(token["characterOffsetBegin"]),
                    "CharacterOffsetEnd": str(token["characterOffsetEnd"]),
                    "Lemma": self.getAlignmentLemma(token["originalText"], token["pos"]),
                    "NamedEntityTag": token["ner"],
                    "PartOfSpeech": token["pos"]
                }] for token in sent["tokens"]]
            } for sent in annotation["sentences"]
        ]}

    def isInfoVocab(self,word):
        if(word in self.infoVocab): return self.infoVocab[word]
        if(word[0].isupper()): word = word.lower()
        if(word[0].islower()): word = word.upper()
        if(word in self.infoVocab): return self.infoVocab[word]
        return False

    def annotateText(self, text):
        text = " ".join(text.split())
        spacyAnn = self.spacyNLP(text)
        rfTaggerText = ""
        coreNlpText = ""
        tokens = []
        for sent in spacyAnn.sents:
            for spacyToken in sent:
                lemmaPos = spacyToken.pos_ if spacyToken.pos_ != "AUX" else "VERB"
                lemmaPos = self.checkPos(spacyToken.text, lemmaPos)

                mergedToken = {
                    "text": spacyToken.text,
                    "spacyPos": spacyToken.pos_,
                    "lemmaPos": [lemmaPos],
                    "id":spacyToken.i,
                    "headId":spacyToken.head.i,
                    "dep":spacyToken.dep_
                }
                tokens.append(mergedToken)
                rfTaggerText += spacyToken.text + "\n"
                coreNlpText += spacyToken.text + " "
            rfTaggerText += "\n"
            coreNlpText = coreNlpText[:-1] + "\n"
        rfTaggerText = rfTaggerText[:-2]
        coreNlpText = coreNlpText[:-2]

        # add coreNlp Annotation
        coreNlpAnn = self.stanfordCoreNLP.pos_tag(coreNlpText)
        for i,(word,pos) in enumerate(coreNlpAnn):
            token = tokens[i]
            lemmaPos = STTS_UD_MAP[pos]
            if(lemmaPos == "AUX"): lemmaPos = "VERB"
            lemmaPos = self.checkPos(word, lemmaPos)
            if(not(lemmaPos in token["lemmaPos"])):
                token["lemmaPos"].append(lemmaPos)
            else:
                token["udPos"] = lemmaPos
            token["coreNlpPos"] = pos

        # add rfTagger Annotation
        with open("lib/RFTagger/temp.txt", "w") as file:
            file.write(rfTaggerText)
        rfAnn = [tuple(word.split("\t")) for word in check_output(["src/rft-annotate", "-q", "lib/german.par", "temp.txt"], cwd="lib/RFTagger").decode("utf-8").split("\n") if word]
        for i,token in enumerate(tokens):
            try:
                pair = rfAnn[i]
            except:
                print([rf[0] for rf in rfAnn], len(rfAnn), [token["text"] for token in tokens], len(tokens))
            word = pair[0]
            if(word):
                pos = pair[1]
            else:
                continue
            posParts = pos.split(".")
            # add lemmaPos
            if isinstance(RFTAGGER_UD_MAP[posParts[0]], str):
                lemmaPos = RFTAGGER_UD_MAP[posParts[0]]
            elif(posParts[1] in RFTAGGER_UD_MAP[posParts[0]]):
                lemmaPos = RFTAGGER_UD_MAP[posParts[0]][posParts[1]]
            else: lemmaPos = "VERB"
            lemmaPos = self.checkPos(word, lemmaPos)
            if(not(lemmaPos) in token["lemmaPos"]):
                token["lemmaPos"].append(lemmaPos)
            else:
                token["udPos"] = lemmaPos
            # add pos - convert to STTS pos first
            if(posParts[0][0] == "V" and not(posParts[1] in RFTAGGER_STTS_MAP[posParts[0]])): posParts[1] = "Aux"
            if(posParts[0] in RFTAGGER_UD_MAP and isinstance(RFTAGGER_UD_MAP[posParts[0]], str)):
                pos = RFTAGGER_UD_MAP[posParts[0]]
            elif(posParts[1] in RFTAGGER_UD_MAP[posParts[0]] and isinstance(RFTAGGER_UD_MAP[posParts[0]][posParts[1]], str)):
                pos = RFTAGGER_UD_MAP[posParts[0]][posParts[1]]
            elif(posParts[2] in RFTAGGER_UD_MAP[posParts[0]][posParts[1]]):
                pos = RFTAGGER_UD_MAP[posParts[0]][posParts[1]][posParts[2]]
            else: pos = posParts[0]
            token["rfTaggerPos"] = pos
            if(posParts[0] in RF_POS_PROPS_MAP):
                if(posParts[0][0] == "V"):
                    token["form"] = posParts[0][1:]
                for i,prop in enumerate(RF_POS_PROPS_MAP[posParts[0]]):
                    token[prop] = posParts[i+1]

        # add lemmatization information
        for token in tokens:
            found = False
            token["lemmas"] = []
            word = token["text"]
            infoVocab = self.isInfoVocab(word)
            if(infoVocab):
                token["lemmas"] = [infoVocab]
            else:
                for p in reversed(token["lemmaPos"]):
                    wordLemmas, composita = self.getPossibleLemmas(word, p, token)
                    if(composita):
                        token["composita"] = composita
                    if(wordLemmas):
                        found = True
                        if(not("udPos") in token):
                            token["udPos"] = p
                        for lemma in wordLemmas:
                            if(not(lemma in token["lemmas"])):
                                token["lemmas"].append(lemma)
            if(not("udPos") in token):
                token["udPos"] = token["lemmaPos"][-1]
            if(token["text"][0].isupper() and not(token["udPos"] in ["NOUN", "PROPN", "X"] )):
                token["text"] = token["text"].lower()
            if(not(found) and not(word in token["lemmas"])):
                token["lemmas"].append(token["text"])
            posShort = ["n", "a", "v"]
            for idx, set in enumerate([["PROPN","NOUN"],["ADJ","ADV"],["VERB","AUX"]]):
                if(token["udPos"] in set):
                    token["slPos"] = SHORT_LEMMA_POS[idx]
                    break

        for token in tokens:
            token["head"] = tokens[token["headId"]]["lemmas"][0] if(tokens[token["headId"]]["id"] == token["headId"]) else print("Head Exception")

        return tokens
