def hunspellTest():
    import hunspell
    spellchecker = hunspell.HunSpell('/usr/share/hunspell/de_DE.dic', '/usr/share/hunspell/de_DE.aff')
    print(spellchecker.spell("?"))
    spellchecker.add("Getter")
    print(spellchecker.spell("Getter-")) # True
    suggestions = spellchecker.suggest("Private")
    print(suggestions)

def hyphenationTest(sentence):
    import pyphen
    from nltk.tokenize import sent_tokenize, word_tokenize
    for key in pyphen.LANGUAGES.keys():
        if(key.split("_")[0] == "de"):
            print(key)
    dic = pyphen.Pyphen(lang='de_DE')
    for word in word_tokenize(sentence, language='german'):
        print(dic.inserted(word))

def rfTaggerTest(text):
    # run this once
    # import nltk
    # nltk.download('punkt')
    # from subprocess import run
    # run(["make"], cwd="lib/RFTagger/src")

    from subprocess import check_output
    from nltk.tokenize import sent_tokenize, word_tokenize

    with open("lib/RFTagger/temp.txt", "w") as file:
        file.write("\n\n".join("\n".join(word_tokenize(sentence, language='german')) for sentence in sent_tokenize(text, language='german')))
    test_tagged = check_output(["src/rft-annotate", "lib/german.par", "temp.txt"], cwd="lib/RFTagger").decode("utf-8").split("\n")
    for res in test_tagged:
        resList = res.split()
        if(resList):
            print(resList)

def nltkTest(sentence):
    import nltk
    from nltk import grammar, parse
    cp = parse.load_parser('lib/nltk_data/grammars/book_grammars/german.fcfg', trace=0)
    sent = 'der Hund folgt der Katze'
    tokens = sent.split()
    trees = cp.parse(tokens)
    for tree in trees: print(tree)

def coreNLPTest(sentence):
    from stanfordcorenlp import StanfordCoreNLP
    nlp = StanfordCoreNLP(r'./lib/stanford-corenlp-full-2018-02-27', lang="de")
    # print('Tokenize: ')
    # print(nlp.word_tokenize(sentence))
    print('Part of Speech: ')
    print(STOP_WORDS)
    for word,pos in nlp.pos_tag(sentence):
        if(word.lower() in STOP_WORDS):
            print("stopword: ", word)
            continue
        udPOS = STTS_UD_MAP[pos]

        lemmaIWNLP = IWNLPlemmatizer.get_lemmas(word, IWNLPLEMMA_MAP[udPOS], ignore_case=False) if udPOS in IWNLPLEMMA_MAP else word
        lemmaGERMA = Germalemmatizer.find_lemma(word, GERMALEMMA_MAP[udPOS]) if udPOS in GERMALEMMA_MAP else word
        print(word,pos,lemmaIWNLP,lemmaGERMA)
    # print('Constituency Parsing: ')
    # print(nlp.parse(sentence))
    # print('Dependency Parsing: ')
    # print(nlp.dependency_parse(sentence))

def spacyTest(sentence):
    import webbrowser, os
    import spacy
    import pandas as pd
    from spacy_iwnlp import spaCyIWNLP
    nlp = spacy.load('de')
    iwnlp = spaCyIWNLP(lemmatizer_path='.lib/IWNLP.Lemmatizer_20170501.json')
    nlp.add_pipe(iwnlp)
    # doc = nlp('Datentypen dienen zur unterschiedlichen semantischen Interpretation von binären Werten')
    doc = nlp(sentence)
    # plot_parse_tree(doc)

    # token = doc[0]
    # for key in dir(token):
    #     print(key)
    # print(token.text)
    # print(token.prob)
    # print(token.prefix_)
    # print(token.suffix_)
    # print(token.ent_type_)
    # print(token.head)
    # print(doc.print_tree())
    #
    # for span in list(doc.noun_chunks):
    #     print(span.text, span.label_, span.label_)

    # span = list(doc.noun_chunks)[0]
    # for key in dir(span):
    #     print(key)
    # print(span.text)

    nlptags = ['LEMMA', 'POS', 'TAG', 'DEP']
    words = [token.text for token in doc]
    results = [[token._.iwnlp_lemmas, token.pos_, token.tag_, token.dep_] for token in doc]

    df = pd.DataFrame(results, index=words, columns=nlptags)
    html = df.to_html()
    filename = "spacy_dep_parse.html"
    with open(filename, 'w') as f:
        f.write("<!DOCTYPE html>\n")
        f.write("<html>\n")
        f.write("<head>\n")
        f.write("<title>Spacy Dependency Parse</title>\n")
        f.write("</head>\n")
        f.write("<body>\n")
        f.write(html)
        f.write("\n")
        f.write("</body>\n")
        f.write("</html>\n")
    webbrowser.open('file://' + os.path.realpath(filename))

#sentence = "Der bronzene Ring würde nicht am Nagel hängen bleiben, weil der Ring nicht eisern ist."
# sentence = "Der Mann, der mich liebte, ging zum Meer."
# sentence = "Er hat gelogen."
# sentence = "Die bösen Männer waren schneller als die Jungen."
# sentence = "Ich sollte gehen. Ich bin dumm. Ich bin gelaufen. Ich gehe."
# sentence = 'Das sind Methoden einer Klasse mit denselben Namen.'
# sentence = 'Der Mann geht in den Park.'
#
# #cleanLemmaVector(sentence)
# #stanfordCoreNLP.close()
# coreNLPTest(sentence)
# # nltkTest(sentence)
# # rfTaggerTest(sentence)
# # spacyTest(sentence)
# # hyphenationTest(sentence)
# from coreNlp import StanfordCoreNLP
# stanfordCoreNLP = StanfordCoreNLP('http://localhost', port=9000)
# pos = [x[0] for x in stanfordCoreNLP.pos_tag(sentence)]
# res = stanfordCoreNLP.dependency_parse(sentence)
# for tuple in res:
#     print(tuple, pos[tuple[2]])

# from stanfordcorenlp import StanfordCoreNLP
text="Hallo Welt! Angela Merkel ist in Berlin."
props={
    'annotators': 'tokenize, ssplit, pos, lemma, parse, depparse',
    'pipelineLanguage':'de',
    'outputFormat':'json'
}

# from coreNlp import StanfordCoreNLP
# nlp = StanfordCoreNLP(r'./lib/stanford-corenlp-full-2018-02-27', lang="en")
# print(nlp.annotate(text, properties=props))
# nlp.close()

# from coreNlp import StanfordCoreNLP
# nlp=StanfordCoreNLP('http://localhost', port=9000)
# print(nlp.annotate(text, properties=props))

from TokenAnnotator import TokenAnnotator
tokenAnnotator = TokenAnnotator()
print(tokenAnnotator.getAlignmentAnnotation(text))
