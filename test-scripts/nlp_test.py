def hunspellTest():
    import hunspell
    spellchecker = hunspell.HunSpell('/usr/share/hunspell/en_US.dic', '/usr/share/hunspell/en_US.aff')
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
    doc = nlp(sentence)

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

def germanetTest():
    from germanet import load_germanet, Synset
    GERMANET = load_germanet(host = "localhost", port = 27027, database_name = 'germanet')
    synsets = GERMANET.synsets("stark")
    for s in synsets:
        print(s)
        print(s.gn_class)
        for l in s.lemmas:
            print(l)
            print(l.category)
            for r in l.rels():
                print(r)
        input()
        for r in s.rels():
            print(r)
        input()

germanetTest()
