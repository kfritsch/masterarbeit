additionalWords = ["Getter", "Setter", "getter", "setter", "protected", "schreibbar", "int", "Int", "double", "Double"]

def addWords(spellchecker):
    for word in additionalWords:
        spellchecker.add(word)
