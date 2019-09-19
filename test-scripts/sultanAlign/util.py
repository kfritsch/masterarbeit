punctuations = ['(','-lrb-','.',',','-','?','!',';','_',':','{','}','[','/', ']','...','"','``','`',"\'",')', '-rrb-', "''", '\\', '/', '\\/', '..', '....', '--', '---', '----', '?!', '!?', '??', '???', '????', '!!', '!!!', '!!!!', "\\*", '|', '-lcb-', '-rcb-', '-lsb-', '-rsb-', '>>', '<<', '+', '\\+', '*', '\\*', '^', '\\^', '#']
stop_words = ["als","am","an","auf","aus","bei","bis","da","dadurch","das","dass","dein","deine","dem","den","der","des","dessen","die","dies","dieser","dieses","doch","dort","du","durch","ein","eine","einem","einen","einer","eines","er","es","euer","eure","für","ich","ihr","ihre","im","in","jener","jenes","jetzt","man","mein","meine","mit","nach","nachdem","nein","nicht","nun","seine","sich","sie","sonst","soweit","sowie","und","unser","unsere","vom","von","vor","was","weiter","weitere","wenn","wer","wie","wieder","wir","zu","zum","zur","über"]

stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

def isSublist(A, B):
    sub = True
    for item in A:
        if item not in B:
            sub = False
            break
    return sub

def findAllCommonContiguousSublists(A, B, turnToLowerCases=True):
    a = []
    b = []
    for item in A:
        a.append(item)
    for item in B:
        b.append(item)

    if turnToLowerCases:
        for i in range(len(a)):
            a[i] = a[i].lower()
        for i in range(len(b)):
            b[i] = b[i].lower()
    commonContiguousSublists = []
    swapped = False
    if len(a) > len(b):
        temp = a
        a = b
        b = temp
        swapped = True
    maxSize = len(a)
    for size in range(maxSize, 0, -1):
        startingIndicesForA = [item for item in range(0, len(a)-size+1)]
        startingIndicesForB = [item for item in range(0, len(b)-size+1)]
        for i in startingIndicesForA:
            for j in startingIndicesForB:
                if a[i:i+size] == b[j:j+size]:
                    # check if a contiguous superset has already been inserted; don't insert this one in that case
                    alreadyInserted = False
                    currentAIndices = [item for item in range(i,i+size)]
                    currentBIndices = [item for item in range(j,j+size)]
                    for item in commonContiguousSublists:
                        if isSublist(currentAIndices, item[0]) and isSublist(currentBIndices, item[1]):
                            alreadyInserted = True
                            break
                    if not alreadyInserted:
                        commonContiguousSublists.append([currentAIndices, currentBIndices])

    if swapped:
        for item in commonContiguousSublists:
            temp = item[0]
            item[0] = item[1]
            item[1] = temp

    return commonContiguousSublists

# return the lemmas in the span [wordIndex-leftSpan, wordIndex+rightSpan] and the positions actually available, left and right
def findTextualNeighborhood(sentenceDetails, wordIndex, leftSpan, rightSpan):

    global punctuations

    sentenceLength = len(sentenceDetails)

    startWordIndex = max(1, wordIndex-leftSpan)
    endWordIndex = min(sentenceLength, wordIndex+rightSpan)

    lemmas = []
    wordIndices = []
    for item in sentenceDetails[startWordIndex-1:wordIndex-1]:
        if item[3] not in stop_words + punctuations:
            lemmas.append(item[3])
            wordIndices.append(item[1])
    for item in sentenceDetails[wordIndex:endWordIndex]:
        if item[3] not in stop_words + punctuations:
            lemmas.append(item[3])
            wordIndices.append(item[1])
    return [wordIndices, lemmas, wordIndex-startWordIndex, endWordIndex-wordIndex]

def isAcronym(word, namedEntity):
    canonicalWord = word.replace('.', '')
    if not canonicalWord.isupper() or len(canonicalWord) != len(namedEntity) or canonicalWord.lower() in ['a', 'i']:
        return False

    acronym = True
    for i in range(len(canonicalWord)):
        if canonicalWord[i] != namedEntity[i][0]:
            acronym = False
            break

    return acronym
