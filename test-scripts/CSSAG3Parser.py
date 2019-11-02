import re, sys, os, codecs, json
import xml.etree.ElementTree as ET

class CSSAG3Parser(object):
    ILLIGAL_UNICHRS = [ (0x00, 0x08), (0x0B, 0x1F), (0x7F, 0x84), (0x86, 0x9F),
                    (0xD800, 0xDFFF), (0xFDD0, 0xFDDF), (0xFFFE, 0xFFFF),
                    (0x1FFFE, 0x1FFFF), (0x2FFFE, 0x2FFFF), (0x3FFFE, 0x3FFFF),
                    (0x4FFFE, 0x4FFFF), (0x5FFFE, 0x5FFFF), (0x6FFFE, 0x6FFFF),
                    (0x7FFFE, 0x7FFFF), (0x8FFFE, 0x8FFFF), (0x9FFFE, 0x9FFFF),
                    (0xAFFFE, 0xAFFFF), (0xBFFFE, 0xBFFFF), (0xCFFFE, 0xCFFFF),
                    (0xDFFFE, 0xDFFFF), (0xEFFFE, 0xEFFFF), (0xFFFFE, 0xFFFFF),
                    (0x10FFFE, 0x10FFFF) ]
    ILLEGAL_RANGES = ["%s-%s" % (chr(low), chr(high))
                  for (low, high) in ILLIGAL_UNICHRS
                  if low < sys.maxunicode]
    ILLEGAL_XML_RE = re.compile(u'[%s]' % u''.join(ILLEGAL_RANGES))

    def removeInvalidXmlChar(self, content):
        content = list(content)
        for i in range(len(content)):
            content[i] = ' ' if(ILLEGAL_XML_RE.search(c) is not None) else c
        content = "".join(content)
        return content

    def removeInvalidXmlForm(self, content):
        # split text into lines
        lines = content.split("\n")
        # remove invalid lines until no err
        error = True
        while(error):
            try:
                text = "\n".join(lines)
                tree = ET.fromstring(text)
                error = False
            except ET.ParseError as e:
                eParts = str(e).split(":")
                if(eParts[0]=="mismatched tag"):
                    line,column = eParts[1].split(",")
                    line = int(line.split(" line ")[1])
                    column = int(column.split(" column ")[1])
                    del lines[line-1]
                else:
                    error = False

        content = "\n".join(lines)
        return content

    def getXMLTree(self, xmlPath):
        tree = ET.parse(xmlPath)
        return tree

    def getXMLRootFromString(self, xmlPath):
        content = None
        tree = None
        with codecs.open(xmlPath, "r", "utf-8") as f:
            content = f.read()
        try:
            tree = ET.fromstring(content)
        except ET.ParseError as e:
            eParts = str(e).split(":")
            # remove invalid xml tokens
            if(eParts[0]=="not well-formed (invalid token)"):
                content = self.removeInvalidXmlChar(content)
            # remove not matching tags
            elif(eParts[0]=="mismatched tag"):
                content = self.removeInvalidXmlForm(content)
            else:
                print("Unknown XML ParseError!")
                print(e)
                sys.exit(1)
            try:
                tree = ET.fromstring(content)
            except ET.ParseError:
                print("XML-ERROR in: " + doc_id)
                sys.exit(1)
        return tree

    def parseXml(self, xmlPath):
        tree = self.getXMLTree(xmlPath)
        root = tree.getroot()
        # go through all pages
        questionDict = {}
        for root_obj in root:
            if(root_obj.tag!="studentAnswers"):
                questionDict[root_obj.tag] = root_obj.text
            else:
                questionDict["studentAnswers"] = []
                for answer in root_obj:
                    answerDict = {}
                    for answerItem in answer:
                        answerDict[answerItem.tag] = answerItem.text
                    questionDict["studentAnswers"].append(answerDict)
        return  questionDict

    def convertToJson(self, xmlPath, jsonPath):
        tree = self.getXMLTree(xmlPath)
        root = tree.getroot()
        questionJSObject = {}
        questionJSObject["studentAnswers"] = [];
        questionJSObject["id"] = root.attrib["id"]
        for root_obj in root:
            if(root_obj.tag in ["text", "type", "title"]):
                questionJSObject[root_obj.tag] = root_obj.text
            if(root_obj.tag=="referenceAnswers"):
                for referenceAnswer in root_obj:
                    if(referenceAnswer.tag == "referenceAnswer"):
                        questionJSObject["referenceAnswer"] = referenceAnswer.find("text").text
                        break
            if(root_obj.tag=="studentAnswers"):
                for answer in root_obj:
                    answerJSObject = {}
                    answerJSObject["text"] = answer.find("text").text
                    answerJSObject["id"] = answer.attrib["id"]
                    questionJSObject["studentAnswers"].append(answerJSObject)
        with open(jsonPath, 'w') as outfile:
            json.dump(questionJSObject, outfile, sort_keys=True, indent=4)


    def modifyXml(self, xmlPath, newXMLPath):
        tree = self.getXMLTree(xmlPath)
        root = tree.getroot()
        for root_obj in root:
            if(root_obj.tag=="studentAnswers"):
                for answer in root_obj:
                    for answerItem in answer:
                        if(answerItem.tag=="studentID"):
                            answer.remove(answerItem)
                    for answerItem in answer:
                        if(answerItem.tag=="answerID"):
                            answer.set('id',answerItem.text)
                            answer.remove(answerItem)
        tree.write(newXMLPath, encoding="UTF-8", xml_declaration=True)

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
    print(questionDict)
    return  questionDict

parseSemval(os.path.join("..","question-corpora", "SEMVAL", "training", "sciEntsBank", "EM-inv1-45b.xml"))

xmlParser = CSSAG3Parser()
for id in ["25","26","29","31"]:
    xmlParser.convertToJson("Question"+id+".xml","Question"+id+".json")
# questionDict = xmlParser.parseXml("./CSSAG 3.0/CSSAG_XML/Question1.xml")
# print(len(questionDict["studentAnswers"]))
