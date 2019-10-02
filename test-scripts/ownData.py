import numpy as np
import pandas as pd
import re, os, sys, json, csv, math, codecs
from os.path import join, dirname, realpath, isdir, isfile, exists
FILE_PATH = dirname(realpath(__file__))
DATA_PATH = join(FILE_PATH, "..", "question-corpora")

umlauts = {
    '\\u00e4': "ä",
    '\\u00f6': "ö",
    '\\u00fc': "ü",
    '\\u00c4': "Ä",
    '\\u00d6': "Ö",
    '\\u00dc': "Ü",
    '\\u00df': "ß",
    '\\r\\n': " ",
    '\\"' : ''
}

a = "a\rv"
print(a)

jsonData = {"questions": []}
data = pd.read_csv(join(DATA_PATH,"vips_solution_info_a_fritsch.csv"))
for name, group in data.groupby("exercise_id"):
    firstEntry = group.iloc[0]
    question = {"text": firstEntry["description"], "id": name, "title": firstEntry["title"], "referenceAnswer": json.loads(firstEntry["task_json"])["answers"][0]["text"], "answersAnnotation":[] }
    for row in group[["response","SHA2(CONCAT(vs.user_id+'secret'), 224)"]].itertuples():
        answer = row[1][2:-2]
        if(len(answer)==0): continue
        for key in umlauts.keys():
            answer = answer.replace(key, umlauts[key])
        question["answersAnnotation"].append({"text": answer, "id": row[2]})
    jsonData["questions"].append(question)
with open(join(DATA_PATH,"vips_solution_info_a_fritsch.json"), "w+") as f:
    json.dump(jsonData, f, indent=4)
# print(jsonData)

with open(join(DATA_PATH,"vips_solution_info_a_fritsch.json"), "r") as f:
    a = json.load(f)
    print(a)
