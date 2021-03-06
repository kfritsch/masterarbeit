const express = require("express");
const bodyParser = require("body-parser");
const fs = require("fs");
const path = require("path");

const PORT = process.env.PORT || 8080;
const HOST = "0.0.0.0";
const BASE = process.cwd();

const CLIENT_BUILD_PATH = path.join(__dirname, "../../client/build");
const ANNOTATION_FILE = "VIPS.json";

// App
const app = express();

// Static files
app.use(express.static(CLIENT_BUILD_PATH));

app.use(bodyParser.json({ limit: "500mb", extended: true }));
app.use(bodyParser.urlencoded({ limit: "10mb", extended: true }));

app.use(function(req, res, next) {
  res.header("Access-Control-Allow-Origin", "*");
  res.header("Access-Control-Allow-Headers", "Origin, X-Requested-With, Content-Type, Accept");
  next();
});

app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

const GOLD_STANDARD = JSON.parse(
  fs.readFileSync(BASE + "/data/annotations/" + ANNOTATION_FILE, "utf8")
);

function updateGoldStandardAnnotation(questionId, annotation) {
  try {
    var filename = BASE + "/data/annotations/" + ANNOTATION_FILE;
    var goldStandard = JSON.parse(fs.readFileSync(filename, "utf8"));
    var questionIdx = goldStandard.questions.findIndex((question) => {
      return question.id == questionId;
    });
    if (questionIdx >= 0) {
      var question = goldStandard.questions[questionIdx];
      var annotationIdx = question.answersAnnotation.findIndex((existingAnnotation) => {
        return existingAnnotation.id == annotation.id;
      });
      if (annotationIdx >= 0) {
        question.answersAnnotation[annotationIdx] = annotation;
      } else {
        question.answersAnnotation.push(annotation);
      }
    } else {
      goldStandard.questions.push({ id: questionId, annotations: [annotation] });
    }
    fs.writeFileSync(filename, JSON.stringify(goldStandard, null, 2));
    return true;
  } catch (err) {
    console.log(err);
    return false;
  }
}

function saveGoldStandardAnnotation(annotations) {
  try {
    var filename = BASE + "/data/annotations/" + ANNOTATION_FILE;
    fs.writeFileSync(filename, JSON.stringify(annotations, null, 2));
    return true;
  } catch (err) {
    console.log(err);
    return false;
  }
}

// API
app.get("/api/get-annotation-data", (req, res, next) => {
  res.send({ questionData: GOLD_STANDARD });
});
app.post("/api/save-annotations", (req, res, next) => {
  // var annotations = JSON.parse(req.body.annotations);
  var annotations = req.body.annotations;
  console.log(annotations);
  var success = saveGoldStandardAnnotation(annotations);
  res.send(success.toString());
});

// All remaining requests return the React app, so it can handle routing.
app.get("*", function(request, response, next) {
  response.sendFile(path.join(CLIENT_BUILD_PATH, "index.html"));
});
app.listen(PORT, HOST, () => console.log(`Running on http://${HOST}:${PORT}`));
