import React from "react";
import QuestionAnnotation from "./QuestionAnnotation";
//import AnswerAnnotation from "./schema-one/AnswerAnnotation";
// import AnswerAnnotationSchemaTwo from "./schema-two/AnswerAnnotationSchemaTwo";
import AspectAnnotation from "./schema-two/AspectAnnotation";
import RoughLabeling from "./schema-two/RoughLabeling";

import { Segment, Header, Button } from "semantic-ui-react";

export default class AnnotationSurvey extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      qIdx: 0,
      annIdx: 0,
      currentAnnotation: null
    };
    this.questionData = null;
    this.answerAnnotation = null;
  }

  getColors(question) {
    var colors = [];
    var r = 105;
    var g = 0;
    var b = 0;
    var step = 300 / (question.referenceAnswer.aspects.length - 1);
    for (var i = 0; i < question.referenceAnswer.aspects.length; i++) {
      g = 255;
      b = 105 + i * step;
      if (b >= 255) {
        g -= b - 255;
        b = 255;
      }
      colors.push("#" + r.toString(16) + g.toString(16) + b.toString(16));
    }
    colors.push("#3c3c40");
    return colors;
  }

  componentDidMount() {
    this.callApi()
      .then((res) => {
        var { questionData } = res;
        console.log(questionData);
        this.questionData = questionData["questions"];
        this.version = "whole";
        var annIdx;
        var qIdx = this.questionData.findIndex(
          (question) =>
            !("answerCategory" in question.answersAnnotation[question.answersAnnotation.length - 1])
        );
        if (qIdx >= 0) {
          annIdx = this.questionData[qIdx].answersAnnotation.findIndex(
            (annotation) => !("answerCategory" in annotation)
          );
        } else {
          this.version = "aspects";
          qIdx = this.questionData.findIndex(
            (question) =>
              !("aspects" in question.answersAnnotation[question.answersAnnotation.length - 1])
          );
          if (qIdx < 0) {
            this.version = "done";
            return;
          }
          annIdx = this.questionData[qIdx].answersAnnotation.findIndex(
            (annotation) => !("aspects" in annotation)
          );
          if (annIdx < 0) {
            this.finish();
            annIdx = this.questionData[qIdx].answersAnnotation.length - 1;
          }
        }

        var currentAnnotation = this.getCurrentAnnotation(qIdx, annIdx);
        this.setState({
          qIdx,
          annIdx,
          currentAnnotation
        });
      })
      .catch((err) => console.log(err));
  }

  callApi = async () => {
    const response = await fetch("http://localhost:8080/api/get-annotation-data");
    const body = await response.json();
    if (response.status !== 200) throw Error(body.message);
    return body;
  };

  updateAnnotations() {
    const { qIdx, annIdx } = this.state;
    var answerAnnotation = this.answerAnnotation.getAnswerAnnotation();
    if (!answerAnnotation) {
      return false;
    }
    if (this.questionData[qIdx].answersAnnotation.length === annIdx) {
      this.questionData[qIdx].answersAnnotation.push(answerAnnotation);
    } else {
      this.questionData[qIdx].answersAnnotation[annIdx] = answerAnnotation;
    }
    this.saveAnnotations();
    return true;
  }

  getDepTree = async (text) => {
    const response = await fetch("http://localhost:5000/dep-tree", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        text: text
      })
    });
    const body = await response.json();
    return body;
  };

  saveAnnotations = async () => {
    const response = await fetch("http://localhost:8080/api/save-annotations", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        annotations: { questions: this.questionData }
      })
    });
    const body = await response.text();
    return body;
  };

  getCurrentAnnotation(qIdx, annIdx) {
    var activeAnswerAnnotation = this.questionData[qIdx].answersAnnotation[annIdx];
    var emptyAspectSchemaTwo = {
      text: "",
      elements: [],
      referenceAspects: []
    };
    if (this.version === "aspects") {
      if ("aspects" in activeAnswerAnnotation) {
        activeAnswerAnnotation.aspects.push(emptyAspectSchemaTwo);
      } else {
        activeAnswerAnnotation["aspects"] = [emptyAspectSchemaTwo];
      }
    }
    if (!("correctionOrComment" in activeAnswerAnnotation)) {
      activeAnswerAnnotation["correctionOrComment"] = activeAnswerAnnotation.text.replace(
        /\s\s+/g,
        " "
      );
    }
    return activeAnswerAnnotation;
  }

  nextAnswer() {
    var success = this.updateAnnotations();
    if (!success) {
      this.setState({ error: "Please annotate first!" });
      return;
    }
    var annIdx = this.state.annIdx + 1;
    this.setState({
      annIdx,
      currentAnnotation: this.getCurrentAnnotation(this.state.qIdx, annIdx),
      error: ""
    });
  }

  nextQuestion() {
    var success = this.updateAnnotations();
    if (!success) {
      this.setState({ error: "Please annotate first!" });
      return;
    }
    var qIdx = this.state.qIdx + 1;
    var annIdx = 0;
    this.setState({
      qIdx,
      annIdx,
      currentAnnotation: this.getCurrentAnnotation(qIdx, annIdx),
      error: ""
    });
  }

  prevQuestion() {
    var success = this.updateAnnotations();
    if (!success) {
      this.setState({ error: "Please annotate first!" });
      return;
    }
    var qIdx = this.state.qIdx - 1;
    var annIdx = this.questionData[qIdx].answersAnnotation.length - 1;
    this.setState({
      qIdx,
      annIdx,
      currentAnnotation: this.getCurrentAnnotation(qIdx, annIdx),
      error: ""
    });
  }

  prevAnswer() {
    var success = this.updateAnnotations();
    if (!success) {
      this.setState({ error: "Please annotate first!" });
      return;
    }
    var annIdx = this.state.annIdx - 1;
    this.setState({
      annIdx,
      currentAnnotation: this.getCurrentAnnotation(this.state.qIdx, annIdx),
      error: ""
    });
  }

  finish() {
    alert("You are Done!");
  }

  render() {
    const { qIdx, annIdx, currentAnnotation } = this.state;
    var lastAnnotation = false;
    var lastQuestion = false;
    if (this.questionData) {
      lastAnnotation = annIdx === this.questionData[qIdx].answersAnnotation.length - 1;
      lastQuestion = qIdx === this.questionData.length - 1;
    }
    return (
      <div className="app-container">
        <div className="header-container">
          <Header className="header-text text-area" textAlign="center" as="h1">
            Question-Annotation
          </Header>
        </div>
        {!this.questionData ? (
          <div>Missing Questions</div>
        ) : this.version === "done" ? (
          <div>Done</div>
        ) : (
          <Segment.Group className="annotation-container">
            <QuestionAnnotation
              activeQuestion={this.questionData[qIdx]}
              questionCount={this.questionData.length}
              qIdx={qIdx}
              getColors={this.getColors}
              getDepTree={this.getDepTree}
            />
            {this.version === "whole" ? (
              <RoughLabeling
                ref={(instance) => {
                  this.answerAnnotation = instance;
                }}
                activeQuestion={this.questionData[qIdx]}
                currentAnnotation={currentAnnotation}
                annIdx={annIdx}
                goldStandard={false}
              />
            ) : (
              <AspectAnnotation
                ref={(instance) => {
                  this.answerAnnotation = instance;
                }}
                getColors={this.getColors}
                getDepTree={this.getDepTree}
                activeQuestion={this.questionData[qIdx]}
                currentAnnotation={currentAnnotation}
                annIdx={annIdx}
              />
            )}
            {this.state.error && <p style={{ color: "red" }}>{this.state.error}</p>}
            {annIdx === 0 ? (
              <Button
                onClick={this.prevQuestion.bind(this)}
                disabled={qIdx === 0}
                style={{ backgroundColor: "#66AB8C", color: "#F2EEE2" }}
                content="Vorherige Frage"
              />
            ) : (
              <Button
                onClick={this.prevAnswer.bind(this)}
                style={{ backgroundColor: "#66AB8C", color: "#F2EEE2" }}
                content="Vorherige Antwort"
              />
            )}
            {lastQuestion && lastAnnotation ? (
              <Button
                onClick={this.finish.bind(this)}
                style={{ backgroundColor: "#66AB8C", color: "#F2EEE2" }}
                content="Finish"
              />
            ) : lastAnnotation ? (
              <Button
                onClick={this.nextQuestion.bind(this)}
                style={{ backgroundColor: "#66AB8C", color: "#F2EEE2" }}
                content="Nächste Frage"
              />
            ) : (
              <Button
                onClick={this.nextAnswer.bind(this)}
                style={{ backgroundColor: "#66AB8C", color: "#F2EEE2" }}
                content="Nächste Anwort"
              />
            )}
          </Segment.Group>
        )}
      </div>
    );
  }
}
