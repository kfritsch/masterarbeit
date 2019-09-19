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
    this.questions = null;
    this.annotations = null;
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
        var { questions, annotations } = res;
        console.log(questions, annotations);
        this.questions = questions.sort((a, b) => parseInt(a.id) - parseInt(b.id));
        this.annotations = annotations;
        this.version = "whole";
        var qIdx = this.annotations.questions.length - 1;
        if (qIdx < 0) {
          qIdx = 0;
          this.annotations.questions.push({
            id: this.questions[qIdx].id,
            answersAnnotation: []
          });
        }
        if (this.questions.length > this.annotations.questions.length) {
          var annIdx = this.annotations.questions[qIdx].answersAnnotation.length;
          if (this.questions[qIdx].studentAnswers.length === annIdx) {
            qIdx += 1;
            annIdx = 0;
          }
        } else {
          qIdx = this.annotations.questions.findIndex(
            (question) =>
              !(
                "answerCategory" in
                question.answersAnnotation[question.answersAnnotation.length - 1]
              )
          );
          if (qIdx >= 0) {
            annIdx = this.annotations.questions[qIdx].answersAnnotation.findIndex(
              (annotation) => !("answerCategory" in annotation)
            );
          } else {
            this.version = "aspects";
            qIdx = this.annotations.questions.findIndex(
              (question) =>
                !("aspects" in question.answersAnnotation[question.answersAnnotation.length - 1])
            );
            if (qIdx < 0) {
              this.version = "done";
              return;
            }
            annIdx = this.annotations.questions[qIdx].answersAnnotation.findIndex(
              (annotation) => !("aspects" in annotation)
            );
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
    if (this.annotations.questions[qIdx].answersAnnotation.length === annIdx) {
      this.annotations.questions[qIdx].answersAnnotation.push(answerAnnotation);
    } else {
      this.annotations.questions[qIdx].answersAnnotation[annIdx] = answerAnnotation;
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
        annotations: this.annotations
      })
    });
    const body = await response.text();
    return body;
  };

  getCurrentAnnotation(qIdx, annIdx) {
    var activeAnswerAnnotation = this.annotations.questions[qIdx].answersAnnotation[annIdx];
    var activeAnswer = this.questions[qIdx].studentAnswers[annIdx];
    var emptyAspectSchemaTwo = {
      text: "",
      elements: [],
      referenceAspects: []
    };
    if (!activeAnswerAnnotation) {
      activeAnswerAnnotation = {
        text: activeAnswer.text.replace(/\s\s+/g, " "),
        id: activeAnswer.id,
        correctionOrComment: activeAnswer.text.replace(/\s\s+/g, " ")
      };
    } else {
      if (this.version == "aspects") {
        if ("aspects" in activeAnswerAnnotation) {
          activeAnswerAnnotation.aspects.push(emptyAspectSchemaTwo);
        } else {
          activeAnswerAnnotation["aspects"] = [emptyAspectSchemaTwo];
        }
      }
      if (!("correctionOrComment" in activeAnswerAnnotation)) {
        activeAnswerAnnotation["correctionOrComment"] = activeAnswer.text.replace(/\s\s+/g, " ");
      }
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
    if (qIdx === this.annotations.questions.length) {
      this.annotations.questions.push({
        id: this.questions[qIdx].id,
        answersAnnotation: []
      });
    }
    var annIdx = 0;
    this.setState({
      qIdx,
      annIdx,
      currentAnnotation: this.getCurrentAnnotation(qIdx, annIdx),
      error: ""
    });
  }

  prevQuestion() {
    var qIdx = this.state.qIdx - 1;
    var annIdx = this.annotations.questions[qIdx].answersAnnotation.length - 1;
    this.setState({
      qIdx,
      annIdx,
      currentAnnotation: this.getCurrentAnnotation(qIdx, annIdx),
      error: ""
    });
  }

  prevAnswer() {
    var annIdx = this.state.annIdx - 1;
    this.setState({
      annIdx,
      currentAnnotation: this.getCurrentAnnotation(this.state.qIdx, annIdx),
      error: ""
    });
  }

  render() {
    const { qIdx, annIdx, currentAnnotation } = this.state;
    var lastAnnotation = false;
    var lastQuestion = false;
    if (this.questions) {
      lastAnnotation = annIdx === this.questions[qIdx].studentAnswers.length - 1;
      lastQuestion = qIdx === this.questions.length - 1;
    }
    return (
      <div className="app-container">
        <div className="header-container">
          <Header className="header-text text-area" textAlign="center" as="h1">
            Question-Annotation
          </Header>
        </div>
        {!this.questions ? (
          <div>Missing Questions</div>
        ) : this.version == "done" ? (
          <div>Done</div>
        ) : (
          <Segment.Group className="annotation-container">
            <QuestionAnnotation
              activeQuestion={this.questions[qIdx]}
              questionCount={this.questions.length}
              qIdx={qIdx}
              getColors={this.getColors}
              getDepTree={this.getDepTree}
            />
            {this.version == "whole" ? (
              <RoughLabeling
                ref={(instance) => {
                  this.answerAnnotation = instance;
                }}
                activeQuestion={this.questions[qIdx]}
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
                activeQuestion={this.questions[qIdx]}
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
