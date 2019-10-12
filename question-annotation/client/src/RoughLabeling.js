import React from "react";

import { Message, Dropdown, Segment, TextArea } from "semantic-ui-react";

const ROUGH_CATEGORIES = [
  "correct",
  "binary_correct",
  "partially_correct",
  "missconception",
  "concept_mix-up",
  "irrelevant",
  "none"
];

export default class RoughLabeling extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      correctionOrComment: props.currentAnnotation.correctionOrComment,
      answerCategory: props.currentAnnotation.answerCategory
    };
  }

  componentDidUpdate(prevProps) {
    if (
      this.props.activeQuestion.id !== prevProps.activeQuestion.id ||
      this.props.annIdx !== prevProps.annIdx
    ) {
      this.setState({
        correctionOrComment: this.props.currentAnnotation.correctionOrComment,
        answerCategory: this.props.currentAnnotation.answerCategory
      });
    }
  }

  getAnswerAnnotation() {
    const { correctionOrComment, answerCategory } = this.state;
    if (!answerCategory) return false;
    var currentAnnotation = JSON.parse(JSON.stringify(this.props.currentAnnotation));
    currentAnnotation["answerCategory"] = answerCategory;
    currentAnnotation["correctionOrComment"] = correctionOrComment;
    return currentAnnotation;
  }

  render() {
    const { correctionOrComment, answerCategory } = this.state;
    const { annIdx, activeQuestion, goldStandard } = this.props;
    return (
      <div>
        <Segment.Group className="student-answer-container">
          <Segment textAlign="center">
            <Message style={{ backgroundColor: "#eff0f6" }}>
              <Message.Header style={{ paddingBottom: "0.5em" }}>
                {"Schülerantwort " + (annIdx + 1) + "/" + activeQuestion.answersAnnotation.length}
              </Message.Header>
              <span key={"whole"}>{correctionOrComment}</span>
            </Message>
          </Segment>
          {goldStandard && (
            <Segment textAlign="center">
              <TextArea
                rows={3}
                style={{ width: "80%" }}
                autoHeight
                placeholder={goldStandard ? "Korrigierter Text " : "Kommentar"}
                onChange={(e) => this.setState({ correctionOrComment: e.target.value })}
                value={correctionOrComment}
              />
            </Segment>
          )}
          <Segment textAlign="center">
            <Dropdown
              value={answerCategory}
              placeholder="Answer Category"
              options={ROUGH_CATEGORIES.map((cat) => {
                return {
                  value: cat,
                  text: cat.replace("_", " "),
                  key: cat
                };
              })}
              button
              onChange={(e, obj) => this.setState({ answerCategory: obj.value })}
              style={{ gridArea: "box2", textAlign: "center", backgroundColor: "#a5696942" }}
            />
          </Segment>
        </Segment.Group>
      </div>
    );
  }
}
