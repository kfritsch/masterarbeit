import React from "react";

import { Message, Label, Segment, Button } from "semantic-ui-react";
import AnswerMarkup from "./AnswerMarkup";
import Tree from "react-d3-tree";

export default class QuestionAnnotation extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      colors: props.getColors(props.activeQuestion),
      questionTree: null,
      referenceAnswerTree: null,
      questionToggle: false,
      answerToggle: false
    };
  }

  componentDidMount() {
    this.props.getDepTree(this.props.activeQuestion.text).then((res) => {
      this.setState({ questionTree: res.tree });
    });
    this.props.getDepTree(this.props.activeQuestion.referenceAnswer.text).then((res) => {
      this.setState({ referenceAnswerTree: res.tree });
    });
  }

  componentDidUpdate(prevProps) {
    if (prevProps.questionIdx !== this.props.questionIdx) {
      this.setState({
        colors: this.props.getColors(this.props.activeQuestion)
      });
      this.props.getDepTree(this.props.activeQuestion.text).then((res) => {
        this.setState({ questionTree: res.tree });
      });
      this.props.getDepTree(this.props.activeQuestion.referenceAnswer.text).then((res) => {
        this.setState({ referenceAnswerTree: res.tree });
      });
    }
  }

  renderReactTree(treeObj) {
    function traverse(top) {
      return {
        name: top.word,
        attributes: {
          // lemma: top.lemma,
          arc: top.arc,
          pos: top.POS_fine
        },
        children: top.modifiers.map((modifier) => {
          return traverse(modifier);
        })
      };
    }
    var data = [traverse(treeObj)];
    return (
      <div id="treeWrapper" style={{ height: "25em" }}>
        <Tree
          data={data}
          zoomable={true}
          orientation="horizontal"
          translate={{ x: 4 * 16, y: 10 * 16 }}
          nodeSize={{ x: 140, y: 80 }}
        />
      </div>
    );
  }

  render() {
    const { activeQuestion, questionCount, qIdx } = this.props;
    const { questionTree, referenceAnswerTree, questionToggle, answerToggle } = this.state;

    return (
      <Segment textAlign="center">
        <Message style={{ backgroundColor: "#eff0f6" }}>
          <Message.Header style={{ paddingBottom: "0.5em" }}>
            <Label
              size="large"
              attached="top right"
              style={{ backgroundColor: "#66AB8C", color: "#F2EEE2" }}>
              {activeQuestion.type}
            </Label>
            {"Frage " + (qIdx + 1) + "/" + questionCount}
          </Message.Header>
          <div>{activeQuestion.text}</div>
          <Button
            onClick={() => this.setState({ questionToggle: !questionToggle })}
            style={{ marginTop: "6px", backgroundColor: "#66AB8C", color: "#F2EEE2" }}
            content="View Tree"
            size="tiny"
          />
        </Message>
        {questionToggle && questionTree && this.renderReactTree(questionTree)}
        <Message style={{ backgroundColor: "#eff0f6" }}>
          <Message.Header style={{ paddingBottom: "0.5em" }}>Musterantwort</Message.Header>
          <AnswerMarkup answer={activeQuestion.referenceAnswer} colors={this.state.colors} />
          <Button
            onClick={() => this.setState({ answerToggle: !answerToggle })}
            style={{ marginTop: "6px", backgroundColor: "#66AB8C", color: "#F2EEE2" }}
            content="View Tree"
            size="tiny"
          />
        </Message>
        {answerToggle && referenceAnswerTree && this.renderReactTree(referenceAnswerTree)}
      </Segment>
    );
  }
}
