import React from "react";

import { Message, Button, Dropdown, Label, Segment, Input, TextArea } from "semantic-ui-react";
import AnswerMarkupSchemaTwo from "./AnswerMarkupSchemaTwo";
import Tree from "react-d3-tree";

const ROUGH_CATEGORIES = [
  "correct",
  "binary_correct",
  "partially_correct",
  "missconception",
  "concept_mix-up",
  "irrelevant",
  "none"
];

const ASPECT_LABELS = [
  { name: "correct", color: "#ddcccc" },
  { name: "unprecise", color: "#8c6363" },
  { name: "contradiction", color: "#663b3b" }
];

export default class AnswerAnnotationSchemaTwo extends React.Component {
  constructor(props) {
    super(props);
    var activeAnswer = props.activeQuestion.studentAnswers[props.annIdx];
    activeAnswer.text = activeAnswer.text.replace(/\s\s+/g, " ");
    this.state = {
      activeAnswer,
      activeAnswerAnnotation: JSON.parse(JSON.stringify(props.currentAnnotation)),
      studentAnswerTree: null,
      answerToggle: false
    };
    this.colors = props.getColors(props.activeQuestion);
    this.matchingLabels = props.activeQuestion.referenceAnswer.aspects.map((aspect, index) => {
      return {
        name: aspect.text,
        color: this.colors[index]
      };
    });
  }

  componentDidMount() {
    this.props.getDepTree("Das ist nicht schön.").then((res) => {
      this.setState({ studentAnswerTree: res.tree });
    });
  }

  componentDidUpdate(prevProps) {
    if (
      this.props.activeQuestion.id !== prevProps.activeQuestion.id ||
      this.props.annIdx !== prevProps.annIdx
    ) {
      this.props
        .getDepTree(this.props.activeQuestion.studentAnswers[this.props.annIdx].text)
        .then((res) => {
          this.setState({ studentAnswerTree: res.tree });
        });
      var activeAnswer = this.props.activeQuestion.studentAnswers[this.props.annIdx];
      activeAnswer.text = activeAnswer.text.replace(/\s\s+/g, " ");
      this.colors = this.props.getColors(this.props.activeQuestion);
      this.matchingLabels = this.props.activeQuestion.referenceAnswer.aspects.map(
        (aspect, index) => {
          return {
            name: aspect.text,
            color: this.colors[index]
          };
        }
      );
      this.setState({
        activeAnswer,
        activeAnswerAnnotation: JSON.parse(JSON.stringify(this.props.currentAnnotation))
      });
    }
  }

  getAnswerAnnotation() {
    var { activeAnswerAnnotation, activeAnswer } = this.state;
    activeAnswerAnnotation["id"] = activeAnswer["id"];
    if (!activeAnswerAnnotation.answerCategory) return false;
    activeAnswerAnnotation.aspects.pop();
    return activeAnswerAnnotation;
  }

  categoryDropdownChange = (e, obj) => {
    var { activeAnswerAnnotation } = this.state;
    activeAnswerAnnotation.answerCategory = obj.value;
    this.setState({ activeAnswerAnnotation });
  };

  handleChange = (e) => {
    var { activeAnswerAnnotation } = this.state;
    var annotationAspects = activeAnswerAnnotation.aspects;
    var answerText = activeAnswerAnnotation.text;
    annotationAspects[e.target.name].text = e.target.value;
    // Add new entry if last received a text value
    if (annotationAspects[annotationAspects.length - 1].text) {
      annotationAspects.push({
        text: "",
        elements: [],
        referenceAspects: [],
        label: ""
      });
    }
    // Mark the text if found in the referenceanswer
    var answerParts = e.target.value.split(";;");
    var answerPart;
    annotationAspects[e.target.name].elements = [];
    for (var i = 0; i < answerParts.length; i++) {
      answerPart = answerParts[i].trim();
      var idx = answerText.indexOf(answerPart);
      if (idx >= 0) {
        annotationAspects[e.target.name].elements.push([idx, idx + answerPart.length]);
      }
    }
    annotationAspects[e.target.name].elements.sort((a, b) => {
      return a[0] - b[0];
    });
    annotationAspects.sort((a, b) => {
      if (a.elements.length === 0 || b.elements.length === 0) {
        return b.elements.length - a.elements.length;
      } else {
        return a.elements[0][0] - b.elements[0][0];
      }
    });
    // Remove entry if a textfield was cleared
    if (!e.target.value && e.target.name !== annotationAspects.length - 1) {
      annotationAspects.splice(e.target.name, 1);
    }

    activeAnswerAnnotation["aspects"] = annotationAspects;
    this.setState({ activeAnswerAnnotation });
  };

  matchingChange = (e, obj) => {
    var values = obj.value.map((val) => val);
    var activeAnswerAnnotation = JSON.parse(JSON.stringify(this.state.activeAnswerAnnotation));
    var activeQuestion = this.props.activeQuestion;
    values.sort();
    activeAnswerAnnotation.aspects[obj.annIndex].referenceAspects = values;
    activeAnswerAnnotation.aspects.sort(
      (aspectA, aspectB) => aspectA.referenceAspects[0] - aspectB.referenceAspects[0]
    );
    this.setState({ activeAnswerAnnotation });
  };

  labelingChange = (e, obj) => {
    var value = obj.value;
    var annIndex = obj.annIndex;
    var { activeAnswerAnnotation } = this.state;
    activeAnswerAnnotation.aspects[annIndex].label = value;
    this.setState({ activeAnswerAnnotation });
  };

  handleAreaChange = (e) => {
    var { activeAnswerAnnotation } = this.state;
    activeAnswerAnnotation.correctionOrComment = e.target.value;
    this.setState({ activeAnswerAnnotation });
  };

  deleteMatch = (e, { value }) => {
    var { activeAnswerAnnotation } = this.state;
    var aspects = activeAnswerAnnotation.aspects;
    if (aspects.length === 1) {
      aspects[0] = {
        text: "",
        elements: [],
        referenceAspects: [],
        label: ""
      };
    } else {
      aspects.splice(value, 1);
    }
    activeAnswerAnnotation.aspects = aspects;
    this.setState({ activeAnswerAnnotation });
  };

  renderDropdownLabel(dropDownTag, label) {
    var color = this.state.colors[label.value];
    var value = label.text.props.children;
    return {
      value: label.value,
      content: this.getDropDownLabel(color, value, label.value)
    };
  }

  getDropdownOptions(options) {
    return options.map((option, index) => {
      return {
        key: index,
        value: index,
        text: this.getDropDownLabel(option.color, option.name, index)
      };
    });
  }

  getDropDownLabel(color, content, value) {
    return (
      <Label
        id="survey-label"
        circular
        value={value}
        className="annotation-label"
        style={{
          borderColor: color
        }}>
        {content}
      </Label>
    );
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

  renderMatchings() {
    var { activeAnswerAnnotation } = this.state;
    return activeAnswerAnnotation.aspects.map((annotationAspect, annIndex) => {
      var empty = !annotationAspect.text;
      var referenceAspects = annotationAspect.referenceAspects;
      var label = annotationAspect.label;
      return (
        <Input
          id={"match" + annIndex}
          key={annIndex}
          fluid
          style={{ marginTop: "1vh", marginBottom: "1vh" }}>
          <input
            placeholder={annIndex + 1 + ". Aspekt"}
            name={annIndex}
            value={annotationAspect.text}
            onChange={this.handleChange.bind(this)}
            width={12}
            style={{ minWidth: "30vw", background: "#ffffff" }}
          />
          <Dropdown
            id={"annDrop_" + annIndex}
            key={"annDrop_" + annIndex}
            annIndex={annIndex}
            selection
            placeholder="Select Match"
            disabled={empty}
            // defaultValue={annotationAspect.referenceAspects}
            value={referenceAspects}
            multiple={true}
            options={this.getDropdownOptions(this.matchingLabels)}
            renderLabel={this.renderDropdownLabel.bind(this, "matching")}
            onChange={this.matchingChange.bind(this)}
            width={3}
            style={{ marginLeft: "0.5vh" }}
          />
          <Dropdown
            id={"nlpDrop_" + annIndex}
            key={"nlpDrop_" + annIndex}
            annIndex={annIndex}
            selection
            placeholder="Select Label"
            disabled={referenceAspects.length > 0}
            defaultValue={0}
            value={annotationAspect.label}
            options={this.getDropdownOptions(ASPECT_LABELS)}
            onChange={this.labelingChange.bind(this)}
            width={3}
            style={{ marginLeft: "0.5vh" }}
          />
          <Button
            disabled={!annotationAspect.text}
            value={annIndex}
            icon="delete"
            onClick={this.deleteMatch.bind(this)}
            width={1}
            style={{ marginLeft: "0.5vh" }}
          />
        </Input>
      );
    });
  }

  render() {
    var { activeAnswerAnnotation, studentAnswerTree, answerToggle } = this.state;
    var { annIdx, activeQuestion, currentAnnotation, goldStandard, version } = this.props;
    return (
      <div>
        <Segment.Group className="student-answer-container">
          <Segment textAlign="center">
            <Message style={{ backgroundColor: "#eff0f6" }}>
              <Message.Header style={{ paddingBottom: "0.5em" }}>
                {"Schülerantwort " + (annIdx + 1) + "/" + activeQuestion.studentAnswers.length}
              </Message.Header>
              <AnswerMarkupSchemaTwo
                refAnswer={false}
                answer={activeAnswerAnnotation}
                colors={this.state.colors}
              />
              <Button
                className="tree-button"
                onClick={() => this.setState({ answerToggle: !answerToggle })}
                icon="code branch"
                size="mini"
              />
            </Message>
            {answerToggle && studentAnswerTree && this.renderReactTree(studentAnswerTree)}

            {version == "aspects" && this.renderMatchings()}
          </Segment>
          {goldStandard && (
            <Segment textAlign="center">
              <TextArea
                rows={3}
                style={{ width: "80%" }}
                autoHeight
                placeholder={goldStandard ? "Korrigierter Text " : "Kommentar"}
                onChange={this.handleAreaChange.bind(this)}
                value={activeAnswerAnnotation.correctionOrComment}
              />
            </Segment>
          )}
          {this.version == "whole" && (
            <Segment textAlign="center">
              {/* <Message textAlign="center" style={{ backgroundColor: "#eff0f6", width: "40%" }}> */}
              <Dropdown
                value={activeAnswerAnnotation.answerCategory}
                placeholder="Answer Category"
                options={ROUGH_CATEGORIES.map((cat) => {
                  return {
                    value: cat,
                    text: cat.replace("_", " "),
                    key: cat
                  };
                })}
                button
                onChange={this.categoryDropdownChange.bind(this)}
                style={{ gridArea: "box2", textAlign: "center", backgroundColor: "#a5696942" }}
              />
              {/* </Message> */}
            </Segment>
          )}
        </Segment.Group>
      </div>
    );
  }
}
