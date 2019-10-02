import React from "react";

import { Message, Button, Dropdown, Label, Segment, Input } from "semantic-ui-react";
import AnswerMarkupSchemaTwo from "./AnswerMarkupSchemaTwo";
import Tree from "react-d3-tree";

const ASPECT_LABELS = [
  { name: "correct", color: "#ddcccc" },
  { name: "unprecise", color: "#8c6363" },
  { name: "contradiction", color: "#663b3b" }
];

export default class AspectAnnotation extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      aspects: JSON.parse(JSON.stringify(props.currentAnnotation.aspects)),
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
    this.props.getDepTree(this.props.currentAnnotation["correctionOrComment"]).then((res) => {
      this.setState({ studentAnswerTree: res.tree });
    });
  }

  componentDidUpdate(prevProps) {
    if (
      this.props.activeQuestion.id !== prevProps.activeQuestion.id ||
      this.props.annIdx !== prevProps.annIdx
    ) {
      this.props.getDepTree(this.props.currentAnnotation["correctionOrComment"]).then((res) => {
        this.setState({ studentAnswerTree: res.tree });
      });
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
        aspects: JSON.parse(JSON.stringify(this.props.currentAnnotation.aspects)),
        studentAnswerTree: null,
        answerToggle: false
      });
    }
  }

  getAnswerAnnotation() {
    var { aspects } = this.state;
    console.log(aspects);
    if (aspects.find((aspect) => aspect.text && !aspect.referenceAspects.length > 0)) return false;
    var activeAnswerAnnotation = JSON.parse(JSON.stringify(this.props.currentAnnotation));
    activeAnswerAnnotation.aspects = aspects;
    activeAnswerAnnotation.aspects.pop();
    return activeAnswerAnnotation;
  }

  handleChange = (e, obj) => {
    var value = e.target.value;
    var annIdx = e.target.name;
    var { aspects } = this.state;
    var currentAspect = aspects[annIdx];
    currentAspect.text = value;

    // Mark the text if found in the referenceanswer
    var answerText = this.props.currentAnnotation.correctionOrComment;
    var answerParts = value.split(";;");
    currentAspect.elements = [];
    for (var i = 0; i < answerParts.length; i++) {
      var answerPart = answerParts[i].trim();
      var idx = answerText.indexOf(answerPart);
      if (idx >= 0) {
        currentAspect.elements.push([idx, idx + answerPart.length]);
      }
    }
    // sort elements ascending from startpoint
    currentAspect.elements.sort((a, b) => {
      return a[0] - b[0];
    });
    // sort aspects ascending reference aspects order
    aspects.sort((a, b) => {
      if (a.elements.length === 0 || b.elements.length === 0) {
        return b.elements.length - a.elements.length;
      } else {
        return a.elements[0][0] - b.elements[0][0];
      }
    });
    // Remove entry if a textfield was cleared
    if (!value && annIdx !== aspects.length - 1) {
      aspects.splice(annIdx, 1);
    }

    this.setState({ aspects });
  };

  matchingChange = (e, obj) => {
    var values = obj.value.map((val) => val);
    var aspects = JSON.parse(JSON.stringify(this.state.aspects));
    values.sort();
    aspects[obj.annIndex].referenceAspects = values;
    if (!("label" in aspects[obj.annIndex])) aspects[obj.annIndex]["label"] = 0;
    aspects.sort((aspectA, aspectB) => aspectA.referenceAspects[0] - aspectB.referenceAspects[0]);
    if (aspects[aspects.length - 1].referenceAspects.length > 0) {
      aspects.push({
        text: "",
        elements: [],
        referenceAspects: []
      });
    }
    this.setState({ aspects });
  };

  labelingChange = (e, obj) => {
    var value = obj.value;
    var annIndex = obj.annIndex;
    var { aspects } = this.state;
    aspects[annIndex].label = value;
    this.setState({ aspects });
  };

  deleteMatch = (e, { value }) => {
    var { aspects } = this.state;
    if (aspects.length === 1) {
      aspects[0] = {
        text: "",
        elements: [],
        referenceAspects: []
      };
    } else {
      aspects.splice(value, 1);
    }
    this.setState({ aspects });
  };

  renderDropdownLabel(dropDownTag, label) {
    var color = this.colors[label.value];
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
    var { aspects } = this.state;
    return aspects.map((annotationAspect, annIndex) => {
      var empty = !annotationAspect.text;
      var referenceAspects = annotationAspect.referenceAspects;
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
            disabled={referenceAspects.length === 0}
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
    var { aspects, studentAnswerTree, answerToggle } = this.state;
    var { annIdx, activeQuestion, currentAnnotation } = this.props;
    var activeAnswerAnnotation = { ...currentAnnotation, aspects };
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
                colors={this.colors}
              />
              <Button
                className="tree-button"
                onClick={() => this.setState({ answerToggle: !answerToggle })}
                icon="code branch"
                size="mini"
              />
            </Message>
            {answerToggle && studentAnswerTree && this.renderReactTree(studentAnswerTree)}

            {this.renderMatchings()}
          </Segment>
        </Segment.Group>
      </div>
    );
  }
}
