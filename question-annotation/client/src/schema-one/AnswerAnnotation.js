import React from "react";

import { Message, Button, Dropdown, Label, Segment, Input, TextArea } from "semantic-ui-react";
import AnswerMarkup from "./AnswerMarkup";
import Tree from "react-d3-tree";

const ROUGH_CATEGORIES = [
  "correct",
  "binary_correct",
  "partially_correct",
  "missconception",
  "concept_mix-up",
  "guessing",
  "none"
];

const SEMANTIC_NLP_LABELS = [
  { name: "same", color: "#ddcccc" },
  { name: "synonynm", color: "#af8e8e" },
  { name: "paraphrase", color: "#8c6363" },
  { name: "expertise", color: "#663b3b" },
  { name: "hyponym", color: "#da7878" },
  { name: "hypernym", color: "#983e3e" }
]; // On top of spellchecking, lemmatization,

const ADDITIONAL_LABELS = [
  { name: "correct", color: "#c6c6c6" },
  { name: "false", color: "#919191" },
  { name: "example", color: "#444444" },
  { name: "redundant", color: "#000000" }
];

export default class AnswerAnnotation extends React.Component {
  constructor(props) {
    super(props);
    var activeAnswer = props.activeQuestion.studentAnswers[props.annIdx];
    activeAnswer.text = activeAnswer.text.replace(/\s\s+/g, " ");
    this.state = {
      colors: props.getColors(props.activeQuestion),
      activeAnswer,
      activeAnswerAnnotation: props.currentAnnotation,
      studentAnswerTree: null,
      answerToggle: false
    };
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
      this.setState({
        colors: this.props.getColors(this.props.activeQuestion),
        activeAnswer,
        activeAnswerAnnotation: this.props.currentAnnotation
      });
    }
  }

  getAnswerAnnotation() {
    // TODO: clean up points
    var { activeAnswerAnnotation, activeAnswer } = this.state;
    var activeQuestion = this.props.activeQuestion;
    activeAnswerAnnotation["id"] = activeAnswer["id"];
    var aspectCount = activeQuestion.referenceAnswer.aspects.length;
    if (activeAnswerAnnotation.answerCategory) {
      activeAnswerAnnotation.aspects.pop();
    } else {
      return false;
    }
    var refAspects;
    const referenceSet = new Set([]);
    for (var i = 0; i < activeAnswerAnnotation.aspects.length; i++) {
      refAspects = activeAnswerAnnotation.aspects[i].referenceAspects;
      if (refAspects.length === 1 && refAspects[0] === aspectCount) {
      } else {
        for (var j = 0; j < refAspects.length; j++) {
          referenceSet.add(refAspects[j]);
        }
      }
    }
    activeAnswerAnnotation["points"] = referenceSet.size;
    return activeAnswerAnnotation;
  }

  handleChange = (e) => {
    var { activeAnswerAnnotation } = this.state;
    var annotationAspects = activeAnswerAnnotation.aspects;
    var answerText = activeAnswerAnnotation.text;
    annotationAspects[e.target.name].text = e.target.value;
    // Add new entry if last received a text value
    if (annotationAspects[annotationAspects.length - 1].text) {
      annotationAspects.push({
        text: "",
        start: 0,
        end: 0,
        referenceAspects: [],
        nlpLabels: [],
        additionalLabels: []
      });
    }
    // Mark the text if found in the referenceanswer
    var idx = answerText.indexOf(e.target.value);
    if (idx >= 0) {
      annotationAspects[e.target.name].start = idx;
      annotationAspects[e.target.name].end = idx + e.target.value.length;
    }
    // Remove entry if a textfield was cleared
    if (!e.target.value && e.target.name !== annotationAspects.length - 1) {
      annotationAspects.splice(e.target.name, 1);
    }

    activeAnswerAnnotation["aspects"] = annotationAspects;
    this.setState({ activeAnswerAnnotation });
  };

  handleAreaChange = (e) => {
    var { activeAnswerAnnotation } = this.state;
    activeAnswerAnnotation.correctionOrComment = e.target.value;
    this.setState({ activeAnswerAnnotation });
  };

  dropdownChange = (e, obj) => {
    var value = obj.value;
    var annIndex = parseInt(obj.id.split("_")[1]);
    var { activeAnswerAnnotation } = this.state;
    var activeQuestion = this.props.activeQuestion;
    var aspectCount = activeQuestion.referenceAnswer.aspects.length;
    var refVals = [];
    for (var i = 0; i < value.length; i++) {
      var val = value[i];
      if (val === aspectCount) {
        refVals = [aspectCount];
        break;
      }
      refVals.push(val % activeQuestion.referenceAnswer.aspects.length);
    }
    refVals.sort();
    activeAnswerAnnotation.aspects[annIndex].referenceAspects = refVals;
    // TODO:clean up sorting
    function sortValues(idxA, idxB) {
      var a = activeAnswerAnnotation.aspects[idxA];
      var b = activeAnswerAnnotation.aspects[idxB];
      if (!a.text) {
        return 1;
      } else if (!b.text) {
        return -1;
      } else if (a.referenceAspects.length === 0 || b.referenceAspects.length === 0) {
        return b.referenceAspects.length - a.referenceAspects.length;
      } else {
        return a.referenceAspects[0] - b.referenceAspects[0];
      }
    }

    var indices = new Array(activeAnswerAnnotation.aspects.length);
    for (i = 0; i < activeAnswerAnnotation.aspects.length; ++i) indices[i] = i;
    indices.sort(sortValues);
    var sortedAspects = [];
    var idx;
    for (i = 0; i < indices.length; i++) {
      idx = indices[i];
      sortedAspects.push(activeAnswerAnnotation.aspects[idx]);
    }
    activeAnswerAnnotation.aspects = sortedAspects;
    this.setState({ activeAnswerAnnotation });
  };

  nlpDropdownChange = (e, obj) => {
    var value = obj.value;
    var annIndex = parseInt(obj.id.split("_")[1]);
    var { activeAnswerAnnotation } = this.state;
    var additional =
      activeAnswerAnnotation.aspects[annIndex].referenceAspects.length === 1 &&
      this.props.activeQuestion.referenceAnswer.aspects.length ===
        activeAnswerAnnotation.aspects[annIndex].referenceAspects[0];

    var refVals = [];
    for (var i = 0; i < value.length; i++) {
      refVals.push(value[i]);
    }
    refVals.sort();
    if (additional) {
      activeAnswerAnnotation.aspects[annIndex].additionalLabels = refVals;
    } else {
      activeAnswerAnnotation.aspects[annIndex].nlpLabels = refVals;
    }

    this.setState({ activeAnswerAnnotation });
  };

  categoryDropdownChange = (e, obj) => {
    var { activeAnswerAnnotation } = this.state;
    activeAnswerAnnotation.answerCategory = obj.value;
    this.setState({ activeAnswerAnnotation });
  };

  deleteMatch = (e, { value }) => {
    var { activeAnswerAnnotation } = this.state;
    var aspects = activeAnswerAnnotation.aspects;
    if (aspects.length === 1) {
      aspects[0] = {
        text: "",
        start: 0,
        end: 0,
        referenceAspects: [],
        nlpLabels: [],
        additionalLabels: []
      };
    } else {
      aspects.splice(value, 1);
    }
    activeAnswerAnnotation.aspects = aspects;
    this.setState({ activeAnswerAnnotation });
  };

  getLabelProperties(dropDownTag, label) {
    switch (dropDownTag) {
      case "matching":
        return {
          color: this.state.colors[label.value],
          value: label.children.props.children
        };
      case "nlp":
        return {
          color: SEMANTIC_NLP_LABELS[label.value].color,
          value: SEMANTIC_NLP_LABELS[label.value].name
        };
      case "additional":
        return {
          color: ADDITIONAL_LABELS[label.value].color,
          value: ADDITIONAL_LABELS[label.value].name
        };
      default:
        return {
          color: this.state.colors[label.value],
          value: label.children.props.children
        };
    }
  }

  renderDropdownLabel(dropDownTag, label) {
    var labelProps = this.getLabelProperties(dropDownTag, label);
    return {
      value: label.value,
      content: (
        <Label
          id="survey-label"
          circular
          className="annotation-label"
          style={{
            borderColor: labelProps.color
          }}>
          {labelProps.value}
        </Label>
      )
    };
  }

  getDropdownOptions(options) {
    return options.map((option, index) => {
      return {
        key: index,
        value: index,
        children: (
          <Label
            id="survey-label"
            circular
            className="annotation-label"
            style={{
              borderColor: option.color
            }}>
            {option.name}
          </Label>
        )
      };
    });
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
    var activeQuestion = this.props.activeQuestion;
    var annotationAspects = activeAnswerAnnotation.aspects;
    var aspects = activeQuestion.referenceAnswer.aspects;
    var matchingLabels = aspects.map((aspect, index) => {
      return {
        name: aspect.text,
        color: this.state.colors[index]
      };
    });
    matchingLabels.push({
      name: "additional",
      color: this.state.colors[this.state.colors.length - 1]
    });

    return annotationAspects.map((annotationAspect, annIndex) => {
      var additional =
        activeAnswerAnnotation.aspects[annIndex].referenceAspects.length === 1 &&
        aspects.length === activeAnswerAnnotation.aspects[annIndex].referenceAspects[0];
      var refVals = annotationAspect.referenceAspects;
      var nlpValues = additional ? annotationAspect.additionalLabels : annotationAspect.nlpLabels;
      var secondLabels = additional ? ADDITIONAL_LABELS : SEMANTIC_NLP_LABELS;
      var matchOptions = this.getDropdownOptions(matchingLabels);
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
            selection
            multiple
            placeholder="Select Match"
            // defaultValue={annotationAspect.referenceAspects}
            value={refVals}
            options={additional ? [matchOptions[matchOptions.length - 1]] : matchOptions}
            renderLabel={this.renderDropdownLabel.bind(this, "matching")}
            onChange={this.dropdownChange.bind(this)}
            width={3}
            style={{ marginLeft: "0.5vh" }}
          />
          {this.props.goldStandard && (
            <Dropdown
              id={"nlpDrop_" + annIndex}
              key={"nlpDrop_" + annIndex}
              selection
              multiple
              placeholder="Select NLP Step"
              // defaultValue={annotationAspect.referenceAspects}
              value={nlpValues}
              options={this.getDropdownOptions(secondLabels)}
              renderLabel={
                additional
                  ? this.renderDropdownLabel.bind(this, "additional")
                  : this.renderDropdownLabel.bind(this, "nlp")
              }
              onChange={this.nlpDropdownChange.bind(this)}
              width={3}
              style={{ marginLeft: "0.5vh" }}
            />
          )}
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
    var { annIdx, activeQuestion, goldStandard } = this.props;
    return (
      <div>
        <Segment.Group className="student-answer-container">
          <Segment textAlign="center">
            <Message style={{ backgroundColor: "#eff0f6" }}>
              <Message.Header style={{ paddingBottom: "0.5em" }}>
                {"Schülerantwort " + (annIdx + 1) + "/" + activeQuestion.studentAnswers.length}
              </Message.Header>
              <AnswerMarkup answer={activeAnswerAnnotation} colors={this.state.colors} />
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
        </Segment.Group>
      </div>
    );
  }
}
