import React from "react";

import { Label } from "semantic-ui-react";

export default class AnswerMarkup extends React.Component {
  getExtraLabels(referenceAspects) {
    var extraLabels = [];
    for (var i = 1; i < referenceAspects.length; i++) {
      extraLabels.push(
        <Label
          key={"aspectExt" + this.props.colors[referenceAspects[i]]}
          circular
          size="tiny"
          empty
          style={{
            marginLeft: "2px",
            border: "1px solid black",
            backgroundColor: this.props.colors[referenceAspects[i]]
          }}>
          {" "}
        </Label>
      );
    }
    return extraLabels;
  }

  render() {
    const { text, aspects } = this.props.answer;
    var sortedAspects = JSON.parse(JSON.stringify(aspects));
    if (sortedAspects.length === 0) {
      return <span key={"whole"}>{text}</span>;
    }
    function compareAspects(a, b) {
      return a.start - b.start;
    }
    sortedAspects.sort(compareAspects);
    var nodes = [];

    if (sortedAspects.length === 1 && !sortedAspects[0].text) {
      return <span key={"span0"}>{text}</span>;
    }
    var refAnswer = !sortedAspects[0].referenceAspects;
    if (sortedAspects[0].start !== 0) {
      nodes.push(<span key={"span0"}>{text.substr(0, sortedAspects[0].start)}</span>);
    }

    sortedAspects[0].text &&
      nodes.push(
        <Label
          id="survey-label"
          key={"aspect0"}
          circular
          className="survey-label annotation-label"
          style={{
            borderColor: refAnswer
              ? this.props.colors[0]
              : this.props.colors[sortedAspects[0].referenceAspects[0]]
          }}>
          {sortedAspects[0].text}
          {!refAnswer &&
            sortedAspects[0].referenceAspects.length > 1 &&
            this.getExtraLabels(sortedAspects[0].referenceAspects)}
        </Label>
      );
    for (var i = 1; i < sortedAspects.length; i++) {
      if (sortedAspects[i].start > sortedAspects[i - 1].end + 1) {
        nodes.push(
          <span key={"span" + i}>
            {text.substr(
              sortedAspects[i - 1].end,
              sortedAspects[i].start - sortedAspects[i - 1].end
            )}
          </span>
        );
      }
      sortedAspects[i].text &&
        nodes.push(
          <Label
            id="survey-label"
            key={"aspect" + i}
            circular
            className="survey-label annotation-label"
            style={{
              borderColor: refAnswer
                ? this.props.colors[i]
                : this.props.colors[sortedAspects[i].referenceAspects[0]]
            }}>
            {sortedAspects[i].text}
            {!refAnswer &&
              sortedAspects[i].referenceAspects.length > 1 &&
              this.getExtraLabels(sortedAspects[i].referenceAspects)}
          </Label>
        );
    }
    var end =
      sortedAspects[sortedAspects.length - 1].end > 0
        ? sortedAspects[sortedAspects.length - 1].end
        : sortedAspects[sortedAspects.length - 2].end;
    if (end < text.length) {
      nodes.push(
        <span key={"span" + sortedAspects.length}>{text.substr(end, text.length - 1 - end)}</span>
      );
    }
    return <div>{nodes}</div>;
  }
}
