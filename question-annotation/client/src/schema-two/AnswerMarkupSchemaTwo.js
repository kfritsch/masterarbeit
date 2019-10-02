import React from "react";

export default class AnswerMarkupSchemaTwo extends React.Component {
  constructor(props) {
    super(props);
  }

  getAspectElements(aspects) {
    var aspectElements = [];
    var elem;
    // create an element for every line on or offset
    for (let i = 0; i < aspects.length; i++) {
      for (let j = 0; j < aspects[i].elements.length; j++) {
        elem = aspects[i].elements[j];
        aspectElements.push({ start: true, pos: elem[0], endPos: elem[1], aIdx: i });
        aspectElements.push({ start: false, pos: elem[1], startPos: elem[0], aIdx: i });
      }
    }
    // sort these elements at their position in the text
    aspectElements.sort((a, b) => {
      if (a.pos !== b.pos) {
        return a.pos - b.pos;
      } else if (a.start && b.start) {
        return b.endPos - a.endPos;
      } else if (!a.start && !b.start) {
        return b.startPos - a.startPos;
      } else {
        return a.start - b.start;
      }
    });
    return aspectElements;
  }

  getOverlayingAspects(markupElements, aspects) {
    var activeAspects = [];
    var elem;
    var aspectOverlaps = Array.from({ length: aspects.length }, (u) => new Set([]));
    for (let i = 0; i < markupElements.length; i++) {
      elem = markupElements[i];
      if (elem.start) {
        for (let j = 0; j < activeAspects.length; j++) {
          // only the smaller ones because levels are blocked iteratively so a later aspect can not block a former one
          if (activeAspects[j] < elem.aIdx) {
            aspectOverlaps[elem.aIdx].add(activeAspects[j]);
          }
        }
        activeAspects.push(elem.aIdx);
      } else {
        activeAspects.splice(activeAspects.indexOf(elem.aIdx), 1);
      }
    }
    return aspectOverlaps;
  }

  getAspectLineLevel(aspectOverlaps, aspects) {
    var aspectLevels = Array.from({ length: aspects.length }, (u) => new Array());
    // var aspectLevels = new Array(aspects.length).fill([]);
    var levelMask;
    var level;
    function markOverlappingLevels(overlapId) {
      for (let k = 0; k < aspectLevels[overlapId].length; k++) {
        levelMask[aspectLevels[overlapId][k]] = 1;
      }
    }
    for (let i = 0; i < aspectLevels.length; i++) {
      levelMask = new Array(aspects.length).fill(0);
      // mark the blocked levels
      aspectOverlaps[i].forEach(markOverlappingLevels);
      // select a level for each refAspect taking from the lowest available
      if (this.props.refAnswer || aspects[i].referenceAspects.length === 0) {
        level = levelMask.indexOf(0);
        aspectLevels[i].push(level);
        levelMask[level] = 1;
      } else {
        for (let j = 0; j < aspects[i].referenceAspects.length; j++) {
          level = levelMask.indexOf(0);
          aspectLevels[i].push(level);
          levelMask[level] = 1;
        }
      }
    }
    return aspectLevels;
  }

  getAspectStyles(aspectLevels, aspects) {
    var color;
    var aspectStyles = aspectLevels.map((levels, aIdx) => {
      return levels.map((level, refIdx) => {
        if (this.props.refAnswer) {
          color = this.props.colors[aIdx];
        } else if (aspects[aIdx].referenceAspects.length === 0) {
          color = "#ccc";
        } else {
          color = this.props.colors[aspects[aIdx].referenceAspects[refIdx]];
        }
        return (
          "background: linear-gradient(0deg," +
          color +
          " 4px, white 2px, transparent 2px); padding-bottom: " +
          (2 + level * 5) +
          "px"
        );
      });
    });
    return aspectStyles;
  }

  generateHtmlMarkupString(aspectElements, aspectStyles, aspects, text) {
    var spanStack = [];
    var curPos = 0;
    var subText;
    var stackIndex;
    var stackCount;
    var spanString;
    var elem;
    function findStackIndex(stackElem) {
      return parseInt(stackElem.id) === elem.aIdx;
    }
    var htmlString = "<p>";
    for (let i = 0; i < aspectElements.length; i++) {
      elem = aspectElements[i];
      // add the substring between aspectElements
      if (curPos < elem.pos) {
        subText = text.substr(curPos, elem.pos - curPos);
        htmlString += subText;
      }
      // if the Element is a start add a span with style und push it to the stack for each referenceAspect
      if (elem.start) {
        if (this.props.refAnswer || aspects[elem.aIdx].referenceAspects.length === 0) {
          spanString =
            '<span style="' +
            aspectStyles[elem.aIdx][0] +
            '" id="' +
            elem.aIdx +
            '" refIndex="none">';
          htmlString += spanString;
          spanStack.push({ domString: spanString, id: elem.aIdx, refIndex: "none" });
        } else {
          for (let j = 0; j < aspects[elem.aIdx].referenceAspects.length; j++) {
            spanString =
              '<span style="' +
              aspectStyles[elem.aIdx][j] +
              '" id="' +
              elem.aIdx +
              '" refIndex="' +
              j +
              '">';
            htmlString += spanString;
            spanStack.push({ domString: spanString, id: elem.aIdx, refIndex: j });
          }
        }
      } else {
        // if it is an end, get the element of the finished span in the stack, close this element and all later elements, remember the later elements and add them again to the string and the stack after closing
        stackIndex = spanStack.findIndex(findStackIndex);
        stackCount = spanStack.length - stackIndex;
        for (let j = 0; j < stackCount; j++) {
          htmlString += "</span>";
        }
        if (stackCount > 1) {
          var transferSpans = [];
          for (let j = stackIndex + 1; j < spanStack.length; j++) {
            if (spanStack[stackIndex].id !== spanStack[j].id) {
              transferSpans.push(spanStack[j]);
            }
          }
          spanStack.splice(stackIndex, stackCount);
          for (let j = 0; j < transferSpans.length; j++) {
            htmlString += transferSpans[j].domString;
            spanStack.push(transferSpans[j]);
          }
        } else {
          spanStack.splice(stackIndex, stackCount);
        }
      }
      curPos = elem.pos;
    }
    htmlString += text.substr(curPos, text.length - curPos);
    htmlString += "</p>";
    return htmlString;
  }

  render() {
    const { answer } = this.props;
    // console.log(answer);
    const text = "correctionOrComment" in answer ? answer.correctionOrComment : answer.text;
    const aspects = JSON.parse(JSON.stringify(answer.aspects));
    !this.props.refAnswer && aspects.splice(aspects.length - 1, 1);
    // if there are no aspects return the text
    if (aspects.length === 0) {
      return <span key={"whole"}>{text}</span>;
    }
    var aspectElements = this.getAspectElements(aspects);
    var aspectOverlaps = this.getOverlayingAspects(aspectElements, aspects);
    var aspectLevels = this.getAspectLineLevel(aspectOverlaps, aspects);
    var aspectStyles = this.getAspectStyles(aspectLevels, aspects);
    // generate the HtmlString
    var htmlString = this.generateHtmlMarkupString(aspectElements, aspectStyles, aspects, text);
    return <div dangerouslySetInnerHTML={{ __html: htmlString }} />;
  }
}
