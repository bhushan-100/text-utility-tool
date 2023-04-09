import axios from "axios";
import React, { useState } from "react";

export default function TextBox() {
  const [text, setText] = useState("");
  const [summary, setSummary] = useState("");

  const [wordCount, setWordCount] = useState(0);
  const [charCount, setCharCount] = useState(0);

  const [summaryWordCount, setSummaryWordCount] = useState(0);

  const getWordCount = (text) =>
    text.split(" ").filter((word) => word !== "").length;

  const getSummary = async (event) => {
    event.preventDefault();

    try {
      const res = await axios.post("http://localhost:5000/summarize", {
        text: text,
      });
      setSummary(res.data.summary);
      setSummaryWordCount(getWordCount(res.data.summary));
    } catch (error) {
      console.log(error);
    }
  };

  return (
    <div className="text-box flex relative">
      <div className="relative">
        <div className="absolute absolute-right counts">{charCount}/2000</div>
        <div className="absolute absolute-left counts">{wordCount} words</div>
        <textarea
          name="text"
          id="text"
          cols="30"
          rows="10"
          value={text}
          onChange={(e) => {
            setText(e.target.value);
            setCharCount(e.target.value.length);
            setWordCount(getWordCount(e.target.value));
          }}
        ></textarea>
      </div>
      <div className="relative">
        <div className="absolute absolute-left counts">
          {summaryWordCount} words
        </div>
        <div className="output">{summary}</div>
      </div>

      <form
        action=""
        className="absolute absolute-bottom"
        onSubmit={getSummary}
      >
        <button type="submit" className="btn">
          Summarize
        </button>
      </form>
    </div>
  );
}
