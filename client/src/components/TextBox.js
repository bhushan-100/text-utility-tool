import axios from "axios";
import React, { useState, useEffect } from "react";
import dlIcon from "../assets/images/download.svg";
import copyIcon from "../assets/images/copy.svg";
import historyIcon from "../assets/images/history.svg";
import crossIcon from "../assets/images/cross.svg";

export default function TextBox() {
  const [text, setText] = useState("");
  const [output, setOutput] = useState("");

  const [wordCount, setWordCount] = useState(0);
  const [charCount, setCharCount] = useState(0);

  const [outputWordCount, setOutputWordCount] = useState(0);

  const [loading, setLoading] = useState({});

  const [alertMsg, setAlertMsg] = useState("");

  const [showHistory, setShowHistory] = useState(false);

  const [summaries, setSummaries] = useState(
    localStorage.getItem("summaries")
      ? JSON.parse(localStorage.getItem("summaries"))
      : []
  );

  useEffect(() => {
    setWordCount(getWordCount(text));
    setCharCount(text.length);
    if (typeof output === "string") setOutputWordCount(getWordCount(output));
    else setOutputWordCount(output.length);

    setSummaries(
      localStorage.getItem("summaries")
        ? JSON.parse(localStorage.getItem("summaries"))
        : []
    );
  }, [output, text]);

  const getWordCount = (text) =>
    text.split(" ").filter((word) => word !== "").length;

  const getSummary = async (event) => {
    event.preventDefault();
    setLoading({ ...loading, btn1: true, any: true });

    try {
      if (text === "") {
        setLoading({ ...loading, btn1: false, any: false });
        return;
      }
      const res = await axios.post("http://localhost:5000/summarize", {
        text: text.trim(),
      });
      setOutput(res.data.summary.split(".").join(". "));

      setLoading({ ...loading, btn1: false, any: false });

      localStorage.setItem(
        "summaries",
        JSON.stringify([
          ...summaries,
          {
            id: text.split(" ")[0] + Math.round(Math.random() * 100000),
            text: text,
            summary: res.data.summary.split(".").join(". "),
          },
        ])
      );

      setSummaries(
        localStorage.getItem("summaries")
          ? JSON.parse(localStorage.getItem("summaries"))
          : []
      );
    } catch (error) {
      console.log(error);
    }
  };

  const getMisspelled = async (event) => {
    if (text === "") return;
    event.preventDefault();
    setLoading({ ...loading, btn2: true, any: true });

    try {
      const res = await axios.post("http://localhost:5000/spellcheck", {
        text: text.trim(),
      });
      let op = [];

      let origWords = text
        .replace(/\n/g, " ")
        .split(" ")
        .filter((word) => word !== "");

      let correctedWords = res.data.misspelled.split(" ");

      console.log(origWords.length, correctedWords.length);

      console.log(origWords, correctedWords);

      for (let i = 0; i < correctedWords.length; i++) {
        let word = correctedWords[i].trim();
        if (word !== origWords[i].trim()) {
          console.log(origWords[i].trim(), correctedWords[i].trim());
          op.push(<span className="autocorrected">{word}</span>);
        } else {
          op.push(word);
        }
      }

      setOutput(
        op.map((word, index) => (
          <React.Fragment key={index}>{word} </React.Fragment>
        ))
      );

      setTimeout(() => {
        setLoading({ ...loading, btn3: false, any: false });
      }, 300);
    } catch (error) {
      console.log(error);
    }
  };

  const getMeaning = async (event) => {
    event.preventDefault();
    setLoading({ ...loading, btn3: true, any: true });

    try {
      const res = await axios.get(
        `https://api.dictionaryapi.dev/api/v2/entries/en/${text.trim()}`
      );
      setOutput(res.data[0].meanings[0].definitions[0].definition);
      setTimeout(() => {
        setLoading({ ...loading, btn3: false, any: false });
      }, 300);
    } catch (error) {
      console.log(error);
    }
  };

  const downloadPdf = async (event) => {
    event.preventDefault();
    try {
      const response = await axios.post(
        "http://localhost:5000/download-pdf",
        { text: text, summary: output },
        { responseType: "blob" }
      );

      // Create a URL for the PDF file
      const url = window.URL.createObjectURL(new Blob([response.data]));

      // Create a link element to download the PDF file
      const link = document.createElement("a");
      link.href = url;
      link.setAttribute("download", "summary.pdf");
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    } catch (error) {
      console.error(error);
    }
  };

  return (
    <>
      {showHistory && summaries && (
        <div
          className="history flex center"
          style={{ position: "fixed", top: "16vh", right: "2vw", zIndex: 1000 }}
        >
          <h2>History</h2>
          <button
            className="flex center icon absolute"
            style={{ top: 0, right: 0 }}
            onClick={() => setShowHistory(false)}
          >
            <img src={crossIcon} alt="Close history" />
          </button>
          {summaries.map((summ) => (
            <button
              key={summ.id}
              onClick={(e) => {
                e.preventDefault();
                setText(summ.text);
                setOutput(summ.summary);
              }}
              className="hist-btn"
            >{`${summ.text.substring(0, 20)}...`}</button>
          ))}
        </div>
      )}
      <div className="flex home text-box relative">
        {alertMsg && (
          <div
            className="absolute flex center alert success"
            style={{ top: 0, right: 0 }}
          >
            {alertMsg}
          </div>
        )}

        <div className="grid input">
          <div className="flex justify-space-betn">
            <div className="counts flex center">{wordCount} words</div>
            <div className="counts flex center">{charCount}/4000</div>
          </div>
          <textarea
            name="text"
            id="text"
            cols="30"
            rows="10"
            value={text}
            onChange={(e) => {
              setText(e.target.value);
            }}
            spellCheck="false"
          />
          <div className="flex gap-8">
            <form action="POST" onSubmit={getSummary}>
              <button
                type="submit"
                className="btn flex center"
                disabled={loading.any ? true : false}
              >
                {loading.btn1 ? <div className="loader"></div> : "Summarize"}
              </button>
            </form>
            <form action="POST" onSubmit={getMisspelled}>
              <button
                type="submit"
                className="btn flex center"
                disabled={loading.any ? true : false}
              >
                {loading.btn2 ? <div className="loader"></div> : "Spell Check"}
              </button>
            </form>
            <form action="POST" onSubmit={getMeaning}>
              <button
                type="submit"
                className="btn flex center"
                disabled={loading.any ? true : false}
              >
                {loading.btn3 ? <div className="loader"></div> : "Meaning"}
              </button>
            </form>
          </div>
        </div>
        <div className="grid output">
          <div className="counts flex center" style={{ width: "6.5rem" }}>
            {outputWordCount} words
          </div>
          <div className="output-text">{output}</div>
          <div className="flex gap-8">
            <button
              className="flex center icon"
              onClick={() => {
                navigator.clipboard.writeText(output);
                setAlertMsg("Copied to clipboard!");
                setTimeout(() => {
                  setAlertMsg("");
                }, 2000);
              }}
            >
              <img src={copyIcon} alt="Copy output" />
            </button>
            {/* <button className="flex center icon" onClick={downloadPdf}>
              <img src={dlIcon} alt="Download output in .pdf" />
            </button> */}
            <button
              className="flex center icon"
              onClick={() => {
                setShowHistory(true);
              }}
            >
              <img src={historyIcon} alt="History" />
            </button>
          </div>
        </div>
      </div>
    </>
  );
}
