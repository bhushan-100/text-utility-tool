import React, { useEffect, useState } from "react";

export default function History({ setText, setOutput }) {
  const [summaries, setSummaries] = useState([]);

  useEffect(() => {
    setSummaries(localStorage.getItem("summaries"));
  }, []);

  return (
    <div className="history">
      {summaries.map((summary) => (
        <button key={summary.id}>{summary.text}</button>
      ))}
    </div>
  );
}
