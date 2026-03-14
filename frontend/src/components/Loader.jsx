import React, { useState, useEffect } from "react";
import "../index.css";

const MESSAGES = [
  "Analyzing content integrity...",
  "Cross-referencing global data...",
  "Running AI factual scan...",
  "Evaluating technical context...",
  "Finalizing trust report..."
];

export default function Loader() {
  const [msgIdx, setMsgIdx] = useState(0);

  useEffect(() => {
    const t = setInterval(() => setMsgIdx(i => (i + 1) % MESSAGES.length), 2000);
    return () => clearInterval(t);
  }, []);

  return (
    <div className="ai-loader">
      <div className="shimmer-circle" />
      <div className="status-text">{MESSAGES[msgIdx]}</div>
    </div>
  );
}
