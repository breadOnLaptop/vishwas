import React, { useState, useRef, useEffect } from "react";
import "../index.css";

export default function InputForm({ onAnalyze }) {
  const [text, setText] = useState("");
  const [file, setFile] = useState(null);
  const taRef = useRef(null);
  const fileRef = useRef(null);

  useEffect(() => {
    const onKey = (e) => {
      if (e.key === "Escape") {
        setText(""); setFile(null);
        if (fileRef.current) fileRef.current.value = "";
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const submitText = async (e) => {
    e?.preventDefault();
    if (!text.trim()) return;
    const fd = new FormData();
    fd.append("text", text.trim());
    await onAnalyze(fd, "text");
  };

  const submitFile = async (e) => {
    e?.preventDefault();
    if (!file) return;
    const fd = new FormData();
    fd.append("file", file);
    await onAnalyze(fd, "image");
  };

  return (
    <div className="card">
      <form className="glass" onSubmit={submitText} aria-label="Analyze form">
        <label className="input-label" htmlFor="glass-textarea">Enter text to analyze</label>

        <textarea
          id="glass-textarea"
          ref={taRef}
          className="glass-textarea"
          placeholder="Paste text, a claim, or a URL to check… (Ctrl/Cmd+Enter to analyze)"
          value={text}
          onChange={(e)=>setText(e.target.value)}
          onKeyDown={(e)=>{ if ((e.ctrlKey||e.metaKey) && e.key === "Enter") submitText(e); }}
          aria-label="Text to analyze"
        />

        <div className="actions-row" style={{ marginTop: 12 }}>
          <button type="submit" className="btn primary" title="Analyze (Ctrl/Cmd+Enter)">
            Analyze (Ctrl/Cmd+Enter)
          </button>

          <div className="file-input" style={{ marginLeft: 8 }}>
            <label className="btn ghost" htmlFor="file-upload" style={{ cursor: "pointer" }}>Choose file</label>
            <input
              id="file-upload"
              ref={fileRef}
              type="file"
              accept="image/*,.pdf,.docx,.txt"
              onChange={(e)=>setFile(e.target.files?.[0] ?? null)}
            />
            <div className="file-name">{ file ? file.name : <span className="helper">No file chosen</span> }</div>
          </div>

          <button
            type="button"
            className="btn"
            onClick={submitFile}
            disabled={!file}
            aria-disabled={!file}
            style={{ opacity: file ? 1 : 0.6 }}
            title={file ? "Analyze selected file" : "Choose a file first"}
          >
            Analyze File
          </button>

          <div className="actions-right">
            <button
              type="button"
              className="btn"
              onClick={() => { setText(""); setFile(null); if (fileRef.current) fileRef.current.value = ""; taRef.current?.focus(); }}
              title="Clear"
            >
              Clear
            </button>
          </div>
        </div>
      </form>
    </div>
  );
}
