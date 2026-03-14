import React, { useState, useRef } from "react";

export default function GlassInput({ onAnalyze }) {
  const [text, setText] = useState("");
  const [file, setFile] = useState(null);
  const textareaRef = useRef();

  const submitText = (e) => {
    e?.preventDefault();
    if (!text.trim()) return;
    const fd = new FormData();
    fd.append("text", text);
    onAnalyze(fd, "text");
  };

  const submitFile = (e) => {
    e?.preventDefault();
    if (!file) return;
    const fd = new FormData();
    fd.append("file", file);
    onAnalyze(fd, "image");
  };

  return (
    <div className="w-full max-w-3xl mx-auto px-4">
      <form onSubmit={submitText} className="mb-4">
        <label htmlFor="glass-textarea" className="sr-only">Enter text to analyze</label>

        <div
          className="backdrop-blur-sm bg-white/10 border border-white/20 rounded-2xl p-4 shadow-md"
          style={{ boxShadow: "0 6px 30px rgba(0,0,0,0.12)", backdropFilter: "blur(8px)" }}
        >
          <textarea
            id="glass-textarea"
            ref={textareaRef}
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Paste text, a claim, or a URL to check…"
            className="w-full min-h-[120px] resize-y bg-transparent text-white placeholder-white/60 focus:outline-none"
            onKeyDown={(e) => {
              if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) submitText(e);
            }}
            aria-label="Text to analyze"
          />
          <div className="flex items-center gap-3 mt-3">
            <button
              type="submit"
              className="px-4 py-2 rounded-lg bg-white/10 hover:bg-white/20 border border-white/10"
            >
              Analyze (Ctrl/Cmd+Enter)
            </button>

            <label className="inline-flex items-center gap-2 px-3 py-2 rounded-lg cursor-pointer border border-white/10 bg-white/5">
              <input
                type="file"
                accept="image/*,.pdf,.docx,.txt"
                onChange={(e) => setFile(e.target.files?.[0] ?? null)}
                className="hidden"
              />
              <span className="text-sm">Upload file</span>
              {file ? <span className="ml-2 text-xs opacity-80">{file.name}</span> : null}
            </label>

            <button
              type="button"
              onClick={submitFile}
              disabled={!file}
              className="px-3 py-1 rounded-md bg-white/6 disabled:opacity-40"
            >
              Analyze File
            </button>

            <button
              type="button"
              onClick={() => { setText(""); setFile(null); if (textareaRef.current) textareaRef.current.focus(); }}
              className="ml-auto px-3 py-1 rounded-md bg-white/6"
            >
              Clear
            </button>
          </div>
        </div>
      </form>
    </div>
  );
}
