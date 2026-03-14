import React, { useState, useRef } from "react";
import "../index.css";

export default function InputForm({ onAnalyze }) {
  const [activeTab, setActiveTab] = useState("text"); // 'text', 'media', 'document'
  const [text, setText] = useState("");
  const [file, setFile] = useState(null);
  const fileRef = useRef(null);

  const handleSubmit = (e) => {
    e?.preventDefault();
    if (activeTab === "text") {
      if (!text.trim()) return;
      onAnalyze(text.trim(), "text");
    } else {
      if (!file) return;
      onAnalyze(file, activeTab === "document" ? "document" : "image", text.trim());
    }
  };

  const handleFileChange = (e) => {
    const selected = e.target.files?.[0];
    if (selected) setFile(selected);
  };

  return (
    <div className="input-container">
      <div className="tabs-container">
        <button 
          className={`tab-trigger ${activeTab === "text" ? "active" : ""}`}
          onClick={() => setActiveTab("text")}
        >
          Text
        </button>
        <button 
          className={`tab-trigger ${activeTab === "media" ? "active" : ""}`}
          onClick={() => setActiveTab("media")}
        >
          Images & Video
        </button>
        <button 
          className={`tab-trigger ${activeTab === "document" ? "active" : ""}`}
          onClick={() => setActiveTab("document")}
        >
          Documents
        </button>
      </div>

      <form onSubmit={handleSubmit} className="input-portal">
        <textarea
          className="portal-textarea"
          placeholder={activeTab === "text" ? "Enter a claim or paste content..." : "Add context about this file..."}
          value={text}
          onChange={(e) => setText(e.target.value)}
        />

        {activeTab !== "text" && (
          <div 
            className="file-drop-area" 
            onClick={() => fileRef.current?.click()}
            style={{
              padding: '20px',
              border: '1px dashed var(--border-subtle)',
              borderRadius: '16px',
              textAlign: 'center',
              marginTop: '10px',
              cursor: 'pointer',
              background: file ? 'rgba(255,255,255,0.02)' : 'transparent'
            }}
          >
            <input type="file" ref={fileRef} hidden onChange={handleFileChange} />
            <span style={{ color: file ? 'var(--text-primary)' : 'var(--text-secondary)' }}>
              {file ? `📎 ${file.name}` : 'Click to attach file'}
            </span>
          </div>
        )}

        <div className="portal-actions">
          <button 
            type="button" 
            className="secondary-action"
            onClick={() => { setText(""); setFile(null); }}
          >
            Clear
          </button>
          <button 
            type="submit" 
            className="action-btn"
            disabled={activeTab === "text" ? !text.trim() : !file}
          >
            Analyze
          </button>
        </div>
      </form>
    </div>
  );
}
