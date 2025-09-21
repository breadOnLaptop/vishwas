import React, { useMemo, useState } from "react";
import "../index.css";

export default function AnalysisResult({ result }) {
  const [showSources, setShowSources] = useState(false);

  const score = result?.score ?? null;
  const color = result?.color ?? "unknown";
  const summary = result?.user_explanation || result?.explanation || "No summary available.";
  const topSources = Array.isArray(result?.top_sources) ? result.top_sources : [];
  const claims = result?.parsed?.claims || [];

  // compute a percentage for the confidence bar (score is 0..10)
  const pct = useMemo(() => {
    const s = typeof score === "number" ? Math.max(0, Math.min(10, score)) : 5;
    return Math.round((s / 10) * 100);
  }, [score]);

  const badgeClass = color === "green" ? "green" : color === "orange" ? "orange" : color === "red" ? "red" : "unknown";

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(summary);
      alert("Summary copied to clipboard");
    } catch {
      alert("Copy failed — use your browser to copy manually");
    }
  };

  return (
    <div className="result-card" role="region" aria-label="analysis-result">
      <div className="result-row">
        <div style={{ flex: 1 }}>
          <h2 style={{ margin: "0 0 6px 0" }}>Analysis result</h2>

          <div style={{ display: "flex", gap: 12, alignItems: "center", marginBottom: 8 }}>
            <div style={{ display:"flex", gap:8, alignItems:"center" }}>
              <div className={`badge ${badgeClass}`} aria-hidden="true">{badgeClass}</div>
              <div className="small"><strong>Score:</strong> {score !== null ? `${score}/10` : "—"}</div>
            </div>

            <div style={{ marginLeft: 6 }}>
              <div className="confidence" aria-hidden="true">
                <div className="conf-bar" style={{ position: "relative" }}>
                  <div className="conf-dot" style={{ left: `calc(${pct}% )` }} />
                </div>
                <div className="small">{pct}%</div>
              </div>
            </div>
          </div>

          <p style={{ marginTop: 6 }}>
            <strong>Summary:</strong> {summary}
          </p>

          <div style={{ marginTop: 8, display: "flex", gap: 8 }}>
            <button className="btn" onClick={handleCopy}>Copy summary</button>
            <button className="btn" onClick={() => setShowSources(s=>!s)}>{showSources ? "Hide sources" : `Sources (${topSources.length})`}</button>
          </div>
        </div>
      </div>

      {/* claims */}
      {claims.length > 0 && (
        <section style={{ marginTop: 14 }}>
          <h3 style={{ marginTop: 0 }}>Claims ({claims.length})</h3>
          <ul style={{ paddingLeft: 14 }}>
            {claims.map((c, i) => (
              <li key={i} style={{ marginBottom: 12 }}>
                <div><strong>Claim:</strong> {c.text}</div>
                <div className="small"><strong>Confidence:</strong> {typeof c.confidence_0_10 === "number" ? `${c.confidence_0_10}/10` : `${(c.misp_confidence_0_1||0)*10}/10`}</div>
                <div className="small"><strong>Reason:</strong> {c.short_reason || "—"}</div>

                {Array.isArray(c.references) && c.references.length > 0 && (
                  <div style={{ marginTop: 6 }}>
                    <strong>Sources:</strong>
                    <ul style={{ marginTop:6 }}>
                      {c.references.map((r, j) => (
                        <li key={j} className="source-item">
                          <a href={r.link || "#"} target="_blank" rel="noreferrer">{r.title || r.link}</a>
                          {r.publisher ? <span style={{ marginLeft:8 }} className="small">— {r.publisher}</span> : null}
                          {r.snippet ? <div className="small">{r.snippet}</div> : null}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </li>
            ))}
          </ul>
        </section>
      )}

      {/* optional top sources */}
      {showSources && topSources.length > 0 && (
        <div className="sources-list" style={{ marginTop: 12 }}>
          <h4 style={{ marginTop:0 }}>Top sources</h4>
          <div>
            {topSources.map((s, i) => (
              <div key={i} className="source-item">
                <a href={s.link || "#"} target="_blank" rel="noreferrer">{s.title || s.link}</a>
                {s.publisher ? <span className="small"> — {s.publisher}</span> : null}
                {s.snippet ? <div className="small">{s.snippet}</div> : null}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
