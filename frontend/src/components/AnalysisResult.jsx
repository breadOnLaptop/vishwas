import React, { useMemo } from "react";
import "../index.css";

export default function AnalysisResult({ result }) {
  const score = typeof result?.score === "number" ? result.score : 0.0;
  const color = result?.color || "red";
  const summary = result?.user_explanation || "No summary available.";
  const claims = result?.parsed?.claims || [];
  const factChecks = result?.fact_checks || [];
  const webEvidence = result?.web_evidence || [];
  const evidenceWeight = result?.evidence_weight || 0;

  // FIX: Explicitly handle 0.0 to prevent 50% default.
  // Correct Mapping: Score 0/10 -> 0%, Score 10/10 -> 100%
  const pct = useMemo(() => {
    return Math.max(0, Math.min(100, Math.round((score / 10) * 100)));
  }, [score]);

  const evidencePct = useMemo(() => {
    return Math.max(0, Math.min(100, Math.round((evidenceWeight / 10) * 100)));
  }, [evidenceWeight]);

  const getStatusColor = (c) => {
    if (c === "green") return "#10b981";
    if (c === "orange") return "#f59e0b";
    if (c === "red") return "#ef4444";
    return "#52525b";
  };

  return (
    <div className="verdict-shell">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 32 }}>
        <div>
          <h2 style={{ margin: 0, fontSize: "0.85rem", textTransform: "uppercase", letterSpacing: "0.15em", color: "var(--text-secondary)" }}>
            RAG-Augmented Analysis
          </h2>
          <div className="score-display" style={{ marginTop: 12 }}>
            <span className="big-score">{score.toFixed(1)}</span>
            <span className="score-total">/ 10</span>
          </div>
        </div>
        <div style={{ background: getStatusColor(color), color: "#000", padding: "6px 16px", borderRadius: "100px", fontWeight: 800, fontSize: "0.75rem" }}>
          {color.toUpperCase()}
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '30px', marginBottom: 32 }}>
        <div>
          <div className="small" style={{ marginBottom: 8 }}>Truthfulness Confidence</div>
          <div className="track-bar">
            <div className="track-fill" style={{ width: `${pct}%`, background: getStatusColor(color) }} />
          </div>
          <div style={{ fontWeight: 600, marginTop: 4 }}>{pct}%</div>
        </div>
        <div>
          <div className="small" style={{ marginBottom: 8 }}>Evidence Weight</div>
          <div className="track-bar">
            <div className="track-fill" style={{ width: `${evidencePct}%` }} />
          </div>
          <div style={{ fontWeight: 600, marginTop: 4 }}>{evidencePct}%</div>
        </div>
      </div>

      <p style={{ fontSize: "1.25rem", lineHeight: "1.6", color: "#fff", marginBottom: 40 }}>
        {summary}
      </p>

      {/* RAG Section: Official Fact Checks */}
      {factChecks.length > 0 && (
        <div style={{ marginBottom: 40, padding: '20px', background: 'rgba(16, 185, 129, 0.05)', borderRadius: '16px', border: '1px solid rgba(16, 185, 129, 0.1)' }}>
          <h3 style={{ fontSize: '0.9rem', textTransform: 'uppercase', color: '#10b981', marginBottom: 15, display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span>✅</span> Official Database Matches
          </h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
            {factChecks.map((f, i) => (
              <div key={i} style={{ fontSize: '0.95rem' }}>
                <div style={{ fontWeight: 600 }}>{f.reviewer} rated this as "{f.textual_rating}"</div>
                <div className="small" style={{ marginTop: 4, fontStyle: 'italic' }}>"{f.claim_text}"</div>
                <a href={f.url} target="_blank" rel="noreferrer" style={{ color: '#10b981', fontSize: '0.8rem', textDecoration: 'none', marginTop: 5, display: 'inline-block' }}>View Source ↗</a>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Web Evidence */}
      {webEvidence.length > 0 && (
        <div style={{ marginBottom: 40 }}>
          <h3 style={{ fontSize: '0.9rem', textTransform: 'uppercase', color: 'var(--text-secondary)', marginBottom: 15 }}>
            Web Context & Sources
          </h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            {webEvidence.map((w, i) => (
              <div key={i} style={{ padding: '12px', background: '#050505', borderRadius: '12px', border: '1px solid var(--border-subtle)' }}>
                <a href={w.link} target="_blank" rel="noreferrer" style={{ color: 'var(--text-primary)', textDecoration: 'none', fontWeight: 500 }}>{w.title}</a>
                <div className="small" style={{ marginTop: 4 }}>{w.source} — {w.snippet}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Factual Claims Breakdown */}
      {claims.length > 0 && (
        <div>
          <h3 style={{ fontSize: "0.9rem", color: "var(--text-secondary)", textTransform: "uppercase", marginBottom: 20 }}>
            Extracted Factual Claims
          </h3>
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            {claims.map((c, i) => {
              const cScore = typeof c.confidence_score === "number" ? c.confidence_score : (c.misp_confidence || 0) * 10;
              return (
                <div key={i} style={{ borderBottom: "1px solid var(--border-subtle)", paddingBottom: 16 }}>
                  <div style={{ display: "flex", gap: 12, alignItems: "flex-start" }}>
                    <div style={{ width: 8, height: 8, borderRadius: "50%", background: getStatusColor(cScore > 7 ? 'green' : (cScore > 4 ? 'orange' : 'red')), marginTop: 7 }} />
                    <div style={{ flex: 1 }}>
                      <div style={{ fontWeight: 500 }}>{c.text}</div>
                      <div className="small" style={{ marginTop: 4 }}>{c.short_reason}</div>
                    </div>
                    <div style={{ fontWeight: 700, fontSize: '0.9rem' }}>{cScore.toFixed(1)}</div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
