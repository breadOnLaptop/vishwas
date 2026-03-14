import React from "react";
import "../index.css";

export default function Header() {
  return (
    <header style={{ textAlign: 'center', marginBottom: 20 }}>
      <h1 className="logo">VISHWAS</h1>
      <p style={{ color: 'var(--text-secondary)', fontSize: '1.1rem', marginTop: 12 }}>
        Truth verification at the speed of thought.
      </p>
    </header>
  );
}
