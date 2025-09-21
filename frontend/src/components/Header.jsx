import React from "react";
import "../index.css";

export default function Header() {
  return (
    <header className="header" role="banner" aria-label="App header">
      <h1 className="logo">Vishwas</h1>
      <p className="subtitle">Smart Misinformation Analyzer — paste text or upload a file to check trustworthiness</p>
    </header>
  );
}
