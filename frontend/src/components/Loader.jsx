import React from "react";
import "../index.css";

export default function Loader() {
  return (
    <div className="loader" role="status" aria-live="polite">
      <div className="spinner" aria-hidden="true" />
      <p className="small">Analyzing… this may take a few seconds</p>
    </div>
  );
}
