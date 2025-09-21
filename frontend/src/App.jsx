import React, { useState } from "react";
import Header from "./components/Header";
import InputForm from "./components/InputForm";
import AnalysisResult from "./components/AnalysisResult";
import Loader from "./components/Loader";
import "./index.css";
import api from "./api";

export default function App() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [bgClass, setBgClass] = useState("");

  const handleAnalysis = async (formData, type = "text") => {
    setLoading(true);
    setResult(null);
    try {
      const res = type === "image" ? await api.analyzeImage(formData) : await api.analyzeText(formData);
      setResult(res);

      // small themed background hint (keeps original behavior)
      const color = res?.color;
      if (color === "red") setBgClass("red-bg");
      else if (color === "orange") setBgClass("orange-bg");
      else if (color === "green") setBgClass("green-bg");
      else setBgClass("");
    } catch (err) {
      console.error("analysis error", err);
      setResult({ user_explanation: "Error querying server: " + (err.message || err) });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={`app-container ${bgClass}`} role="application">
      <Header />
      <main style={{ width: "100%", display: "flex", justifyContent: "center" }}>
        <div style={{ width: "100%", maxWidth: 1000 }}>
          <InputForm onAnalyze={handleAnalysis} />
          {loading ? <Loader /> : result && <AnalysisResult result={result} />}
        </div>
      </main>
    </div>
  );
}
