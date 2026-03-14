import React, { useState } from "react";
import Header from "./components/Header";
import InputForm from "./components/InputForm";
import AnalysisResult from "./components/AnalysisResult";
import Loader from "./components/Loader";
import LoaderScreen from "./components/LoaderScreen";
import "./index.css";
import api from "./api";

export default function App() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [showSplash, setShowSplash] = useState(true);

  const handleAnalysis = async (input, type = "text", contextText = "") => {
    setLoading(true);
    setResult(null);
    try {
      let res;
      if (type === "image" || type === "document") {
        res = await api.analyzeFile(input, contextText, type);
      } else {
        res = await api.analyzeText(input);
      }
      setResult(res);
    } catch (err) {
      console.error("Analysis failed", err);
      setResult({ user_explanation: "The connection to the truth engine was interrupted." });
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      {showSplash && <LoaderScreen onLoaded={() => setShowSplash(false)} />}
      {!showSplash && (
        <div className="app-container" role="application">
          <Header />
          <main style={{ width: "100%" }}>
            <InputForm onAnalyze={handleAnalysis} />
            {loading ? <Loader /> : result && <AnalysisResult result={result} />}
          </main>
        </div>
      )}
    </>
  );
}
