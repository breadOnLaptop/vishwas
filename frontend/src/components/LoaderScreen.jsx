import React, { useEffect, useState } from "react";
import "../index.css";

export default function LoaderScreen({ onLoaded }) {
  const [dots, setDots] = useState(".");

  // Animate dots
  useEffect(() => {
    const interval = setInterval(() => {
      setDots(d => (d.length < 3 ? d + "." : "."));
    }, 500);
    return () => clearInterval(interval);
  }, []);

  // Auto fade out after 2.2s (adjustable)
  useEffect(() => {
    const timer = setTimeout(() => {
      onLoaded && onLoaded();
    }, 2200);
    return () => clearTimeout(timer);
  }, [onLoaded]);

  return (
    <div className="loader-screen">
      {/* Animated geometric shapes */}
      <div className="loader-shapes">
        <span className="circle c1" />
        <span className="circle c2" />
        <span className="circle c3" />
        <span className="circle c4" />
      </div>

      <h2 className="loader-title">Vishwas</h2>
      <p className="loader-subtitle">Smart Misinformation Analyzer{dots}</p>
    </div>
  );
}
