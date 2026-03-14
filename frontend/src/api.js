// Base URL points to the Go backend
const BASE = import.meta.env.VITE_API_URL || "http://localhost:8080";

/**
 * Generic POST request supporting JSON or FormData
 * @param {string} path - API path
 * @param {object|FormData} data - Data to send
 * @param {number} timeout - Increased to 120s for heavy RAG synthesis
 */
async function postForm(path, data, timeout = 120000) {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeout);

  const isJSON = !(data instanceof FormData);
  const options = {
    method: "POST",
    signal: controller.signal,
    headers: isJSON ? { "Content-Type": "application/json" } : undefined,
    body: isJSON ? JSON.stringify(data) : data,
  };

  try {
    const res = await fetch(`${BASE}${path}`, options);
    clearTimeout(id);

    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(`${res.status} ${res.statusText} ${text}`);
    }
    return await res.json();
  } finally {
    clearTimeout(id);
  }
}

export function analyzeText(text) {
  const fd = new FormData();
  fd.append("text", text);
  return postForm("/api/analyze/text", fd);
}

/**
 * Sends a file along with optional text context
 */
export function analyzeFile(file, text = "", type = "image") {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("text", text);
  const path = type === "document" ? "/api/analyze/document" : "/api/analyze/image";
  return postForm(path, fd);
}

export default { analyzeText, analyzeFile };
