// Base URL points to the same origin (frontend server)
const BASE = import.meta.env.VITE_API_URL || "";

/**
 * Generic POST request supporting JSON or FormData
 * @param {string} path - API path (e.g., /analyze/text)
 * @param {object|FormData} data - Data to send
 * @param {number} timeout - Optional timeout in ms
 */
async function postForm(path, data, timeout = 30000) {
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

// --- API functions ---
// Prepend '/api' to match FastAPI routes
export function analyzeText(text) {
  const fd = new FormData();
  fd.append("text", text);
  return postForm("/api/analyze/text", fd);
}

export function analyzeImage(file) {
  const fd = new FormData();
  fd.append("file", file);
  return postForm("/api/analyze/image", fd);
}

export function analyzeDocument(file) {
  const fd = new FormData();
  fd.append("file", file);
  return postForm("/api/analyze/document", fd);
}

export default { analyzeText, analyzeImage, analyzeDocument };
