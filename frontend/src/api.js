const BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000/api";

async function postForm(path, formData, timeout = 30000) {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeout);
  try {
    const res = await fetch(`${BASE}${path}`, { method: "POST", body: formData, signal: controller.signal });
    clearTimeout(id);
    if (!res.ok) {
      const t = await res.text().catch(()=>"");
      throw new Error(`${res.status} ${res.statusText} ${t}`);
    }
    return await res.json();
  } finally { clearTimeout(id); }
}

export function analyzeText(fd){ return postForm("/analyze/text", fd); }
export function analyzeImage(fd){ return postForm("/analyze/image", fd); }
export default { analyzeText, analyzeImage };
