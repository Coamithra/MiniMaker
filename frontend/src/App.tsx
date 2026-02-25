import { useState } from "react";

function App() {
  const [description, setDescription] = useState("");
  const [modelData, setModelData] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState<string | null>(null);

  async function handleGenerateMiniature() {
    setLoading(true);
    setModelData(null);
    setError(null);
    setStatus("Generating your miniature... This takes ~2-3 minutes.");
    try {
      const res = await fetch("/generate-miniature", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ description }),
      });
      const data = await res.json();
      if (data.model) {
        setModelData(data.model);
        setStatus(null);
      } else {
        setError(data.message || "No 3D model returned from server.");
        setStatus(null);
      }
    } catch {
      setError("Error connecting to backend.");
      setStatus(null);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ maxWidth: 600, margin: "80px auto", fontFamily: "system-ui, sans-serif" }}>
      <h1>MiniMaker</h1>
      <p style={{ color: "#666" }}>Describe your D&amp;D character and generate a 3D-printable miniature.</p>

      <textarea
        value={description}
        onChange={(e) => setDescription(e.target.value)}
        placeholder="A fierce half-orc barbarian wielding a greataxe..."
        rows={4}
        style={{ width: "100%", boxSizing: "border-box", padding: 8, fontSize: 14 }}
      />

      <div style={{ marginTop: 12 }}>
        <button
          onClick={handleGenerateMiniature}
          disabled={!description.trim() || loading}
          style={{ padding: "8px 24px", fontSize: 14, cursor: "pointer" }}
        >
          {loading ? "Generating..." : "Generate Miniature"}
        </button>
      </div>

      {status && (
        <div style={{ marginTop: 24, padding: 16, color: "#1565c0", background: "#e3f2fd", borderRadius: 4 }}>
          {status}
        </div>
      )}

      {error && (
        <div style={{ marginTop: 24, padding: 16, color: "#d32f2f", background: "#fdecea", borderRadius: 4 }}>
          {error}
        </div>
      )}

      {modelData && (
        <div style={{ marginTop: 24, padding: 16, background: "#e8f5e9", borderRadius: 4 }}>
          <p style={{ margin: "0 0 8px", fontWeight: 600 }}>3D model ready!</p>
          <a
            href={`data:model/stl;base64,${modelData}`}
            download="miniature.stl"
            style={{ color: "#2e7d32", fontWeight: 500 }}
          >
            Download STL
          </a>
        </div>
      )}
    </div>
  );
}

export default App;
