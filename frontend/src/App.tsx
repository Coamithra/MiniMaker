import { useState } from "react";

function App() {
  const [imageData, setImageData] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const [modelData, setModelData] = useState<string | null>(null);
  const [loading3D, setLoading3D] = useState(false);

  async function handleGenerate() {
    setLoading(true);
    setImageData(null);
    setError(null);
    try {
      const res = await fetch("/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ description: "" }),
      });
      const data = await res.json();
      if (data.image) {
        setImageData(data.image);
      } else {
        setError("No image returned from server.");
      }
    } catch {
      setError("Error connecting to backend.");
    } finally {
      setLoading(false);
    }
  }

  async function handleGenerate3D() {
    setLoading3D(true);
    setModelData(null);
    setError(null);
    try {
      const res = await fetch("/generate-3d", { method: "POST" });
      const data = await res.json();
      if (data.model) {
        setModelData(data.model);
      } else {
        setError("No 3D model returned from server.");
      }
    } catch {
      setError("Error connecting to backend.");
    } finally {
      setLoading3D(false);
    }
  }

  return (
    <div style={{ maxWidth: 600, margin: "80px auto", fontFamily: "system-ui, sans-serif" }}>
      <h1>MiniMaker</h1>
      <p style={{ color: "#666" }}>Describe your D&amp;D character and generate a 3D-printable miniature.</p>

      <textarea
        disabled
        placeholder="A fierce half-orc barbarian wielding a greataxe..."
        rows={4}
        style={{ width: "100%", boxSizing: "border-box", padding: 8, fontSize: 14 }}
      />

      <div style={{ display: "flex", gap: 12, marginTop: 12 }}>
        <button
          onClick={handleGenerate}
          disabled={loading}
          style={{ padding: "8px 24px", fontSize: 14, cursor: "pointer" }}
        >
          {loading ? "Generating..." : "Generate Image"}
        </button>

        <button
          onClick={handleGenerate3D}
          disabled={loading3D}
          style={{ padding: "8px 24px", fontSize: 14, cursor: "pointer" }}
        >
          {loading3D ? "Generating 3D..." : "Generate 3D"}
        </button>
      </div>

      {error && (
        <div style={{ marginTop: 24, padding: 16, color: "#d32f2f", background: "#fdecea", borderRadius: 4 }}>
          {error}
        </div>
      )}

      {imageData && (
        <div style={{ marginTop: 24 }}>
          <img
            src={`data:image/png;base64,${imageData}`}
            alt="Generated D&D miniature"
            style={{ width: "100%", borderRadius: 4 }}
          />
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
