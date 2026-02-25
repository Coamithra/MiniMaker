import { useState } from "react";

function App() {
  const [imageData, setImageData] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

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

      <button
        onClick={handleGenerate}
        disabled={loading}
        style={{ marginTop: 12, padding: "8px 24px", fontSize: 14, cursor: "pointer" }}
      >
        {loading ? "Generating..." : "Generate"}
      </button>

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
    </div>
  );
}

export default App;
