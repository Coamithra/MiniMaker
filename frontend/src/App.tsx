import { useState } from "react";

function App() {
  const [result, setResult] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function handleGenerate() {
    setLoading(true);
    setResult(null);
    try {
      const res = await fetch("/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ description: "" }),
      });
      const data = await res.json();
      setResult(data.message);
    } catch {
      setResult("Error connecting to backend");
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

      {result && (
        <div style={{ marginTop: 24, padding: 16, background: "#f4f4f4", borderRadius: 4 }}>
          <strong>Result:</strong> {result}
        </div>
      )}
    </div>
  );
}

export default App;
