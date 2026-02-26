import { useEffect, useRef, useState } from "react";
import { StlViewer } from "react-stl-viewer";

type AppState = "idle" | "generating_images" | "selecting" | "generating_model" | "viewing";

function App() {
  const [state, setState] = useState<AppState>("idle");
  const [description, setDescription] = useState("");
  const [images, setImages] = useState<string[]>([]);
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const [stlUrl, setStlUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [statusMsg, setStatusMsg] = useState<string | null>(null);
  const blobUrlRef = useRef<string | null>(null);

  // Clean up blob URL on unmount or reset
  useEffect(() => {
    return () => {
      if (blobUrlRef.current) URL.revokeObjectURL(blobUrlRef.current);
    };
  }, []);

  function reset() {
    if (blobUrlRef.current) {
      URL.revokeObjectURL(blobUrlRef.current);
      blobUrlRef.current = null;
    }
    setState("idle");
    setImages([]);
    setSelectedIndex(null);
    setStlUrl(null);
    setError(null);
    setStatusMsg(null);
  }

  async function handleGeneratePreviews() {
    setState("generating_images");
    setError(null);
    setStatusMsg("Generating 4 preview images... This takes ~2 minutes.");
    try {
      const res = await fetch("/generate-images", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ description }),
      });
      const data = await res.json();
      if (data.images && data.images.length > 0) {
        setImages(data.images);
        setSelectedIndex(null);
        setState("selecting");
        setStatusMsg(null);
      } else {
        setError(data.message || "No images returned from server.");
        setState("idle");
        setStatusMsg(null);
      }
    } catch {
      setError("Error connecting to backend.");
      setState("idle");
      setStatusMsg(null);
    }
  }

  async function handleCreateModel() {
    if (selectedIndex === null) return;
    const frontImage = images[selectedIndex];

    setState("generating_model");
    setError(null);
    setStatusMsg("Generating back view and 3D model... This takes ~3 minutes.");

    try {
      // Step 1: Generate back view
      const backRes = await fetch("/generate-back-view", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: frontImage, description }),
      });
      const backData = await backRes.json();
      if (!backData.image) {
        setError(backData.message || "Failed to generate back view.");
        setState("selecting");
        setStatusMsg(null);
        return;
      }

      setStatusMsg("Back view done. Generating 3D mesh...");

      // Step 2: Generate 3D model from front + back
      const modelRes = await fetch("/generate-model", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ front_image: frontImage, back_image: backData.image }),
      });
      const modelData = await modelRes.json();
      if (!modelData.model) {
        setError(modelData.message || "Failed to generate 3D model.");
        setState("selecting");
        setStatusMsg(null);
        return;
      }

      // Convert base64 STL to blob URL for the viewer
      const stlBytes = Uint8Array.from(atob(modelData.model), (c) => c.charCodeAt(0));
      const blob = new Blob([stlBytes], { type: "model/stl" });
      const url = URL.createObjectURL(blob);
      if (blobUrlRef.current) URL.revokeObjectURL(blobUrlRef.current);
      blobUrlRef.current = url;

      setStlUrl(url);
      setState("viewing");
      setStatusMsg(null);
    } catch {
      setError("Error connecting to backend.");
      setState("selecting");
      setStatusMsg(null);
    }
  }

  function handleDownload() {
    if (!stlUrl) return;
    const a = document.createElement("a");
    a.href = stlUrl;
    a.download = "miniature.stl";
    a.click();
  }

  return (
    <div style={{ maxWidth: 700, margin: "60px auto", fontFamily: "system-ui, sans-serif", padding: "0 16px" }}>
      <h1>MiniMaker</h1>
      <p style={{ color: "#666" }}>
        Describe your D&amp;D character and generate a 3D-printable miniature.
      </p>

      {/* Description input — visible in idle and generating_images */}
      {(state === "idle" || state === "generating_images") && (
        <>
          <textarea
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            placeholder="A fierce half-orc barbarian wielding a greataxe..."
            rows={4}
            disabled={state === "generating_images"}
            style={{ width: "100%", boxSizing: "border-box", padding: 8, fontSize: 14 }}
          />
          <div style={{ marginTop: 12 }}>
            <button
              onClick={handleGeneratePreviews}
              disabled={!description.trim() || state === "generating_images"}
              style={{ padding: "8px 24px", fontSize: 14, cursor: "pointer" }}
            >
              {state === "generating_images" ? "Generating..." : "Generate Previews"}
            </button>
          </div>
        </>
      )}

      {/* Image selection grid */}
      {state === "selecting" && (
        <>
          <p style={{ fontWeight: 600, marginBottom: 8 }}>Pick your favorite:</p>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
            {images.map((img, i) => (
              <img
                key={i}
                src={`data:image/png;base64,${img}`}
                alt={`Preview ${i + 1}`}
                onClick={() => setSelectedIndex(i)}
                style={{
                  width: "100%",
                  borderRadius: 8,
                  cursor: "pointer",
                  border: selectedIndex === i ? "3px solid #1565c0" : "3px solid transparent",
                  boxSizing: "border-box",
                }}
              />
            ))}
          </div>
          <div style={{ marginTop: 16, display: "flex", alignItems: "center", gap: 16 }}>
            <button
              onClick={handleCreateModel}
              disabled={selectedIndex === null}
              style={{ padding: "8px 24px", fontSize: 14, cursor: "pointer" }}
            >
              Create 3D Model
            </button>
            <a
              href="#"
              onClick={(e) => { e.preventDefault(); handleGeneratePreviews(); }}
              style={{ color: "#1565c0", fontSize: 14 }}
            >
              Regenerate
            </a>
          </div>
        </>
      )}

      {/* Generating model — show selected image + spinner */}
      {state === "generating_model" && selectedIndex !== null && (
        <div style={{ textAlign: "center" }}>
          <img
            src={`data:image/png;base64,${images[selectedIndex]}`}
            alt="Selected preview"
            style={{ maxWidth: 300, borderRadius: 8, marginBottom: 16 }}
          />
        </div>
      )}

      {/* 3D viewer */}
      {state === "viewing" && stlUrl && (
        <>
          <div style={{ width: "100%", height: 500, background: "#f5f5f5", borderRadius: 8, overflow: "hidden" }}>
            <StlViewer
              url={stlUrl}
              orbitControls
              shadows
              style={{ width: "100%", height: "100%" }}
              modelProps={{ color: "#7b8cde" }}
            />
          </div>
          <div style={{ marginTop: 16, display: "flex", gap: 16 }}>
            <button
              onClick={handleDownload}
              style={{ padding: "8px 24px", fontSize: 14, cursor: "pointer" }}
            >
              Download STL
            </button>
            <button
              onClick={reset}
              style={{ padding: "8px 24px", fontSize: 14, cursor: "pointer" }}
            >
              Start Over
            </button>
          </div>
        </>
      )}

      {/* Status message */}
      {statusMsg && (
        <div style={{ marginTop: 24, padding: 16, color: "#1565c0", background: "#e3f2fd", borderRadius: 4 }}>
          {statusMsg}
        </div>
      )}

      {/* Error message */}
      {error && (
        <div style={{ marginTop: 24, padding: 16, color: "#d32f2f", background: "#fdecea", borderRadius: 4 }}>
          {error}
        </div>
      )}
    </div>
  );
}

export default App;
