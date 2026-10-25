import React, { useEffect, useState } from "react";
import { api } from "./api.js";
import AlertQueue from "./components/AlertQueue.jsx";
import RingGraphView from "./components/RingGraphView.jsx";
import ExplanationPanel from "./components/ExplanationPanel.jsx";

// Seed queue -- see the note in AlertQueue.jsx about the missing "/rings" list endpoint.
// "r_00042" was the only entry here, which only exists against the DummyModel's synthetic
// stub ring -- it doesn't exist once a real checkpoint is loaded (GRAPHGUARD_CHECKPOINT set),
// so the demo alert queue was empty/broken against real data. Replaced with real, high-score
// ring ids pulled from the actual Elliptic checkpoint (results/checkpoints/elliptic/best.pt),
// highest fraud score first, so the queue has something real to click through for a demo.
const KNOWN_RING_IDS = [
  "r_06942", // score 0.998, size 11
  "r_06516", // score 0.997, size 10, chain
  "r_00323", // score 0.996, size 24
  "r_00132", // score 0.995, size 9, star
  "r_05772", // score 0.995, size 15
];

export default function App() {
  const [selectedRingId, setSelectedRingId] = useState(KNOWN_RING_IDS[0]);
  const [ring, setRing] = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!selectedRingId) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    setExplanation(null);

    Promise.all([api.getRing(selectedRingId), api.getExplanation(selectedRingId)])
      .then(([ringRes, explainRes]) => {
        if (cancelled) return;
        setRing(ringRes);
        setExplanation(explainRes);
      })
      .catch((err) => !cancelled && setError(err.message))
      .finally(() => !cancelled && setLoading(false));

    return () => {
      cancelled = true;
    };
  }, [selectedRingId]);

  const evidenceEdgeKeys = explanation ? explanation.evidence_edges.map(([s, d]) => `${s}-${d}`) : [];

  return (
    <div className="app">
      <header className="app-header">
        <span className="app-header__logo">GraphGuard</span>
        <span className="app-header__subtitle">Analyst Console</span>
      </header>

      <main className="app-body">
        <AlertQueue
          ringIds={KNOWN_RING_IDS}
          selectedRingId={selectedRingId}
          onSelect={setSelectedRingId}
        />
        <RingGraphView
          ring={ring}
          evidenceNodeIds={explanation?.evidence_nodes ?? []}
          evidenceEdgeKeys={evidenceEdgeKeys}
        />
        <ExplanationPanel
          explanation={explanation}
          loading={loading}
          error={error}
          ringId={selectedRingId}
          onFeedback={api.sendFeedback}
        />
      </main>
    </div>
  );
}
