import React, { useEffect, useState } from "react";
import { api } from "./api.js";
import AlertQueue from "./components/AlertQueue.jsx";
import RingGraphView from "./components/RingGraphView.jsx";
import ExplanationPanel from "./components/ExplanationPanel.jsx";

// Seed queue -- see the note in AlertQueue.jsx about the missing "/rings" list endpoint.
const KNOWN_RING_IDS = ["r_00042"];

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
