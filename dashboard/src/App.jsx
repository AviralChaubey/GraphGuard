import React, { useEffect, useState } from "react";
import { api } from "./api.js";
import AlertQueue from "./components/AlertQueue.jsx";
import RingGraphView from "./components/RingGraphView.jsx";
import ExplanationPanel from "./components/ExplanationPanel.jsx";

// Fallback only -- used if GET /rings itself can't be reached (server down, CORS, etc), so the
// queue isn't completely empty while you're debugging that. NOT trustworthy as real ring ids:
// ring extraction's floating-point scores (and therefore which rings exist and what they're
// named) are not guaranteed bit-identical across different machines/OS/PyTorch builds, so an id
// that's real on one machine can be genuinely absent on another with the same checkpoint and
// code. The actual queue is populated from GET /rings below, which asks the server you're
// talking to right now what really exists there -- always prefer that over any hardcoded list.
const FALLBACK_RING_IDS = ["r_00042"];

export default function App() {
  const [ringIds, setRingIds] = useState(FALLBACK_RING_IDS);
  const [selectedRingId, setSelectedRingId] = useState(null);
  const [ring, setRing] = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [queueError, setQueueError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    api
      .listRings()
      .then((res) => {
        if (cancelled) return;
        const ids = res.rings.map((r) => r.ring_id);
        if (ids.length > 0) {
          setRingIds(ids);
          setSelectedRingId(ids[0]);
        } else {
          setQueueError("Server has no rings above threshold right now.");
        }
      })
      .catch((err) => {
        if (cancelled) return;
        setQueueError(`Couldn't load ring list: ${err.message}`);
        setSelectedRingId(FALLBACK_RING_IDS[0]);
      });
    return () => {
      cancelled = true;
    };
  }, []);

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
          ringIds={ringIds}
          selectedRingId={selectedRingId}
          onSelect={setSelectedRingId}
          queueError={queueError}
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
