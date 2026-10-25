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
const FALLBACK_ROWS = FALLBACK_RING_IDS.map((id) => ({ ring_id: id, time_step: null }));

export default function App() {
  const [ringRows, setRingRows] = useState(FALLBACK_ROWS);
  const ringIds = ringRows.map((r) => r.ring_id);
  const [selectedRingId, setSelectedRingId] = useState(null);
  const [ring, setRing] = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [queueError, setQueueError] = useState(null);
  // "Up to time step" cutoff -- there's no live feed here, and Elliptic/PaySim never
  // published real dates, only an abstract time-step ordering (1..49 for Elliptic) used
  // for the train/test split. null means "no cutoff, show everything". maxTimeStep comes
  // back from the server so the input's bounds always match whatever data it actually has.
  const [beforeTimeStep, setBeforeTimeStep] = useState(null);
  const [maxTimeStep, setMaxTimeStep] = useState(null);

  useEffect(() => {
    let cancelled = false;
    api
      .listRings(20, beforeTimeStep)
      .then((res) => {
        if (cancelled) return;
        setMaxTimeStep(res.max_time_step ?? null);
        if (res.rings.length > 0) {
          setRingRows(res.rings);
          setSelectedRingId(res.rings[0].ring_id);
          setQueueError(null);
        } else {
          setRingRows([]);
          setQueueError(
            beforeTimeStep != null
              ? `No rings at or before time step ${beforeTimeStep}.`
              : "Server has no rings above threshold right now."
          );
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
  }, [beforeTimeStep]);

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
          ringRows={ringRows}
          selectedRingId={selectedRingId}
          onSelect={setSelectedRingId}
          queueError={queueError}
          beforeTimeStep={beforeTimeStep}
          onBeforeTimeStepChange={setBeforeTimeStep}
          maxTimeStep={maxTimeStep}
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
