import React from "react";
import FeedbackButtons from "./FeedbackButtons.jsx";

export default function ExplanationPanel({ explanation, loading, error, ringId, onFeedback }) {
  return (
    <div className="panel explanation-panel">
      <div className="panel-header">Explanation</div>

      {loading && <p className="muted">Running GNNExplainer…</p>}
      {error && <p className="error">Couldn't load explanation: {error}</p>}

      {explanation && !loading && !error && (
        <>
          <p className="reason-text">{explanation.reason}</p>

          <dl className="metric-list">
            <div>
              <dt>Fidelity</dt>
              <dd className="mono">{explanation.fidelity.toFixed(2)}</dd>
            </div>
            <div>
              <dt>Sparsity</dt>
              <dd className="mono">{explanation.sparsity.toFixed(2)}</dd>
            </div>
            <div>
              <dt>Evidence nodes</dt>
              <dd className="mono">{explanation.evidence_nodes.length}</dd>
            </div>
          </dl>

          {explanation.top_features.length > 0 && (
            <>
              <div className="subheader">Top contributing features</div>
              <p className="muted feature-list__caption">
                Shown as feature index (e.g. "f89"), not a real name — Elliptic's public
                dataset never published what each of its 165 features actually measures, so
                no name exists to show here. This is a limitation of the published dataset
                itself, not something missing from the explanation.
              </p>
              <ul className="feature-list">
                {explanation.top_features.map((f) => (
                  <li key={f.name}>
                    <span className="mono">{f.name}</span>
                    <div className="bar-track">
                      <div className="bar-fill" style={{ width: `${Math.min(100, f.importance * 100)}%` }} />
                    </div>
                    <span className="mono">{f.importance.toFixed(2)}</span>
                  </li>
                ))}
              </ul>
            </>
          )}

          <FeedbackButtons ringId={ringId} onSubmit={onFeedback} />
        </>
      )}
    </div>
  );
}
