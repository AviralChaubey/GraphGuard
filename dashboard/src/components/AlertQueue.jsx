import React, { useState } from "react";

/**
 * Left-hand queue of ring alerts to triage.
 *
 * `ringIds` comes from GET /rings (App.jsx), so this always reflects whichever rings actually
 * exist on the server it's talking to right now, highest score first. The "jump to ring_id"
 * box below still lets an analyst open any ring directly by id, e.g. one from outside the
 * top-20 /rings returns by default.
 */
export default function AlertQueue({ ringIds, selectedRingId, onSelect, queueError }) {
  const [draft, setDraft] = useState("");

  function submitDraft(e) {
    e.preventDefault();
    if (draft.trim()) {
      onSelect(draft.trim());
      setDraft("");
    }
  }

  return (
    <div className="panel alert-queue">
      <div className="panel-header">Alert Queue</div>
      {queueError && <p className="error alert-queue__error">{queueError}</p>}
      <ul className="alert-list">
        {ringIds.map((id) => (
          <li key={id}>
            <button
              className={`alert-item ${id === selectedRingId ? "alert-item--active" : ""}`}
              onClick={() => onSelect(id)}
            >
              <span className="alert-item__id">{id}</span>
              <span className="alert-item__hint">open ring</span>
            </button>
          </li>
        ))}
      </ul>
      <form className="alert-queue__jump" onSubmit={submitDraft}>
        <input
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          placeholder="jump to ring_id…"
          aria-label="Jump to ring id"
        />
        <button type="submit">Go</button>
      </form>
    </div>
  );
}
