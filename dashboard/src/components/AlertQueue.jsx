import React, { useState } from "react";

/**
 * Left-hand queue of ring alerts to triage.
 *
 * NOTE: the frozen API contract (docs/CONTRACTS.md) has no "list all rings" endpoint yet --
 * only GET /ring/{id}. Until the team adds one, this queue is seeded with known ids and lets
 * the analyst jump to any ring_id directly. Raise a `/rings` list endpoint with the team if
 * the real alert volume needs paging.
 */
export default function AlertQueue({ ringIds, selectedRingId, onSelect }) {
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
