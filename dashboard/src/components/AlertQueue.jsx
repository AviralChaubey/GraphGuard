import React, { useState } from "react";

/**
 * Left-hand queue of ring alerts to triage.
 *
 * `ringRows` comes from GET /rings (App.jsx), so this always reflects whichever rings actually
 * exist on the server it's talking to right now, highest score first. The "jump to ring_id"
 * box below still lets an analyst open any ring directly by id, e.g. one from outside the
 * top-20 /rings returns by default.
 *
 * The "up to time step" box lets an analyst pick a manual cutoff and only see rings active at
 * or before it -- there's no live feed to read a current time from, and the underlying
 * datasets (Elliptic/PaySim) never published real dates, only an abstract step ordering (the
 * same one used for the train/test split), so this is the closest equivalent to "alerts before
 * time X" that the data actually supports.
 */
export default function AlertQueue({
  ringRows,
  selectedRingId,
  onSelect,
  queueError,
  beforeTimeStep,
  onBeforeTimeStepChange,
  maxTimeStep,
}) {
  const [draft, setDraft] = useState("");
  const [timeDraft, setTimeDraft] = useState(beforeTimeStep ?? "");

  function submitDraft(e) {
    e.preventDefault();
    if (draft.trim()) {
      onSelect(draft.trim());
      setDraft("");
    }
  }

  function submitTime(e) {
    e.preventDefault();
    if (timeDraft === "") {
      onBeforeTimeStepChange(null);
    } else {
      const n = Number(timeDraft);
      if (Number.isInteger(n)) onBeforeTimeStepChange(n);
    }
  }

  function clearTime() {
    setTimeDraft("");
    onBeforeTimeStepChange(null);
  }

  return (
    <div className="panel alert-queue">
      <div className="panel-header">Alert Queue</div>

      <form className="alert-queue__time" onSubmit={submitTime}>
        <label htmlFor="before-time-step">
          Show rings up to time step{maxTimeStep != null ? ` (1–${maxTimeStep})` : ""}
        </label>
        <div className="alert-queue__time-row">
          <input
            id="before-time-step"
            type="number"
            min={1}
            max={maxTimeStep ?? undefined}
            value={timeDraft}
            onChange={(e) => setTimeDraft(e.target.value)}
            placeholder={maxTimeStep != null ? `e.g. ${maxTimeStep}` : "e.g. 23"}
          />
          <button type="submit">Set</button>
          {beforeTimeStep != null && (
            <button type="button" onClick={clearTime}>
              Clear
            </button>
          )}
        </div>
        <p className="muted alert-queue__time-hint">
          No calendar date here -- this dataset's "time" is just a step number used to split
          training from testing, not a real clock time.
        </p>
      </form>

      {queueError && <p className="error alert-queue__error">{queueError}</p>}
      <ul className="alert-list">
        {ringRows.map((row) => (
          <li key={row.ring_id}>
            <button
              className={`alert-item ${row.ring_id === selectedRingId ? "alert-item--active" : ""}`}
              onClick={() => onSelect(row.ring_id)}
            >
              <span className="alert-item__id">{row.ring_id}</span>
              <span className="alert-item__hint">
                {row.time_step != null ? `t=${row.time_step}` : "open ring"}
              </span>
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
