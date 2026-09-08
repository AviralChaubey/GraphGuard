import React, { useState } from "react";

export default function FeedbackButtons({ ringId, onSubmit }) {
  const [sent, setSent] = useState(null); // 'confirmed' | 'dismissed' | null
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState(false);

  async function send(verdict) {
    setBusy(true);
    try {
      await onSubmit(ringId, verdict, note || undefined);
      setSent(verdict);
    } finally {
      setBusy(false);
    }
  }

  if (sent) {
    return (
      <div className="feedback feedback--done">
        Marked <strong>{sent}</strong> for {ringId}. Saved to the feedback log —
        not yet wired into retraining.
      </div>
    );
  }

  return (
    <div className="feedback">
      <div className="subheader">Analyst verdict</div>
      <textarea
        placeholder="optional note for the record…"
        value={note}
        onChange={(e) => setNote(e.target.value)}
        rows={2}
      />
      <div className="feedback__buttons">
        <button className="btn btn--confirm" disabled={busy} onClick={() => send("confirmed")}>
          Confirm fraud ring
        </button>
        <button className="btn btn--dismiss" disabled={busy} onClick={() => send("dismissed")}>
          Dismiss as false positive
        </button>
      </div>
    </div>
  );
}
