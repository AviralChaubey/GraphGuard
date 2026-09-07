import React, { useMemo, useRef } from "react";
import ForceGraph2D from "react-force-graph-2d";

/**
 * Renders a Ring's nodes/edges from GET /ring/{id}, highlighting whichever nodes/edges are
 * named as evidence by GET /explain/{id} (passed in as `evidenceNodeIds` / `evidenceEdgeKeys`).
 */
export default function RingGraphView({ ring, evidenceNodeIds = [], evidenceEdgeKeys = [] }) {
  const fgRef = useRef();
  const evidenceNodeSet = useMemo(() => new Set(evidenceNodeIds), [evidenceNodeIds]);
  const evidenceEdgeSet = useMemo(() => new Set(evidenceEdgeKeys), [evidenceEdgeKeys]);

  const graphData = useMemo(() => {
    if (!ring) return { nodes: [], links: [] };
    return {
      nodes: ring.nodes.map((n) => ({
        id: n.id,
        label: n.node_id,
        type: n.type,
        score: n.score,
        isEvidence: evidenceNodeSet.has(n.id),
      })),
      links: ring.edges.map((e) => ({
        source: e.src,
        target: e.dst,
        relation: e.relation,
        weight: e.weight,
        isEvidence: evidenceEdgeSet.has(`${e.src}-${e.dst}`) || evidenceEdgeSet.has(`${e.dst}-${e.src}`),
      })),
    };
  }, [ring, evidenceNodeSet, evidenceEdgeSet]);

  if (!ring) {
    return <div className="panel graph-panel graph-panel--empty">Select a ring to view its graph.</div>;
  }

  return (
    <div className="panel graph-panel">
      <div className="panel-header">
        Ring {ring.ring_id} · <span className="mono">{ring.pattern}</span> · score{" "}
        <span className="mono">{ring.score.toFixed(2)}</span>
      </div>
      <ForceGraph2D
        ref={fgRef}
        graphData={graphData}
        width={640}
        height={460}
        backgroundColor="rgba(0,0,0,0)"
        nodeRelSize={5}
        nodeColor={(n) => (n.isEvidence ? "#f2545b" : n.type === "device" || n.type === "address" ? "#5b8dee" : "#7c8798")}
        nodeLabel={(n) => `${n.label}\nscore ${n.score.toFixed(2)}`}
        linkColor={(l) => (l.isEvidence ? "#f2545b" : "rgba(124,135,152,0.35)")}
        linkWidth={(l) => (l.isEvidence ? 2.5 : 1)}
        linkDirectionalArrowLength={0}
        cooldownTicks={80}
      />
      <div className="graph-legend">
        <span><i className="dot dot--evidence" /> evidence</span>
        <span><i className="dot dot--entity" /> device / address</span>
        <span><i className="dot dot--account" /> account</span>
      </div>
    </div>
  );
}
