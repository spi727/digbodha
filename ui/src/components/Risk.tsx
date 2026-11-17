import type { Risk } from "../lib/data";
import { RISK_COLOR, RISK_COPY } from "../lib/format";

const GLYPH: Record<Risk, string> = { safe: "●", uncertain: "◌", caution: "▲", danger: "◆" };

export function RiskBadge({ risk, small }: { risk: Risk; small?: boolean }) {
  return (
    <span className={`badge badge--${risk} ${small ? "badge--sm" : ""}`} style={{ ["--risk" as string]: RISK_COLOR[risk] }}>
      <span className="badge__glyph" aria-hidden="true">{GLYPH[risk]}</span>
      {RISK_COPY[risk].name}
    </span>
  );
}

export function RiskLegend() {
  return (
    <div className="legend">
      {(["safe", "uncertain", "caution", "danger"] as Risk[]).map((r) => (
        <span key={r} title={RISK_COPY[r].rule}>
          <i className="swatch" style={{ background: RISK_COLOR[r] }} />
          {RISK_COPY[r].name}
        </span>
      ))}
    </div>
  );
}
