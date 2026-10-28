"use client";
import { useApi } from "@/hooks/use-api";
import type { ResultsResponse } from "@/lib/types";
import { modelLabel, percent, number } from "@/lib/formatters";
import { PageHeading, Loading, ErrorState, EmptyState, Badge } from "@/components/ui/states";
import { ComparisonChart } from "@/components/charts/research-charts";

export default function ComparisonPage() {
  const api = useApi<ResultsResponse>("/api/results/comparison");
  const results = api.data?.results ?? [];
  const hasDemo = results.some(result => result.purpose === "demo");
  const hasNonResearchSmoke = results.some(result => result.purpose === "smoke" || result.purpose === "synthetic");

  return <>
    <PageHeading eyebrow="MODEL COMPARISON" title="Compare performance. Consider the cost.">
      Compare saved model runs when available. Domain gaps use shared ground-truth classes and retain the full classifier output.
    </PageHeading>
    {api.loading ? <Loading/> : api.error ? <ErrorState message={api.error} retry={api.retry}/> : !results.length ? <EmptyState/> : <>
      {hasNonResearchSmoke && <div className="notice warning">Smoke-test and synthetic outputs verify the pipeline; they are not final experimental findings.</div>}
      {hasDemo && <div className="notice warning">Teacher-demo metrics are separate from full research results. Interpret them using their recorded dataset sizes and protocol.</div>}
      <section className="panel"><h2>Accuracy across domains</h2><ComparisonChart results={results}/></section>
      <section className="panel table-panel">
        <h2>Classification and calibration</h2>
        <div className="table-scroll"><table><thead><tr><th>Model / run</th><th>Domain</th><th>Images</th><th>Accuracy</th><th>Macro F1</th><th>ECE raw / calibrated</th><th>Brier raw / calibrated</th><th>Coverage</th><th>Accepted accuracy</th></tr></thead>
          <tbody>{results.flatMap(result => Object.entries(result.domains).map(([domain, data]) => <tr key={`${result.id}-${domain}`}>
            <td><strong>{modelLabel(result.model_name)}</strong><small>{result.run_id} · {result.purpose}</small></td>
            <td>{domain.replaceAll("_", " ")}</td><td>{data.classification.n}</td><td>{percent(data.classification.accuracy)}</td>
            <td>{number(data.classification.macro_f1, 3)}</td><td>{number(data.raw.ece, 3)} / {number(data.calibrated.ece, 3)}</td>
            <td>{number(data.raw.brier, 3)} / {number(data.calibrated.brier, 3)}</td><td>{percent(data.selective.coverage)}</td><td>{percent(data.selective.accuracy)}</td>
          </tr>))}</tbody>
        </table></div>
      </section>
      <div className="grid-two">{results.map(result => <section className="panel" key={result.id}>
        <div className="panel-title"><h3>{modelLabel(result.model_name)}</h3><Badge>{result.efficiency.device}</Badge></div>
        <p className="muted">{result.run_id} · {result.purpose}</p>
        <dl className="definition-list">
          <div><dt>Parameters</dt><dd>{number(result.efficiency.parameters, 0)}</dd></div>
          <div><dt>Checkpoint size</dt><dd>{number(result.efficiency.checkpoint_mb)} MiB</dd></div>
          <div><dt>Median / p95 latency</dt><dd>{number(result.efficiency.latency_ms_median)} / {number(result.efficiency.latency_ms_p95)} ms</dd></div>
          <div><dt>Shared-class accuracy drop</dt><dd>{result.domain_gap ? `${number(result.domain_gap.accuracy * 100)} pp` : "Pending external evaluation"}</dd></div>
          <div><dt>Shared-class macro F1 drop</dt><dd>{result.domain_gap ? number(result.domain_gap.macro_f1, 3) : "—"}</dd></div>
        </dl>
        <p className="footnote muted">{result.efficiency.protocol}</p>
      </section>)}</div>
      <p className="muted footnote">Different class frequencies and image conditions confound aggregate comparisons. No statistical significance is claimed. Zero accepted predictions have undefined selective risk and accuracy (—).</p>
    </>}
  </>;
}
