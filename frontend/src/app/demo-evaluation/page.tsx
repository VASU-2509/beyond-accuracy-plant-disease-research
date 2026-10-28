"use client";
import { useApi } from "@/hooks/use-api";
import { API_BASE } from "@/lib/api";
import type { Evaluation, Plot, ResultsResponse } from "@/lib/types";
import { PageHeading, Loading, ErrorState, EmptyState, Badge } from "@/components/ui/states";
import { label, percent, modelLabel } from "@/lib/formatters";

export default function DemoEvaluationPage() {
  const results = useApi<ResultsResponse>("/api/results/summary");
  const plots = useApi<{ plots: Plot[] }>("/api/results/plots");
  const evaluation = results.data?.results.find(
    (row: Evaluation) => row.purpose === "demo" && row.run_id === "demo"
  );
  const confusion = plots.data?.plots.find(
    plot => plot.run_id === "demo" && plot.name === "resnet18_in_domain_confusion"
  );
  const test = evaluation?.domains.in_domain;
  const external = evaluation?.domains.external;

  return <>
    <PageHeading eyebrow="COLLEGE DEMONSTRATION" title="Demo evaluation">
      A fixed, balanced PlantVillage test sample and a separate PlantDoc external sample. These demo results are not the full research evaluation.
    </PageHeading>
    {results.loading || plots.loading ? <Loading/> : results.error ? <ErrorState message={results.error} retry={results.retry}/> : plots.error ? <ErrorState message={plots.error} retry={plots.retry}/> : !evaluation ? <EmptyState command="python -m research.scripts.prepare_demo"/> : <>
      <section className="panel">
        <div className="panel-title"><h2>{modelLabel(evaluation.model_name)}</h2><Badge tone="warning">DEMO · not full research results</Badge></div>
        <div className="inline-metrics">
          <div><small>PlantVillage test accuracy</small><strong>{percent(test?.classification.accuracy)}</strong></div>
          <div><small>PlantVillage test images</small><strong>{test?.classification.n ?? "—"}</strong></div>
          <div><small>PlantDoc external accuracy</small><strong>{external ? percent(external.classification.accuracy) : "—"}</strong></div>
          <div><small>PlantDoc images · mapped classes</small><strong>{external ? `${external.classification.n} · ${external.classification.metric_classes?.length ?? "—"}` : "0"}</strong></div>
        </div>
        <p className="muted">{evaluation.classes.length} classes · ImageNet-pretrained checkpoint · 224×224 inputs · source checkpoint {evaluation.source_run_id ?? "not recorded"}</p>
        <p className="muted">PlantDoc is evaluated separately over its mapped classes and is never used for training or validation-only calibration.</p>
      </section>
      {confusion ? <section className="panel">
        <div className="panel-title"><h2>PlantVillage test confusion matrix</h2><Badge>Actual held-out demo test images</Badge></div>
        <p className="muted">Rows are true classes; columns are predicted classes. Counts come from the same {test?.classification.n ?? 0}-image test sample shown above.</p>
        <a className="figure-card" href={`${API_BASE}${confusion.url}`} target="_blank" rel="noreferrer">
          <img src={`${API_BASE}${confusion.url}`} alt={`Confusion matrix for ${test?.classification.n ?? 0} PlantVillage test images`}/>
          <small>Open full-resolution confusion matrix</small>
        </a>
      </section> : <EmptyState/>}
      <section className="notice disclaimer">
        <strong>Interpretation:</strong> The test sample is a fixed subset of the existing PlantVillage held-out split. The model was reused, not retrained for this demo. Confidence is not a guarantee of correctness, and these results do not establish field diagnostic performance.
      </section>
    </>}
  </>;
}
