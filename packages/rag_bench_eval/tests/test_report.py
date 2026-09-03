import json

from rag_bench_eval.report import build_report


def _write_run(runs_dir, experiment, timestamp, ndcg):
    payload = {
        "experiment": experiment,
        "dataset": "toy",
        "timestamp": timestamp,
        "config": {},
        "metrics": {"ndcg@10": ndcg, "recall@100": 0.5, "mrr@10": 0.5, "precision@10": 0.5},
        "per_query": [{"query_id": "q1", "ndcg@10": ndcg, "latency_ms": 10, "retrieved": ["d1"]}],
        "llm_calls": 0,
    }
    ts = timestamp.replace(":", "-")
    (runs_dir / f"{experiment}_{ts}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_report_sorts_by_ndcg_descending(tmp_path):
    _write_run(tmp_path, "bm25", "2026-01-01T00:00:00+00:00", ndcg=0.30)
    _write_run(tmp_path, "dense", "2026-01-01T00:00:00+00:00", ndcg=0.34)

    report = build_report(tmp_path)
    lines = report.strip().splitlines()

    assert "dense" in lines[2]
    assert "bm25" in lines[3]


def test_report_keeps_only_latest_run_per_experiment(tmp_path):
    _write_run(tmp_path, "bm25", "2026-01-01T00:00:00+00:00", ndcg=0.10)
    _write_run(tmp_path, "bm25", "2026-01-02T00:00:00+00:00", ndcg=0.99)

    report = build_report(tmp_path)

    assert "0.9900" in report
    assert "0.1000" not in report


def test_report_empty_runs_dir_still_has_header(tmp_path):
    report = build_report(tmp_path)
    assert report.startswith("| experiment |")
