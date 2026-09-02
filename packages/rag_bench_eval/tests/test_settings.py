from rag_bench_eval import settings


def test_nfcorpus_dir_under_data_beir():
    assert settings.NFCORPUS_DIR == settings.DATA_DIR / "beir" / "nfcorpus"


def test_runs_dir_under_results():
    assert settings.RUNS_DIR == settings.RESULTS_DIR / "runs"
