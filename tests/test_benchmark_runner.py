from pathlib import Path

from filesystem_rag_mcp.benchmark import run_benchmark


def test_benchmark_runner_evaluation():
    dataset_path = Path(__file__).parent / "data" / "evaluation_dataset.json"
    assert dataset_path.exists()

    report = run_benchmark(dataset_path=dataset_path)
    assert report.modes is not None
    assert len(report.modes) == 4

    # Verify all 4 modes are present
    assert "Mode 1: Pure Lexical (FTS5)" in report.modes
    assert "Mode 2: Pure Dense Vector" in report.modes
    assert "Mode 3: Hybrid RRF" in report.modes
    assert "Mode 4: Hybrid + Cross-Encoder" in report.modes

    m4 = report.modes["Mode 4: Hybrid + Cross-Encoder"]
    assert m4.mrr_at_5 > 0.0
    assert m4.hit_at_1 > 0.0
    assert m4.avg_latency_ms >= 0.0
    assert len(report.gate_details) > 0
