import csv
import json
import yaml
from src.config import Config
from src.analysis.plotting import aggregate


def test_numerical_failure_is_in_summary(tmp_path):
    run = tmp_path/"failed_run"
    run.mkdir()
    (run/"config.yaml").write_text(yaml.safe_dump(Config(model="structured").to_dict()))
    (run/"metadata.json").write_text(json.dumps({"parameters": 100}))
    (run/"failure.json").write_text(json.dumps({"error_type": "FloatingPointError", "message": "Nonfinite loss"}))
    rows = aggregate(tmp_path)
    assert len(rows) == 1 and rows[0]["status"] == "failed"
    saved = list(csv.DictReader((tmp_path/"summary/summary.csv").open()))
    assert saved[0]["error_type"] == "FloatingPointError"
