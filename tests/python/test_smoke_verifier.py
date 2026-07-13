from pathlib import Path

import pandas as pd
import pytest

from adaptforecast.verification import verify_smoke_artifact


def test_smoke_verifier_rejects_incomplete_contract(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "manifest.json").write_text(
        '{"status":"complete","data_sha256":"' + "a" * 64 + '"}', encoding="utf-8"
    )
    pd.DataFrame([{"model": "seasonal_naive", "variant": "sales_only"}]).to_csv(
        run / "metrics.csv", index=False
    )

    with pytest.raises(RuntimeError, match="Missing smoke models"):
        verify_smoke_artifact(tmp_path)
