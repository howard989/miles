from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_rlix_drivers_forward_option_beta_env_var() -> None:
    for relpath in (
        "examples/rlix/run_miles_rlix.py",
        "examples/rlix/run_miles_dual.py",
    ):
        source = (REPO_ROOT / relpath).read_text(encoding="utf-8")
        assert (
            '"MILES_INIT_DEFER_ADD_WORKER"' in source
        ), f"{relpath} must forward MILES_INIT_DEFER_ADD_WORKER into runtime_env"


def test_sglang_actor_runtime_env_receives_option_beta_flag() -> None:
    source = (REPO_ROOT / "miles" / "ray" / "rollout.py").read_text(
        encoding="utf-8"
    )
    assert "MILES_INIT_DEFER_ADD_WORKER" in source
    assert (
        'env_vars["MILES_INIT_DEFER_ADD_WORKER"] = value' in source
    ), "SGLangEngine actors must receive the Option beta env flag"
