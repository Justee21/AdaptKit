import pytest

from examples.configuration import required_confidence_threshold


@pytest.mark.parametrize(
    "value",
    [None, "", " ", "insert-calibrated-threshold", "zero", "0", "-0.1", "1.1", "nan", "inf"],
)
def test_required_threshold_rejects_missing_or_invalid_values(value: str | None) -> None:
    environ = {} if value is None else {"THRESHOLD": value}
    with pytest.raises(RuntimeError, match="THRESHOLD"):
        required_confidence_threshold("THRESHOLD", environ=environ)


@pytest.mark.parametrize("value", ["0.01", "0.70", "1"])
def test_required_threshold_accepts_explicit_finite_values(value: str) -> None:
    assert required_confidence_threshold(
        "THRESHOLD", environ={"THRESHOLD": value}
    ) == float(value)
