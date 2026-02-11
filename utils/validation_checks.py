from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from pyspark.sql import DataFrame, functions as F


# -----------------------------
# Core types
# -----------------------------

class Severity(str, Enum):
    ERROR = "ERROR"
    WARN = "WARN"
    INFO = "INFO"


@dataclass(frozen=True)
class CheckResult:
    name: str
    severity: Severity
    passed: bool
    message: str
    metrics: Optional[Dict[str, float]] = None


class QualityCheckError(RuntimeError):
    """Raised when an ERROR-level check fails."""
    pass


# -----------------------------
# Logging / enforcement helpers
# -----------------------------

def emit(result: CheckResult) -> None:
    """
    Print a check result and raise if it's an ERROR failure.
    Keep this simple and consistent; it becomes your run audit trail.
    """
    prefix = f"[{result.severity}] {result.name}"
    print(f"{prefix} | passed={result.passed} | {result.message}")
    if result.metrics:
        print(f"{prefix} | metrics={result.metrics}")

    if result.severity == Severity.ERROR and not result.passed:
        raise QualityCheckError(f"{prefix} FAILED: {result.message}")


def emit_many(results: Iterable[CheckResult]) -> None:
    for r in results:
        emit(r)


# -----------------------------
# Basic utility functions
# -----------------------------

def row_count(df: DataFrame, name: str) -> CheckResult:
    c = float(df.count())
    return CheckResult(
        name=f"{name}__row_count",
        severity=Severity.INFO,
        passed=True,
        message=f"Row count for {name}: {int(c)}",
        metrics={"rows": c},
    )


def required_columns(df: DataFrame, required: Sequence[str], name: str) -> CheckResult:
    missing = [c for c in required if c not in df.columns]
    passed = (len(missing) == 0)
    msg = "All required columns present." if passed else f"Missing columns: {missing}"
    return CheckResult(
        name=f"{name}__required_columns",
        severity=Severity.ERROR,
        passed=passed,
        message=msg,
    )


def required_non_null(df: DataFrame, required: Sequence[str], name: str) -> CheckResult:
    null_counts: Dict[str, float] = {}
    for c in required:
        if c not in df.columns:
            null_counts[c] = float("nan")
            continue
        null_counts[c] = float(df.filter(F.col(c).isNull()).count())

    passed = all((c in df.columns and null_counts[c] == 0.0) for c in required)
    msg = "No nulls in required columns." if passed else f"Null counts: {null_counts}"
    return CheckResult(
        name=f"{name}__required_non_null",
        severity=Severity.ERROR,
        passed=passed,
        message=msg,
        metrics=null_counts,
    )


def warn_duplicates(df: DataFrame, key_cols: Sequence[str], name: str) -> CheckResult:
    """
    Bronze: duplicates are allowed, but we log how many duplicate key groups exist.
    """
    for c in key_cols:
        if c not in df.columns:
            return CheckResult(
                name=f"{name}__duplicates_warn",
                severity=Severity.WARN,
                passed=True,
                message=f"Key column '{c}' missing; cannot compute duplicates.",
            )

    dup_groups = (
        df.groupBy([F.col(c) for c in key_cols])
        .count()
        .filter(F.col("count") > 1)
    )
    dup_count = float(dup_groups.count())
    return CheckResult(
        name=f"{name}__duplicates_warn",
        severity=Severity.WARN,
        passed=True,
        message=f"Duplicate key groups count={int(dup_count)} (allowed in Bronze).",
        metrics={"duplicate_key_groups": dup_count},
    )


def assert_timestamp_non_null(df: DataFrame, ts_col: str, name: str) -> CheckResult:
    """
    Assumes ingestion already created *_ts columns using to_timestamp().
    """
    if ts_col not in df.columns:
        return CheckResult(
            name=f"{name}__timestamp_exists",
            severity=Severity.ERROR,
            passed=False,
            message=f"Timestamp column '{ts_col}' is missing.",
        )

    nulls = float(df.filter(F.col(ts_col).isNull()).count())
    passed = (nulls == 0.0)
    return CheckResult(
        name=f"{name}__timestamp_non_null",
        severity=Severity.ERROR,
        passed=passed,
        message=f"Null timestamps in '{ts_col}': {int(nulls)}",
        metrics={f"{ts_col}_nulls": nulls},
    )


def soft_range_warn(df: DataFrame, col: str, min_val: float, max_val: float, name: str) -> CheckResult:
    """
    Bronze: soft WARN checks only. Missing column => INFO.
    """
    if col not in df.columns:
        return CheckResult(
            name=f"{name}__range_{col}",
            severity=Severity.INFO,
            passed=True,
            message=f"Column '{col}' not present; skipping range check.",
        )

    violations = float(
        df.filter((F.col(col) < F.lit(min_val)) | (F.col(col) > F.lit(max_val))).count()
    )

    if violations > 0:
        return CheckResult(
            name=f"{name}__range_{col}",
            severity=Severity.WARN,
            passed=True,
            message=f"Range violations for {col}: {int(violations)} rows outside [{min_val}, {max_val}]",
            metrics={f"{col}_range_violations": violations},
        )

    return CheckResult(
        name=f"{name}__range_{col}",
        severity=Severity.INFO,
        passed=True,
        message=f"No range violations for {col}.",
        metrics={f"{col}_range_violations": violations},
    )


# -----------------------------
# Relational / temporal checks
# -----------------------------

def fk_admission_exists(
    df_events: DataFrame,
    df_patients: DataFrame,
    name: str,
) -> CheckResult:
    """
    ERROR if any events have no matching (patient_id, admission_id) in patients.
    """
    joined = (
        df_events.alias("e")
        .join(
            df_patients.select("patient_id", "admission_id").alias("p"),
            on=["patient_id", "admission_id"],
            how="left",
        )
    )
    missing = float(joined.filter(F.col("p.patient_id").isNull()).count())
    passed = (missing == 0.0)
    return CheckResult(
        name=f"{name}__fk_admission_exists",
        severity=Severity.ERROR,
        passed=passed,
        message=f"Events with missing patient admission: {int(missing)}",
        metrics={"missing_patient_admission": missing},
    )


def event_within_admission_window(
    df_events: DataFrame,
    df_patients: DataFrame,
    event_ts_col: str,
    name: str,
) -> CheckResult:
    """
    ERROR if event timestamp is outside [admission_time_ts, discharge_time_ts] (if discharge exists).
    Requires patients to have admission_time_ts and discharge_time_ts.
    """
    needed_patient_cols = ["patient_id", "admission_id", "admission_time_ts", "discharge_time_ts"]
    missing_cols = [c for c in needed_patient_cols if c not in df_patients.columns]
    if missing_cols:
        return CheckResult(
            name=f"{name}__event_within_admission_window",
            severity=Severity.ERROR,
            passed=False,
            message=f"Patients missing required columns for window check: {missing_cols}",
        )

    if event_ts_col not in df_events.columns:
        return CheckResult(
            name=f"{name}__event_within_admission_window",
            severity=Severity.ERROR,
            passed=False,
            message=f"Events missing timestamp column: {event_ts_col}",
        )

    joined = (
        df_events.alias("e")
        .join(
            df_patients.select(*needed_patient_cols).alias("p"),
            on=["patient_id", "admission_id"],
            how="left",
        )
    )

    # If admission_time_ts is null, FK check should catch it, but keep logic safe.
    out_of_window = float(
        joined.filter(
            (F.col("p.admission_time_ts").isNotNull()) &
            (
                (F.col(f"e.{event_ts_col}") < F.col("p.admission_time_ts")) |
                (
                    F.col("p.discharge_time_ts").isNotNull() &
                    (F.col(f"e.{event_ts_col}") > F.col("p.discharge_time_ts"))
                )
            )
        ).count()
    )

    passed = (out_of_window == 0.0)
    return CheckResult(
        name=f"{name}__event_within_admission_window",
        severity=Severity.ERROR,
        passed=passed,
        message=f"Events outside admission window: {int(out_of_window)}",
        metrics={"outside_admission_window": out_of_window},
    )


# -----------------------------
# Bronze check bundles
# -----------------------------

def bronze_checks_patients(df_patients: DataFrame) -> List[CheckResult]:
    results: List[CheckResult] = []
    results.append(row_count(df_patients, "patients"))
    results.append(required_columns(df_patients, ["patient_id", "admission_id", "admission_time_ts"], "patients"))
    results.append(required_non_null(df_patients, ["patient_id", "admission_id", "admission_time_ts"], "patients"))
    results.append(assert_timestamp_non_null(df_patients, "admission_time_ts", "patients"))
    # discharge_time_ts may be null; no ERROR check for nulls
    return results


def bronze_checks_vitals(df_vitals: DataFrame) -> List[CheckResult]:
    results: List[CheckResult] = []
    results.append(row_count(df_vitals, "vitals_events"))
    results.append(required_columns(df_vitals, ["patient_id", "admission_id", "event_time_ts"], "vitals_events"))
    results.append(required_non_null(df_vitals, ["patient_id", "admission_id", "event_time_ts"], "vitals_events"))
    results.append(assert_timestamp_non_null(df_vitals, "event_time_ts", "vitals_events"))
    results.append(warn_duplicates(df_vitals, ["patient_id", "admission_id", "event_time_ts"], "vitals_events"))

    # Soft ranges (WARN)
    results.append(soft_range_warn(df_vitals, "spo2", 0, 100, "vitals_events"))
    results.append(soft_range_warn(df_vitals, "temp_c", 30, 45, "vitals_events"))
    results.append(soft_range_warn(df_vitals, "hr", 20, 220, "vitals_events"))
    results.append(soft_range_warn(df_vitals, "sbp", 50, 250, "vitals_events"))
    results.append(soft_range_warn(df_vitals, "rr", 5, 60, "vitals_events"))
    return results


def bronze_checks_labs(df_labs: DataFrame) -> List[CheckResult]:
    results: List[CheckResult] = []
    results.append(row_count(df_labs, "labs_events"))
    results.append(required_columns(df_labs, ["patient_id", "admission_id", "event_time_ts"], "labs_events"))
    results.append(required_non_null(df_labs, ["patient_id", "admission_id", "event_time_ts"], "labs_events"))
    results.append(assert_timestamp_non_null(df_labs, "event_time_ts", "labs_events"))
    results.append(warn_duplicates(df_labs, ["patient_id", "admission_id", "event_time_ts"], "labs_events"))

    # Soft ranges (WARN)
    results.append(soft_range_warn(df_labs, "wbc", 0, 50, "labs_events"))
    results.append(soft_range_warn(df_labs, "creatinine", 0.2, 10, "labs_events"))
    results.append(soft_range_warn(df_labs, "lactate", 0, 15, "labs_events"))
    return results


def bronze_checks_outcomes(df_outcomes: DataFrame) -> List[CheckResult]:
    results: List[CheckResult] = []
    results.append(row_count(df_outcomes, "outcomes"))
    results.append(required_columns(df_outcomes, ["patient_id", "admission_id", "outcome_time_ts", "outcome_type"], "outcomes"))
    results.append(required_non_null(df_outcomes, ["patient_id", "admission_id", "outcome_time_ts", "outcome_type"], "outcomes"))
    results.append(assert_timestamp_non_null(df_outcomes, "outcome_time_ts", "outcomes"))
    results.append(warn_duplicates(df_outcomes, ["patient_id", "admission_id", "outcome_time_ts", "outcome_type"], "outcomes"))
    return results
