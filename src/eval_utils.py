"""
Evaluation utilities: execution accuracy, exact match, valid-SQL rate,
and bootstrap confidence intervals.

Requires Spider's SQLite database files on disk, laid out as:
    {databases_dir}/{db_id}/{db_id}.sqlite
This is Spider's standard release layout (database.zip from the official
Spider download). The HF `datasets` text data does NOT include these files —
download them separately and point --databases_dir at the unzipped folder.
"""

import re
import sqlite3
from pathlib import Path

import numpy as np
import sqlglot
from sqlglot import exp


def is_valid_sql(sql: str, dialect: str = "sqlite") -> bool:
    """Parses (but does not execute) the SQL. Catches syntax errors cheaply."""
    try:
        sqlglot.parse_one(sql, read=dialect)
        return True
    except Exception:
        return False


def is_read_only(sql: str, dialect: str = "sqlite") -> bool:
    """
    True only if the statement is a SELECT (optionally with CTEs).
    Used both for the eval harness and reused later in Phase 3's safety layer.
    """
    try:
        tree = sqlglot.parse_one(sql, read=dialect)
    except Exception:
        return False
    if isinstance(tree, exp.With):
        tree = tree.this
    return isinstance(tree, exp.Select)


def execute_sql(db_path: Path, sql: str, timeout_s: float = 5.0):
    """
    Executes SQL against a SQLite DB file, read-only.
    Returns (rows, error). rows is None on failure.
    """
    try:
        # uri=True + mode=ro enforces read-only at the connection level —
        # a real defense (Phase 3), not just "we didn't call INSERT" hygiene.
        uri = f"file:{db_path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=timeout_s)
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        conn.close()
        return rows, None
    except Exception as e:
        return None, str(e)


def _normalize_rows(rows):
    """Order-insensitive, type-loose comparison of result sets."""
    if rows is None:
        return None
    normalized = []
    for row in rows:
        normalized.append(tuple(str(v) for v in row))
    return sorted(normalized)


def execution_match(db_path: Path, pred_sql: str, gold_sql: str) -> bool:
    """
    True if predicted SQL executes AND returns the same result set as gold.
    Note (per review feedback): this can false-positive when two DIFFERENT
    but coincidentally-equivalent-on-this-data queries return the same rows.
    That's why Day 7 / Phase 4 also does manual semantic review on a sample —
    don't treat this number alone as ground truth.
    """
    pred_rows, pred_err = execute_sql(db_path, pred_sql)
    if pred_err is not None:
        return False
    gold_rows, gold_err = execute_sql(db_path, gold_sql)
    if gold_err is not None:
        # gold itself failed to execute — data issue, shouldn't normally happen
        return False
    return _normalize_rows(pred_rows) == _normalize_rows(gold_rows)


def normalize_sql_text(sql: str) -> str:
    """Loose text normalization for exact-match scoring (not a substitute for parsing)."""
    sql = sql.strip().rstrip(";")
    sql = re.sub(r"\s+", " ", sql)
    return sql.lower()


def exact_match(pred_sql: str, gold_sql: str) -> bool:
    return normalize_sql_text(pred_sql) == normalize_sql_text(gold_sql)


def bootstrap_ci(binary_outcomes: list, n_boot: int = 2000, ci: float = 0.95, seed: int = 42):
    """
    Bootstrap confidence interval for a proportion (e.g. execution accuracy)
    given a list of 0/1 outcomes.
    """
    rng = np.random.default_rng(seed)
    arr = np.array(binary_outcomes, dtype=float)
    if len(arr) == 0:
        return (float("nan"), float("nan"), float("nan"))
    means = []
    n = len(arr)
    for _ in range(n_boot):
        sample = arr[rng.integers(0, n, n)]
        means.append(sample.mean())
    means = np.array(means)
    lower_p = (1 - ci) / 2 * 100
    upper_p = (1 + ci) / 2 * 100
    return (arr.mean(), np.percentile(means, lower_p), np.percentile(means, upper_p))


def evaluate_predictions(records: list, predictions: list, databases_dir: Path):
    """
    records: list of dicts with at least {db_id, gold_sql, difficulty}
    predictions: list of predicted SQL strings, same order/length as records

    Returns a results dict with overall + per-difficulty + per-db metrics,
    plus the raw per-example outcome lists needed for regression analysis
    (Day 5) and bootstrap CIs.
    """
    assert len(records) == len(predictions)

    per_example = []
    for rec, pred in zip(records, predictions):
        db_path = databases_dir / rec["db_id"] / f"{rec['db_id']}.sqlite"
        valid = is_valid_sql(pred)
        ex_match = execution_match(db_path, pred, rec["gold_sql"]) if valid else False
        exact = exact_match(pred, rec["gold_sql"])
        per_example.append({
            "db_id": rec["db_id"],
            "difficulty": rec["difficulty"],
            "question": rec["question"],
            "gold_sql": rec["gold_sql"],
            "pred_sql": pred,
            "valid_sql": valid,
            "execution_match": ex_match,
            "exact_match": exact,
        })

    exec_outcomes = [int(e["execution_match"]) for e in per_example]
    exact_outcomes = [int(e["exact_match"]) for e in per_example]
    valid_outcomes = [int(e["valid_sql"]) for e in per_example]

    exec_mean, exec_lo, exec_hi = bootstrap_ci(exec_outcomes)

    by_difficulty = {}
    for diff in sorted(set(e["difficulty"] for e in per_example)):
        subset = [e for e in per_example if e["difficulty"] == diff]
        outcomes = [int(e["execution_match"]) for e in subset]
        by_difficulty[diff] = {
            "n": len(subset),
            "execution_accuracy": float(np.mean(outcomes)) if subset else float("nan"),
        }

    by_db = {}
    for db_id in sorted(set(e["db_id"] for e in per_example)):
        subset = [e for e in per_example if e["db_id"] == db_id]
        outcomes = [int(e["execution_match"]) for e in subset]
        by_db[db_id] = {
            "n": len(subset),
            "execution_accuracy": float(np.mean(outcomes)) if subset else float("nan"),
        }

    return {
        "n": len(per_example),
        "execution_accuracy": float(np.mean(exec_outcomes)),
        "execution_accuracy_ci95": (float(exec_lo), float(exec_hi)),
        "exact_match": float(np.mean(exact_outcomes)),
        "valid_sql_rate": float(np.mean(valid_outcomes)),
        "by_difficulty": by_difficulty,
        "by_db": by_db,
        "per_example": per_example,  # keep raw outcomes for regression analysis later
    }

