"""
SQL difficulty classification, approximating Spider's official hardness rules
(easy / medium / hard / extra hard) from taoyds/spider's evaluation.py.

Spider's canonical hardness is computed from the *parsed* SQL (via their
process_sql.py). This module gives a lightweight, dependency-free approximation
based on surface SQL text, which is enough for stratified sampling. For a
Day-3 evaluation table you should ideally use the actual difficulty labels
Spider's eval script assigns to the parsed SQL, if you have that script
available — the approximation here is only meant to unblock Day 1 sampling.
"""

import re


def _count(pattern: str, sql: str) -> int:
    return len(re.findall(pattern, sql, flags=re.IGNORECASE))


def classify_hardness(sql: str) -> str:
    """
    Approximate Spider hardness classification.
    Components counted (mirrors the spirit of the official rule set):
      - number of SELECT columns
      - number of WHERE conditions (AND/OR)
      - GROUP BY / HAVING presence
      - ORDER BY / LIMIT presence
      - nested subqueries
      - set operations (UNION/INTERSECT/EXCEPT)
      - aggregation functions
      - number of joined tables (rough, via JOIN count / FROM comma count)
    """
    sql_upper = sql.upper()

    select_cols = sql.split("FROM")[0].count(",") + 1 if "FROM" in sql_upper else 1
    where_conds = _count(r"\bAND\b|\bOR\b", sql) + (1 if " WHERE " in f" {sql_upper} " else 0)
    has_group = " GROUP BY " in f" {sql_upper} "
    has_having = " HAVING " in f" {sql_upper} "
    has_order = " ORDER BY " in f" {sql_upper} "
    has_limit = " LIMIT " in f" {sql_upper} "
    nested = _count(r"\(\s*SELECT", sql)
    set_ops = _count(r"\bUNION\b|\bINTERSECT\b|\bEXCEPT\b", sql)
    agg_funcs = _count(r"\bCOUNT\s*\(|\bSUM\s*\(|\bAVG\s*\(|\bMIN\s*\(|\bMAX\s*\(", sql)
    joins = _count(r"\bJOIN\b", sql) + max(sql.split("FROM")[-1].split("WHERE")[0].count(",") if "FROM" in sql_upper else 0, 0)

    component_count = 0
    component_count += 1 if select_cols > 1 else 0
    component_count += 1 if where_conds > 0 else 0
    component_count += 1 if has_group or has_having else 0
    component_count += 1 if has_order or has_limit else 0
    component_count += 2 if nested > 0 else 0
    component_count += 2 if set_ops > 0 else 0
    component_count += 1 if agg_funcs > 0 else 0
    component_count += 1 if joins >= 2 else 0

    if nested > 1 or set_ops > 0 and (agg_funcs > 0 or joins >= 2):
        return "extra_hard"
    if nested > 0 or (agg_funcs > 0 and joins >= 2 and (has_group or where_conds > 1)):
        return "hard"
    if component_count >= 2:
        return "medium"
    return "easy"

