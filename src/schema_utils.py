"""
Schema serialization utilities.

Two serialization strategies, used for the Day 5 ablation:
  - minimal:  table names + column names only
  - rich:     table names + columns WITH types, primary keys, foreign keys

Spider's tables.json (per DB) gives us table_names_original, column_names_original,
column_types, primary_keys, foreign_keys. This module turns that structured
schema into a prompt-ready string.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SchemaSpec:
    """A single DB's schema, in the shape Spider's tables.json provides."""
    db_id: str
    table_names: list          # ["singer", "concert", ...]
    column_names: list         # [(table_idx, col_name), ...] table_idx=-1 for '*'
    column_types: list         # ["text", "number", ...] aligned with column_names
    primary_keys: list         # [col_idx, ...] global column indices
    foreign_keys: list         # [(col_idx_a, col_idx_b), ...] global column indices


def _columns_for_table(schema: SchemaSpec, table_idx: int):
    """Return list of (global_col_idx, col_name, col_type) for a given table."""
    cols = []
    for global_idx, (t_idx, col_name) in enumerate(schema.column_names):
        if t_idx == table_idx:
            cols.append((global_idx, col_name, schema.column_types[global_idx]))
    return cols


def serialize_minimal(schema: SchemaSpec) -> str:
    """
    Minimal serialization: table names + column names only.
    No types, no keys. This is the 'lean' condition in the Day 5 ablation.
    """
    lines = []
    for t_idx, table_name in enumerate(schema.table_names):
        cols = _columns_for_table(schema, t_idx)
        col_str = ", ".join(c[1] for c in cols)
        lines.append(f"Table {table_name}: {col_str}")
    return "\n".join(lines)


def serialize_rich(schema: SchemaSpec) -> str:
    """
    Rich serialization: table names + columns with types, primary keys marked,
    and an explicit foreign-key relationship list.
    This is the 'rich' condition in the Day 5 ablation.
    """
    pk_set = set(schema.primary_keys)
    lines = []

    for t_idx, table_name in enumerate(schema.table_names):
        cols = _columns_for_table(schema, t_idx)
        col_parts = []
        for global_idx, col_name, col_type in cols:
            marker = " [PK]" if global_idx in pk_set else ""
            col_parts.append(f"{col_name} ({col_type}){marker}")
        lines.append(f"Table {table_name} (" + ", ".join(col_parts) + ")")

    if schema.foreign_keys:
        lines.append("Foreign keys:")
        for col_a, col_b in schema.foreign_keys:
            table_a, name_a = schema.column_names[col_a]
            table_b, name_b = schema.column_names[col_b]
            table_a_name = schema.table_names[table_a]
            table_b_name = schema.table_names[table_b]
            lines.append(f"  {table_a_name}.{name_a} -> {table_b_name}.{name_b}")

    return "\n".join(lines)


def build_prompt(question: str, schema_str: str) -> str:
    """Standard prompt template used across baseline, fine-tune, and eval."""
    return (
        "You are a text-to-SQL assistant. Given a database schema and a "
        "natural language question, write a single valid SQL query that "
        "answers the question. Only output the SQL query, nothing else.\n\n"
        f"### Schema:\n{schema_str}\n\n"
        f"### Question:\n{question}\n\n"
        "### SQL:\n"
    )

