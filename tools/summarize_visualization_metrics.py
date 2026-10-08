#!/usr/bin/env python3
"""Aggregate visualization summary.json files into markdown and CSV tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


TRACKING_COLUMNS = [
    "checkpoint",
    "dataset",
    "num_samples",
    "mean_2d_px_err",
    "p90_2d_px_err",
    "mean_3d_euc_err",
    "p90_3d_euc_err",
    "vis_acc",
    "gt_vis_ratio",
    "pred_vis_ratio",
]

DENSE_COLUMNS = [
    "checkpoint",
    "dataset",
    "num_samples",
    "dense_gt_mean_selected_points",
    "dense_pred_mean_visible_points",
    "dense_pred_world_mean_visible",
    "canonical_mean_visible_points",
]

CSV_COLUMNS = [
    "checkpoint",
    "dataset",
    "num_samples",
    "mean_2d_px_err",
    "median_2d_px_err",
    "p90_2d_px_err",
    "mean_3d_euc_err",
    "median_3d_euc_err",
    "p90_3d_euc_err",
    "vis_acc",
    "gt_vis_ratio",
    "pred_vis_ratio",
    "dense_gt_mean_selected_points",
    "dense_gt_max_selected_points",
    "dense_pred_mean_visible_points",
    "dense_pred_max_visible_points",
    "dense_pred_world_mean_visible",
    "dense_pred_world_max_visible",
    "canonical_mean_visible_points",
    "canonical_max_visible_points",
    "summary_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/vis_all", help="Search root for summary.json files.")
    parser.add_argument("--output-md", default=None, help="Where to write the markdown table.")
    parser.add_argument("--output-csv", default=None, help="Where to write the CSV table.")
    return parser.parse_args()


def is_nan(value: object) -> bool:
    return isinstance(value, float) and math.isnan(value)


def format_value(column: str, value: object) -> str:
    if value is None or is_nan(value):
        return "-"
    if column in {"checkpoint", "dataset", "summary_path"}:
        return str(value)
    if column == "num_samples":
        return str(int(value))
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if "points" in column or "visible" in column:
            return f"{value:.1f}"
        return f"{value:.4f}"
    return str(value)


def load_rows(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for summary_path in sorted(root.glob("**/summary.json")):
        rel = summary_path.relative_to(root)
        if len(rel.parts) < 3:
            continue

        checkpoint = "/".join(rel.parts[:-2])
        dataset = rel.parts[-2]
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        row: dict[str, object] = {
            "checkpoint": checkpoint,
            "dataset": dataset,
            "num_samples": int(summary.get("num_samples", len(summary.get("samples", [])))),
            "summary_path": str(summary_path),
        }
        row.update(summary.get("mean_metrics", {}))
        rows.append(row)

    rows.sort(key=lambda item: (str(item["dataset"]), str(item["checkpoint"])))
    return rows


def build_markdown_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    headers = [col for col in columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = [format_value(column, row.get(column)) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            csv_row = {}
            for column in CSV_COLUMNS:
                value = row.get(column)
                if value is None or is_nan(value):
                    csv_row[column] = ""
                else:
                    csv_row[column] = value
            writer.writerow(csv_row)


def build_markdown_document(root: Path, rows: list[dict[str, object]]) -> str:
    lines = [
        "# Visualization Metrics",
        "",
        f"- root: `{root}`",
        f"- summaries: {len(rows)}",
        "",
        "## Tracking Metrics",
        "",
        build_markdown_table(rows, TRACKING_COLUMNS),
        "",
        "## Dense Metrics",
        "",
        build_markdown_table(rows, DENSE_COLUMNS),
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    rows = load_rows(root)
    if not rows:
        raise SystemExit(f"No summary.json files found under {root}")

    markdown = build_markdown_document(root, rows)

    if args.output_md:
        output_md = Path(args.output_md).resolve()
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(markdown, encoding="utf-8")
        print(f"[metrics] markdown -> {output_md}")

    if args.output_csv:
        output_csv = Path(args.output_csv).resolve()
        write_csv(rows, output_csv)
        print(f"[metrics] csv -> {output_csv}")

    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
