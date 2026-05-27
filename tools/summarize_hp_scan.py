#!/usr/bin/env python3
"""Summarize hp prototype scan CSV files into compact Markdown tables."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def short_mesh(path: str) -> str:
    name = Path(path).name
    return name.replace("-singlepatch-nurbs", "").replace(".mesh", "")


def int_or_blank(value: str) -> str:
    return value if value else "-"


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_core_table(rows: list[dict[str, str]], out: Path) -> None:
    grouped: dict[tuple[str, str, str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        key = (
            short_mesh(row["mesh"]),
            row["epsilon_mode"],
            row["order"],
            row["ref_levels"],
        )
        grouped[key][row["proto_mode"]] = row

    lines = [
        "| mesh | eps | p | ref | true DOFs | zero | FDFD | AMS | edge Yee | edge gain | status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for key in sorted(grouped):
        mesh, eps, order, ref = key
        protos = grouped[key]
        first = next(iter(protos.values()))
        edge = protos.get("edge_yee_proto", {})
        ams = protos.get("ams", {})
        first = edge or ams or first
        zero = edge.get("zero_iters") or ams.get("zero_iters", "")
        fdfd = edge.get("fdfd_iters") or ams.get("fdfd_iters", "")
        edge_iters = edge.get("auxprec_iters", "")
        gain = "-"
        if zero and edge_iters and edge_iters != "0":
            gain = f"{float(zero) / float(edge_iters):.2f}"
        status = ",".join(sorted({r.get("status", "") for r in protos.values() if r.get("status")}))
        lines.append(
            "| "
            + " | ".join(
                [
                    mesh,
                    eps,
                    order,
                    ref,
                    int_or_blank(first.get("true_vsize", "")),
                    int_or_blank(zero),
                    int_or_blank(fdfd),
                    int_or_blank(ams.get("auxprec_iters", "")),
                    int_or_blank(edge_iters),
                    gain,
                    status or "-",
                ]
            )
            + " |"
        )
    out.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("covariant_aux_space/hp_proto_scan_summary.md"),
    )
    args = parser.parse_args()

    rows = load_rows(args.csv)
    write_core_table(rows, args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
