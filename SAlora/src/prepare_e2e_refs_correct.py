#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
from collections import OrderedDict
from typing import Dict, List


def prepare_refs(test_file: str, output_file: str) -> None:
    """Read CSV test file and write references in the expected official format."""
    mr_to_refs: "OrderedDict[str, List[str]]" = OrderedDict()
    with open(test_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mr = row["mr"]
            ref = row["ref"]
            if mr not in mr_to_refs:
                mr_to_refs[mr] = []
            mr_to_refs[mr].append(ref)

    # Write references: one per line, blank line between different MRs
    with open(output_file, "w", encoding="utf-8") as f:
        first = True
        for mr, refs in mr_to_refs.items():
            if not first:
                f.write("\n")
            first = False
            for ref in refs:
                f.write(ref + "\n")

    print(f"Reference file saved: {output_file}")
    print(f"Total MRs: {len(mr_to_refs)}")


def main():
    parser = argparse.ArgumentParser(description="Prepare references for official e2e-metrics.")
    parser.add_argument("--test_file", type=str, required=True, help="Input CSV file containing 'mr' and 'ref' columns.")
    parser.add_argument("--output_file", type=str, required=True, help="Output path for the prepared reference file.")
    args = parser.parse_args()
    prepare_refs(args.test_file, args.output_file)


if __name__ == "__main__":
    main()
