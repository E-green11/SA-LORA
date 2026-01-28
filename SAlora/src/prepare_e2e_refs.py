#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import csv
from collections import OrderedDict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()
    
    # Read test file and preserve order
    mr_to_refs = OrderedDict()
    
    with open(args.test_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            mr = row['mr']
            ref = row['ref']
            if mr not in mr_to_refs:
                mr_to_refs[mr] = []
            mr_to_refs[mr].append(ref)
    
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for mr, refs in mr_to_refs.items():
            for ref in refs:
                f.write(ref + "\n")
    
    print(f"Saved {sum(len(refs) for refs in mr_to_refs.values())} references for {len(mr_to_refs)} MRs")
    print(f"Output: {args.output_file}")


if __name__ == "__main__":
    main()

