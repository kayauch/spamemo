"""Summarize shared-policy runs without confusing frame and room holdouts."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path, nargs='+')
    args = parser.parse_args()
    rows = []
    seen = set()
    for run in args.runs:
        status = json.loads((run/'status.json').read_text(encoding='utf-8'))
        alignment_path = run/'colmap/alignment.json'
        alignment = json.loads(alignment_path.read_text(encoding='utf-8')) if alignment_path.exists() else {}
        identity = alignment.get('source_npz_sha256')
        rows.append(dict(scene=status['scene'], policy=status['policy'], state=status['state'],
                         repeated_reconstruction=identity in seen, decision=status.get('decision')))
        if identity and status['state'] == 'complete' and status.get('decision'):
            seen.add(identity)
    print(json.dumps(dict(runs=rows, distinct_completed_reconstructions=len(seen),
        scope='Distinct reconstructions are not necessarily distinct rooms. Verify capture provenance. '
              'Frame holdouts are used for candidate selection; reserve separate rooms for final quality validation.'),
        ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
