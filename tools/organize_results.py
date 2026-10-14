#!/usr/bin/env python3
"""Move flat historical results into method/execution directories; keep their provenance."""
import argparse
from datetime import datetime
from pathlib import Path
import re
from zoneinfo import ZoneInfo


def organize(root):
    moved = removed = 0
    pattern = re.compile(r'(.+)_(FedAvg|FedLoad|HERAFL)_(.+)_(\d+)\.h5')
    for path in sorted(root.glob('*.h5')):
        match = pattern.fullmatch(path.name)
        if not match:
            continue
        dataset, algorithm, goal, run = match.groups()
        stamp = datetime.fromtimestamp(path.stat().st_mtime, ZoneInfo('America/Belem')).strftime('%Y%m%d_%Hh%Mm%Ss')
        method = algorithm.lower()
        directory = root / method / f'{stamp}_{method}_experiment_{dataset.lower()}_legacy_{goal}_run{run}'
        directory.mkdir(parents=True, exist_ok=True)
        policy = path.with_name(path.stem + '_policy.json')
        for source in (path, policy):
            if source.exists():
                target = directory / source.name
                if target.exists():
                    raise FileExistsError(target)
                source.rename(target)
                moved += 1
    for source in root.glob('comparison/*/*.png'):
        target = source.parent / 'plots' / source.name
        target.parent.mkdir(exist_ok=True)
        if target.exists():
            raise FileExistsError(target)
        source.rename(target)
        moved += 1
    for pdf in root.rglob('*.pdf'):
        # Remove only redundant exports; preserve any PDF with no PNG equivalent.
        if pdf.with_suffix('.png').exists() or (pdf.parent / 'plots' / pdf.with_suffix('.png').name).exists():
            pdf.unlink()
            removed += 1
    return moved, removed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'results')
    args = parser.parse_args()
    moved, removed = organize(args.results_dir)
    print(f'Moved {moved} files; removed {removed} redundant PDF exports.')
