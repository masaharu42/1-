"""Compare fixed-window in-sample and held-out absolute composition errors."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from data import ELEMENTS


def read_evaluation(directory, expected_partition):
    directory = Path(directory)
    config = json.loads((directory / 'evaluation_config.json').read_text(encoding='utf-8'))
    if config['partition'] != expected_partition:
        raise ValueError(f'{directory} is not the {expected_partition} partition')
    with (directory / 'predictions.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    for element in ELEMENTS:
        selected = [r for r in rows if r['element'] == element]
        if not selected or len({r['alloy_id'] for r in selected}) != len(selected):
            raise ValueError(f'Missing or duplicated {element} results in {directory}')
    return config, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', required=True, help='Directory from train partition evaluation')
    parser.add_argument('--test', required=True, help='Directory from test partition evaluation')
    parser.add_argument('--val', help='Directory from validation partition evaluation')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    sources = [('train', args.train), ('test', args.test)]
    if args.val:
        sources.insert(1, ('val', args.val))
    evaluations = [(part, *read_evaluation(path, part)) for part, path in sources]
    first = evaluations[0][1]
    for _, config, _ in evaluations[1:]:
        for key in ('checkpoint_sha256', 'windows', 'split', 'manifest'):
            if config.get(key) != first.get(key):
                raise ValueError(f'Evaluation {key} mismatch')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError('Use a new empty output directory')
    summary = []
    for element in ELEMENTS:
        means = []
        for part, _, rows in evaluations:
            errors = [float(r['abs_error_wt_pct']) for r in rows if r['element'] == element]
            means.append(float(np.mean(errors)))
            summary.append({'element': element, 'partition': part, 'alloy_count': len(errors),
                            'mae_wt_pct': means[-1], 'max_abs_error_wt_pct': max(errors)})
        fig, ax = plt.subplots(figsize=(6, 4))
        labels = [f'{part} (n={sum(r["element"] == element for r in rows)})' for part, _, rows in evaluations]
        bars = ax.bar(labels, means, color=['#598ab4', '#e5a54e', '#ca6964'][:len(means)])
        ax.bar_label(bars, fmt='%.2f', padding=3)
        ax.set_ylim(0, max(max(means) * 1.2, 0.1))
        ax.set_ylabel('Mean absolute error (wt. percentage points)')
        ax.set_title(f'{element}: train / held-out error')
        fig.tight_layout()
        fig.savefig(output / f'{element}_训练与留出误差.png', dpi=160)
        plt.close(fig)
    with (output / '逐元素误差对照.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    (output / '比较配置.json').write_text(json.dumps({'sources': sources, 'checkpoint_sha256': first['checkpoint_sha256'], 'windows': first['windows']}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Saved comparison to {output}. Train is in-sample; individual held-out partitions may contain only one alloy.')


if __name__ == '__main__':
    main()
