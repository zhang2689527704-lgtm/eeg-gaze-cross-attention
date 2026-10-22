"""Run the two synthetic offsets missing from the original five-point table
(1.10 and 1.20) and merge them into mask_verification.json.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_mask import synthetic_recovery

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, '..', 'results')
path = os.path.join(RESULTS, 'mask_verification.json')

report = json.load(open(path))
new = synthetic_recovery(seeds=(0, 1, 2), target_offsets=(1.10, 1.20))
keep = [x for x in report['synthetic']
        if x['target_offset'] not in (1.10, 1.20)]
report['synthetic'] = sorted(keep + new,
                             key=lambda x: (x['target_offset'], x['seed']))
with open(path, 'w') as f:
    json.dump(report, f, indent=1)
print('synthetic entries:', len(report['synthetic']))
for x in report['synthetic']:
    print(f"  offset={x['target_offset']:.2f} seed={x['seed']} "
          f"delta={x['delta_learned']:.4f} beta={x['bandwidth']:.4f}")
