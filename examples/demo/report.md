# Synthetic room: what changed?

Changes found: 3 (moved 1, added 1, removed 1).

- The chair-sized object moved 2.31 m (+2.15 m in x and +0.83 m in y). It was at (1.8, 2.8) and is now at (4.0, 3.6). Free-space path confirmed.
- A small object (bag or box) that was in the first recording is gone from (4.7, 0.7).
- A new small object (bag or box) appeared at (2.5, 4.1), about 0.60 m tall.

## Coverage
- Rule: a change requires occupied in one pass and *seen empty* in the other. Unknown never counts.
- Voxel size 0.06 m; grid [106, 89, 49].
- 75.2% of the box was seen by both recordings. Only that part is compared.
- 7583 voxels are occupied in one recording and unseen in the other. They are reported as unverified, not as changes.

## Evaluation against ground truth
- Precision 1.00, recall 1.00 (TP 3, FP 0, FN 0, tolerance 0.5 m).
