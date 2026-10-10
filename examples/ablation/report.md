# Occlusion ablation: why unknown ≠ removed

Change only where both recordings proved free space; unknown is never removed.

## Naive occupancy differencing
- Rule: `occupied_before AND not_occupied_after (UNKNOWN counts as gone)`.
- Removal clusters: **11** (1802 voxels).
- False-removal clusters (beyond ground-truth removals): **10**.

## Ours (visibility-first)
- Rule: `occupied_before AND seen_empty_after (UNKNOWN unverified)`.
- Removal clusters: **1** (315 voxels).
- False-removal clusters: **0**.
- Reported changes: 3 (moved 1, added 1, removed 1).

Unverified voxels (occupied on one side, unseen on the other): 7583.

See `naive_vs_ours.png` (left = naive false removals in red; right = ours; amber = unverified).
