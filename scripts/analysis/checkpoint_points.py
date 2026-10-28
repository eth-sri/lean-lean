"""Match a replay event without confusing a final submission with its last edit."""
from pathlib import Path


def match_checkpoint(points, frame):
    from scripts.analysis.extract_checkpoint_graphs import source_key
    candidates = [p for p in (points.values() if isinstance(points, dict) else points)
                  if p['edit_index'] == frame['edit']]
    if 'seconds' in frame:
        candidates = [p for p in candidates if p.get('elapsed_seconds') == frame['seconds']]
    if frame.get('source_key'):
        candidates = [p for p in candidates if source_key(p) == frame['source_key']]
    if frame.get('snapshot'):
        candidates = [p for p in candidates if Path(p['snapshot_path']).name == frame['snapshot']]
    if not candidates:
        raise ValueError(f'No captured checkpoint matches edit {frame["edit"]} at {frame.get("seconds")}')
    identities = {(source_key(p), p.get('snapshot_artifact_sha256')) for p in candidates}
    if len(identities) != 1:
        raise ValueError(f'Ambiguous captured checkpoint for edit {frame["edit"]}; source identity is required')
    return candidates[0]
