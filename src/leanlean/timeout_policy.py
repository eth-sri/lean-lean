"""Shared timeout policy for Lean verification commands."""

# Large stripped developments can spend several hours in the final comparator
# after the Lake build has completed. Keep this budget shared by generation-time
# verification and postprocessing so they cannot silently diverge.
LEAN_VERIFY_TIMEOUT_SECONDS = 12 * 60 * 60
