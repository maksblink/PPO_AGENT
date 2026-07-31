# Runtime artifacts

This directory stores generated, immutable project artifacts such as:

- PPO checkpoints,
- evaluation equity curves,
- evaluation trade records,
- evaluation summaries.

Artifact files are generated at runtime and are not committed to Git.
Their relative paths, SHA-256 hashes, and sizes are recorded in PostgreSQL.
