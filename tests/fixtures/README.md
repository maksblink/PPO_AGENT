# Test fixtures

These inputs belong to the test suite and are independent of experiment files
under the repository's top-level `configs/` directory.

- `example_5m_*.yml`: ordinary run configuration contracts.
- `temporal_run.yml`: fixed temporal ranges for range, source and window tests.
- `walk_forward_protocol.yml`: a complete explicit grid with a synthetic source ID.
- `queue_project/`: an isolated project root with a sequential test queue and its
  run inputs; queue tests override the production discovery root.

Paths to market data and database IDs in YAML are test inputs, not dependencies
on existing datasets or experiments. Tests mock loaders or create synthetic data
and temporary databases as appropriate. Changes to research configurations must
not change the behavior or expected values of these fixtures.
