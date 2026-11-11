# Experiment reproduction contract

The selected experiment uses accurate ResubALS initialization, not imported approximate best checkpoints. Area and timing belong to the same search-selected graph. Final validation must not reselect the winner, trigger retries, or feed back into the same optimization run.

The frozen algorithm is preserved in this release to avoid silently combining later HBF changes with the paper results. There are two explicit source origins in `release_inventory.json`: frozen numerical/search implementations and later experiment/model orchestration.

Full reproduction requires:

- The exact cell library listed by SHA256 in `assets/required_assets.json`.
- Accurate ISCAS85 baseline netlists, source BLIFs, port mappings, and equivalence proof records.
- Accurate BACS baseline netlists, DC/reference netlists, original output order, and source BLIFs.
- The frozen generic and learned knowledge used by the repeated experiment.
- ABC with the recorded behavior, and a configured execution environment.
- A provider account and an environment-supplied API key for real agent runs. Model names and historical price constants in source describe the recorded experiment, not a verified current service/pricing claim.

The published repeated-experiment entry imports a content-checked, relocatable asset bundle. Windows/WSL was validated. Some auxiliary research scripts retain author-tool generation dependencies and are not the supported public quick start. Use the commands in the repository README. Published portability adapters preserve the declared numerical and statistical protocol.
