# AC-ALS: Agentic Search for Compensation-Aware Approximate Logic Synthesis

AC-ALS provides agent-controlled compensation-aware approximate synthesis of mapped combinational circuits. This release covers **ISCAS85 strict vector ER** and **BACS NMED**, including the DeepSeek/GLM repeated-experiment protocol.

## Experiment scope

| Benchmark | Circuits | Error constraint |
| --- | --- | --- |
| ISCAS85 | c880, c1908, c2670, c3540, c5315, c7552 | vector ER <= 0.005 or 0.05 |
| BACS | abs_diff, adder32, buttfly, mac, mult8, mult16 | NMED <= 0.0059 or 0.0294 |

Vector ER counts an input as wrong if any output bit differs. NMED measures normalized absolute numerical error. These are not NHD constraints.

The repeated protocol uses 24 circuit/constraint combinations and five search seeds, giving 120 jobs per model. Input groups contain 10,000 vectors; summary statistics report mean and sample standard deviation (`ddof=1`). Final validation does not reselect the graph. Technical failures and over-limit outcomes are retained. See [experiment details](docs/REPRODUCTION.md).

## Requirements

- Python 3.10+ and NumPy.
- ABC for exact mapping/equivalence and same-graph timing; no ABC binary is bundled.
- Windows with WSL/Ubuntu is the validated external-tool environment. Native POSIX execution is supported by the tool adapter but has not undergone the complete benchmark validation here.
- A separately acquired, authorized benchmark/library bundle. Third-party assets are not included in this repository.
- A model-provider API key for actual Agent runs. Offline checks and preparation do not call the API.

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The offline tests use an original toy library and circuit. They do not claim to reproduce benchmark optimization results.

## Install benchmark assets

Obtain an authorized bundle and import it locally:

```sh
python scripts/import_assets.py /path/to/authorized_bundle
```

Files are checked by SHA256 before installation. The default destination `ac_als/benchmark_assets/` is ignored by Git. Set `AC_ALS_ASSETS` when using a custom destination. Asset fingerprints are documented in `assets/required_assets.json`; redistribution notes are in [THIRD_PARTY.md](THIRD_PARTY.md).

Maintainers with the original research checkout can create a local bundle using `scripts/export_research_assets.py --research-root /path/to/netlist_frontier --output /path/to/local_bundle`. This export does not confer redistribution permission and must not be committed without reviewing the included third-party terms.

## Prepare and run

Prepare the paired DeepSeek experiment first. This freezes accurate inputs, code and knowledge without paid model calls:

```sh
python -u -B ac_als/run_agent_repeats.py --prepare-only
```

A real run asks for a hidden key and can incur provider charges:

```sh
python -u -B ac_als/run_agent_repeats.py --prompt-key
```

Continue the same frozen run with `--resume --prompt-key`. For the paired GLM comparison, prepare/run it after the DeepSeek protocol exists:

```sh
python -u -B ac_als/run_agent_repeats.py --model zai-org/GLM-5.3-Flash --prepare-only
python -u -B ac_als/run_agent_repeats.py --model zai-org/GLM-5.3-Flash --prompt-key
```

Provider keys may alternatively be supplied through `SILICONFLOW_API_KEY`. They are never part of the repository. `--max-jobs 1` runs at most one pending job; it is not a complete repeated experiment. Pricing constants describe the historical experiment, not verified current prices. Cumulative per-circuit ledgers and run outcomes are preserved.

ABC defaults to `/usr/local/bin/abc` and WSL distribution `Ubuntu`. Configure `AC_ALS_ABC` and `AC_ALS_WSL_DISTRO` for your installation.

## Provenance and validation

The numerical/search implementation was selected from the frozen five-seed research sources rather than later HBF edits. Model/repeat orchestration uses the later wrappers that support both providers. Portability changes replace local drive restrictions, external-tool invocation, asset paths and archived-proof verification; they do not intentionally change candidate generation, error definitions, numerical selection, seeds or statistics.

Original and release source hashes are recorded in `docs/release_inventory.json`. Portability changes are tracked separately. HBF applications and unrelated experiment logs are outside this release.

Validation includes toy dual-simulator tests, source integrity checks and full preparation/preflight of the declared benchmark assets. No new paid Agent run was performed for this release. Archived baseline CEC logs are verified and retained as provenance; preparation does not pretend to regenerate those historical proofs.

## License

Project code is available under the [MIT License](LICENSE). Third-party libraries, circuits and tools retain their own terms and are not relicensed by this repository. Personal author names and contact details are intentionally omitted for this release.

## 中文说明

本仓库公开ISCAS85向量ER与BACS NMED的智能体补偿近似综合代码，包含DeepSeek/GLM重复实验编排。算法取自论文实验的冻结版本；移植修改集中于路径、外部工具与数据导入。第三方单元库、基准电路、原始运行日志和密钥不随代码发布。真实模型运行需要自行配置API凭据并承担调用费用，数据准备和离线测试不调用模型。
