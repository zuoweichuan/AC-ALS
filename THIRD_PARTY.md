# Third-party materials

This release does not redistribute circuit benchmarks, standard-cell libraries, tool binaries, or original results.

The existing `stdcells.lib` begins with a Nangate copyright and license-agreement notice. No separate permission record for this copy has been verified during release preparation. It is omitted rather than relicensed under the AC-ALS MIT license.

The local ResubALS checkout contains a BSD 3-Clause license. No ResubALS source, original benchmark archive, or compiled object is included here. A license on one upstream project does not automatically establish the redistribution terms of every bundled benchmark and cell library.

ABC is an external dependency of the declared ER/NMED research protocol, including exact mapping/equivalence and same-graph timing measurements. Its source and binary are not included. This benchmark protocol is distinct from the separate ABC-free HBF workflow.

The publication package must record upstream locations, revisions, licenses, baseline generation, and checksums before redistributing any additional third-party material. Expected asset fingerprints are retained in `assets/required_assets.json` without copying private machine paths or binaries.
