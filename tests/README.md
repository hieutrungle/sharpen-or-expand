# Tests layout

Same convention as verl: each folder mirrors a sub-namespace of `soe/` (`tests/utils/metric` tests
`soe/utils/metric`), and files ending in `_on_cpu.py` need no GPU and run in CI
(`.github/workflows/cpu_unit_tests.yml`). Special folders:

- `special_sanity/`: quick repo-wide checks (every example config validates).
- `special_e2e/`: GPU end-to-end tests run on FASTER inside the container (added in M1).
