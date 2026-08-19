# Benchmark summary

| Dataset | Implementation | Status | Epochs | Total wall (s) | Training wall (s) | Peak CPU RSS (GiB) | Peak GPU process (GiB) | Peak torch allocated (GiB) | Mean GPU util (%) |
|---|---|---|---|---|---|---|---|---|---|
| Bone Marrow | dense | ok | 1 | 2.57 | 2.46 | 2.26 | 1.49 | 0.60 | 23.00 |
| Bone Marrow | sparse | ok | 1 | 2.63 | 2.52 | 2.25 | 1.49 | 0.60 | 15.33 |
| Bone Marrow | dense | ok | 1 | 0.78 | — | 0.93 | 0.90 | 0.28 | — |
| Cancer | dense | ok | 1 | 5.04 | 4.30 | 5.42 | 1.36 | 0.51 | 13.50 |
| Cancer | sparse | ok | 1 | 5.44 | 4.70 | 5.42 | 1.36 | 0.51 | 19.60 |
| Cancer | dense | ok | 1 | 31.24 | — | 34.64 | 6.76 | 6.14 | — |
