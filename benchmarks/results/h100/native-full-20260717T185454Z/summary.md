# Benchmark summary

| Dataset | Implementation | Status | Epochs | Total wall (s) | Training wall (s) | Peak CPU RSS (GiB) | Peak GPU process (GiB) | Peak torch allocated (GiB) | Mean GPU util (%) |
|---|---|---|---|---|---|---|---|---|---|
| Bone Marrow | dense | ok | 200 | 1660.78 | 1660.03 | 3.05 | 1.63 | 0.79 | 36.26 |
| Bone Marrow | sparse | ok | 200 | 1596.67 | 1596.12 | 2.89 | 1.67 | 0.78 | 36.66 |
| Cancer | dense | ok | 100 | 649.23 | 630.47 | 33.25 | 6.96 | 6.14 | 34.63 |
| Cancer | sparse | ok | 100 | 698.30 | 696.90 | 6.62 | 1.39 | 0.54 | 34.01 |
