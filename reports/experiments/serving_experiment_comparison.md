# AML Serving API — Threshold Experiment Comparison

MLflow experiment: `aml-serving-threshold-comparison`  
SLA target: P95 < 200.0 ms

| experiment | threshold | mean_ms | std_ms | p50_ms | p95_ms | p99_ms | throughput_rps | flagged_rate | sla_passed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| conservative | 0.3 | 1.91 | 1.24 | 1.56 | 3.57 | 7.31 | 523.2 | 100.0% | ✓ |
| balanced | 0.5 | 3.80 | 13.59 | 1.97 | 8.09 | 21.04 | 263.2 | 100.0% | ✓ |
| strict | 0.7 | 1.89 | 0.86 | 1.58 | 3.36 | 4.61 | 530.3 | 0.0% | ✓ |
