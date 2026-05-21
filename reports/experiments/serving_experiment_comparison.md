# AML Serving API — Threshold Experiment Comparison

MLflow experiment: `aml-serving-threshold-comparison`  
SLA target: P95 < 200.0 ms

| experiment | threshold | mean_ms | std_ms | p50_ms | p95_ms | p99_ms | throughput_rps | flagged_rate | sla_passed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| conservative | 0.3 | 1.46 | 0.42 | 1.33 | 2.18 | 3.20 | 684.6 | 100.0% | ✓ |
| balanced | 0.5 | 1.53 | 2.10 | 1.35 | 1.88 | 2.14 | 652.1 | 100.0% | ✓ |
| strict | 0.7 | 1.47 | 0.24 | 1.40 | 1.87 | 2.55 | 679.5 | 0.0% | ✓ |
