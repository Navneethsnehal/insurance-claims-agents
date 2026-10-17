# Evaluation report

Provider: `fake`. Cases: 17. Fraud holdout: 1200 rows.

| Metric | Result | Gate |
| --- | ---: | ---: |
| routing_accuracy | 1.0000 | 0.90 |
| coverage_accuracy | 1.0000 | 0.85 |
| citation_faithfulness | 1.0000 | 0.90 |
| payout_accuracy | 1.0000 | 0.90 |
| fraud_roc_auc | 0.9037 | 0.80 |
| fraud_precision_at_k | 0.9700 | 0.70 |
| retrieval_recall | 1.0000 | report |
| pii_leaks | 0 | 0 |
| mean_latency_ms | 12.15 | report |
| p50_latency_ms | 11.33 | report |
| mean_input_tokens | 266.5 | report |
| mean_output_tokens | 216.4 | report |
| total_cost_usd | 0.000000 | report |

Gate result: **pass**.

## Cases

| Claim | Route | Coverage | Payout | Recall | Latency ms |
| --- | --- | --- | --- | ---: | ---: |
| CLM-1001 | ok | covered | ok | 1.00 | 23.3 |
| CLM-1002 | ok | covered | ok | 1.00 | 14.7 |
| CLM-1003 | ok | covered | ok | 1.00 | 12.0 |
| CLM-1004 | ok | excluded | ok | 1.00 | 11.7 |
| CLM-1005 | ok | excluded | ok | 1.00 | 11.2 |
| CLM-1006 | ok | excluded | ok | 1.00 | 11.9 |
| CLM-1007 | ok | excluded | ok | 1.00 | 11.1 |
| CLM-1008 | ok | partial | ok | 1.00 | 10.4 |
| CLM-1009 | ok | excluded | ok | 1.00 | 11.4 |
| CLM-1010 | ok | covered | ok | 1.00 | 11.2 |
| CLM-1011 | ok | covered | ok | 1.00 | 11.1 |
| CLM-1012 | ok | excluded | ok | 1.00 | 11.0 |
| CLM-1013 | ok | partial | ok | 1.00 | 11.7 |
| CLM-1014 | ok | covered | ok | 1.00 | 9.8 |
| CLM-1015 | ok | excluded | ok | 1.00 | 11.3 |
| CLM-1016 | ok | covered | ok | 1.00 | 10.8 |
| CLM-1017 | ok | covered | ok | 1.00 | 11.7 |
