# ADR 0004: Fraud score is the max of the model and the rules; explanations are ablations

## Status

Accepted

## Context

Special investigations wants two different signals. A gradient-boosting model picks up combinations in a synthetic history (report lag, prior referrals, repair ratio, attorney involvement). A rule catches a file the model has not seen often enough: a prior SIU referral, a high-value loss with no police report, an inconsistent statement. Missing either one is a bad triage.

Reviewers also need a reason. TreeSHAP is the usual answer and it is a heavy, version-sensitive dependency for a model this small.

## Decision

The published score is `min(1, max(model_score, rule_score))`. A rule hit at `0.80` escalates even when the model is calm. The model can escalate on its own when no rule fires.

Rules and weights live in `fraud/rules.py`: early high value `0.72`, delayed report `0.55`, inconsistent statements `0.70`, prior SIU `0.80`, no police report on a high-value loss `0.60`, low damage with an attorney `0.66`, repeating provider `0.58`. The rule score is the max weight, not a sum, so stacked weak rules do not automatically cross `0.65`.

Explanations score the training-median baseline, then replace one feature at a time with the claim's value. The change in predicted probability is the contribution. Direction is `increases_risk`, `decreases_risk`, or `neutral`.

Training is seeded. `load_bundle` retrains when the artifact is missing, fails to unpickle, or was fit on a different feature list. The holdout metrics travel with the artifact in `data/models/fraud_gbm.metrics.json`.

Label noise is applied after features are sampled from the true class. Noise that is applied before sampling is not noise.

## Consequences

SIU recall is favored over a calibrated probability. The score is not a frequency. A `0.80` from a rule and a `0.80` from the model mean different things, which is why the API returns `model_score`, `rule_score`, and `flags` separately.

Ablations miss interactions. Two features that only matter together can each look neutral. The memo still shows the rule flags, which are the interactions we decided to write down. TreeSHAP can replace `explain_contributions` later without a change to the score contract.

The training set is synthetic. The AUC in the eval report measures separation on that generator's holdout. It is not a claim about a production book.

## Alternatives

A sum of rule weights was rejected because it turns three mild flags into an automatic referral. A pure model score was rejected because a prior SIU referral has to escalate even if the trees have not split on it yet. Shipping `shap` was rejected to keep CI small and the explanation function unit-testable with a stub estimator.
