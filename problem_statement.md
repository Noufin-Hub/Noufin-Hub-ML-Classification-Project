# Predicting Complaint Escalation for Mobile Subscribers

**Stream:** Telecom  |  **Task type:** Classification

## Background

Zenith Mobile Networks wants to resolve complaints before they reach regulators. The service team needs a model predicting whether a subscriber's complaint will be escalated (Yes/No), using subscriber profile data, call quality logs and complaint ticket logs, to route risky cases to senior agents.

## Your Deliverable

Build a classification model predicting escalated (Yes/No) for each subscriber using the subscriber profile, call quality logs and complaint ticket logs.

## Files Provided

- subscriber\_profile.csv  
- call\_quality\_logs.csv  
- complaint\_tickets.csv  
- escalated\_labeled.csv  
- escalated\_to\_predict.csv

All tables that share the subscriber\_id column can be joined on it. Not every file is already clean or already in the shape you need \-- inspect each one before you start modeling.

Predictions are required for every subscriber\_id listed in the \*\_to\_predict.csv file (these entities have no label in the training data).

## Suggested Steps

1. Inspect subscriber\_profile, call\_quality\_logs, complaint\_tickets; find the shared key subscriber\_id.  
2. Aggregate the repeated-measurement tables per entity (counts, sums, means, max).  
3. Merge the aggregated features with subscriber\_profile on subscriber\_id.  
4. Handle missing values (segment, days\_open) and outliers in latency\_ms.  
5. Encode categorical columns (plan\_type, region, segment, issue\_type).  
6. Min-max normalize numeric features before modeling.  
7. Train a classifier for escalated; predict for ids in \_to\_predict.csv.  
8. Report precision/recall/F1 and discuss the business cost of the two error types.

## Notes

- At minimum, apply min-max normalization to your numeric features before modeling; you're welcome to also explore standardization and compare.  
- Document your merge keys, cleaning decisions, and any assumptions in your submission.