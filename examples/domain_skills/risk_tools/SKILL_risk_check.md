# Skill: risk_check

## Description
Check the risk level of an order by evaluating the order amount, customer tier, and destination region against known risk patterns. Returns a risk score from 0 (safe) to 100 (high risk).

## Parameters
- order_id (string): The unique order identifier. Required.
- amount (number): The order amount in USD. Required.
- customer_tier (string): Customer tier: new, bronze, silver, gold. Required.
- region (string): Destination region code: NA, EU, AS, AF, SA, OC. Required.

## Boundaries
- Amount range: 0 to 1000000
- Customer tier must be one of: new, bronze, silver, gold
- Region must be one of: NA, EU, AS, AF, SA, OC
- Risk assessment is pattern-based, not real-time fraud detection
