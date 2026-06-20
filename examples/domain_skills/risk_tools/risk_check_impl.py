"""Risk assessment skill implementation.

Evaluates orders against known risk patterns to produce a 0-100 risk score.
Uses a rule-based engine with configurable thresholds.

Layer: Skill layer (first layer).
"""

from liteagent.core.base_skill import BaseSkill


class RiskCheckImpl(BaseSkill):
    """Evaluate order risk using a pattern-based rules engine."""

    VALID_TIERS = {"new", "bronze", "silver", "gold"}
    VALID_REGIONS = {"NA", "EU", "AS", "AF", "SA", "OC"}

    HIGH_RISK_REGIONS = {"AF", "SA", "AS"}
    MEDIUM_RISK_REGIONS = {"OC"}

    TIER_WEIGHTS = {
        "new": 30,
        "bronze": 15,
        "silver": 5,
        "gold": 0,
    }

    def execute(
        self,
        order_id: str,
        amount: float,
        customer_tier: str,
        region: str,
    ) -> dict:
        """Execute risk assessment for an order.

        Args:
            order_id: Unique order identifier.
            amount: Order amount in USD.
            customer_tier: Customer tier (new, bronze, silver, gold).
            region: Destination region code (NA, EU, AS, AF, SA, OC).

        Returns:
            dict with 'status' and risk assessment or 'error'.
        """
        try:
            customer_tier = customer_tier.lower().strip()
            region = region.upper().strip()

            if customer_tier not in self.VALID_TIERS:
                return {
                    "status": "error",
                    "error": f"Invalid customer_tier '{customer_tier}'. Must be one of: {', '.join(sorted(self.VALID_TIERS))}",
                }

            if region not in self.VALID_REGIONS:
                return {
                    "status": "error",
                    "error": f"Invalid region '{region}'. Must be one of: {', '.join(sorted(self.VALID_REGIONS))}",
                }

            if not (0 <= amount <= 1000000):
                return {
                    "status": "error",
                    "error": f"Amount {amount} out of valid range (0-1000000).",
                }

            risk_score = 0
            flags = []

            tier_weight = self.TIER_WEIGHTS.get(customer_tier, 15)
            risk_score += tier_weight
            if tier_weight >= 30:
                flags.append("new_customer_high_risk")
            elif tier_weight >= 15:
                flags.append("bronze_tier_elevated_risk")

            if region in self.HIGH_RISK_REGIONS:
                risk_score += 30
                flags.append(f"high_risk_region:{region}")
            elif region in self.MEDIUM_RISK_REGIONS:
                risk_score += 15
                flags.append(f"medium_risk_region:{region}")

            if amount > 10000:
                risk_score += 25
                flags.append("high_value_order")
            elif amount > 5000:
                risk_score += 10
                flags.append("elevated_value_order")

            if amount < 1:
                risk_score += 10
                flags.append("micro_transaction")

            risk_score = min(risk_score, 100)

            level = "low"
            if risk_score >= 70:
                level = "critical"
            elif risk_score >= 50:
                level = "high"
            elif risk_score >= 30:
                level = "medium"

            return {
                "status": "ok",
                "order_id": order_id,
                "risk_score": risk_score,
                "risk_level": level,
                "flags": flags,
                "factors": {
                    "customer_tier": customer_tier,
                    "region": region,
                    "amount": amount,
                },
            }

        except Exception as e:
            return {"status": "error", "error": f"Risk check failed: {e}"}
