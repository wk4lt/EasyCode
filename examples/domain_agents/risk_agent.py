"""Risk Agent for order risk assessment.

Domain agent that evaluates orders using the risk_check skill and produces
a risk assessment with recommended actions.

Layer: Agent layer (second layer).
"""

from liteagent.core.base_agent import BaseAgent


class RiskAgent(BaseAgent):
    """Agent specialized in order risk assessment."""

    def _build_user_message(self, local_input: dict) -> str:
        """Format the order details as a risk assessment request.

        Args:
            local_input: Dict with order fields from the input_mapper.

        Returns:
            Formatted risk assessment request string.
        """
        order_id = local_input.get("order_id", "UNKNOWN")
        amount = local_input.get("amount", 0)
        customer_tier = local_input.get("customer_tier", "new")
        region = local_input.get("region", "NA")

        return (
            f"Please evaluate the risk level for the following order:\n\n"
            f"  Order ID: {order_id}\n"
            f"  Amount: ${amount:.2f}\n"
            f"  Customer Tier: {customer_tier}\n"
            f"  Region: {region}\n\n"
            f"Use the risk_check tool to assess this order."
        )
