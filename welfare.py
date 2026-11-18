"""
Consumer value of a subscription service and the incidence of a levy on subscription revenue.

A streaming service holds the inputs internally: members, average monthly revenue per member,
and the response of sign-ups and cancellations to past price changes (the price elasticity of
membership). This module turns them into:

    consumer surplus   the value members get above what they pay
    levy incidence     who bears a levy on subscription revenue: members, through higher prices
                       and lost surplus; the service, through lower net revenue; and the excess
                       burden, the loss that no one collects

Demand is linear through the current price and membership with the stated point elasticity. This
is the conservative choice: a constant-elasticity curve gives a much larger consumer surplus for
the same elasticity. All values are annual.

Marginal cost per member (payments, delivery, customer service) is set by default from the pricing
condition of a service that sets its price to maximise profit: (P - MC) / P = 1 / elasticity. This
keeps the price, elasticity and cost assumptions consistent with each other; an elasticity above 1
with a marginal cost of zero would imply the service could raise revenue by cutting its price.
On the same basis the module reports the share of a levy a profit-maximising service would pass on.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Membership:
    members_m: float          # members, millions
    monthly_price: float      # average monthly revenue per member, £
    elasticity: float         # point price elasticity of membership (positive number)
    marginal_cost_share: float | None = None  # cost of serving one more member, share of price;
                                              # None sets it from the pricing condition
    def __post_init__(self):
        if self.members_m <= 0 or self.monthly_price <= 0 or self.elasticity <= 0:
            raise ValueError("members, price and elasticity must be positive")
        if self.marginal_cost_share is None:
            self.marginal_cost_share = self.lerner_marginal_cost_share()
        if not 0.0 <= self.marginal_cost_share < 1.0:
            raise ValueError("marginal_cost_share must be at least 0 and below 1")

    def lerner_marginal_cost_share(self) -> float:
        """Marginal cost as a share of price for a profit-maximising price: 1 - 1/e (0 if e <= 1)."""
        return max(0.0, 1.0 - 1.0 / self.elasticity)

    @property
    def revenue_m(self) -> float:
        return 12 * self.members_m * self.monthly_price

    def consumer_surplus_m(self) -> float:
        """Linear demand: CS = Q x P / (2e), annualised, £m."""
        return 12 * self.members_m * self.monthly_price / (2 * self.elasticity)

    def consumer_surplus_constant_elasticity_m(self) -> float:
        """Constant-elasticity demand (upper bound, needs e > 1): CS = P Q / (e - 1)."""
        if self.elasticity <= 1:
            return float("inf")
        return self.revenue_m / (self.elasticity - 1)


def levy_incidence(m: Membership, levy_rate: float, pass_through: float) -> dict:
    """Levy charged on gross subscription revenue; the service raises its price by
    pass_through x levy_rate x price. Returns annual £m and members lost (millions)."""
    P, Q, e = m.monthly_price, m.members_m, m.elasticity
    P1 = P * (1 + pass_through * levy_rate)
    Q1 = Q * (1 - e * (P1 - P) / P)
    levy = 12 * levy_rate * P1 * Q1
    cs_loss = 12 * ((P1 - P) * Q1 + 0.5 * (P1 - P) * (Q - Q1))
    cost_saved = 12 * m.marginal_cost_share * P * (Q - Q1)
    service_loss = m.revenue_m - 12 * P1 * Q1 * (1 - levy_rate) - cost_saved
    excess = cs_loss + service_loss - levy
    return {
        "levy_rate": levy_rate,
        "pass_through": pass_through,
        "new_price": P1,
        "members_lost_m": Q - Q1,
        "levy_raised_m": levy,
        "borne_by_members_m": cs_loss,
        "borne_by_service_m": service_loss,
        "excess_burden_m": excess,
        "excess_burden_per_pound_raised": excess / levy if levy else float("nan"),
    }


def profit_max_pass_through(m: Membership, levy_rate: float) -> float:
    """Share of a levy on revenue that a profit-maximising service facing linear demand passes on
    in its price. With demand Q = a - bP and constant marginal cost c, the price rises from
    a/2b + c/2 to a/2b + c/(2(1 - levy)), so pass-through = c / (2 (1 - levy) P). A service whose
    marginal cost is close to zero passes on little of a revenue levy, because the levy takes the
    same share of revenue at every price."""
    c = m.marginal_cost_share * m.monthly_price
    return c / (2 * (1 - levy_rate) * m.monthly_price)
