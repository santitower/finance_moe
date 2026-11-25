"""Pydantic models for portfolio input data."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class Holding(BaseModel):
    ticker: str = Field(description="Ticker symbol, e.g. 'AAPL'")
    shares: float = Field(description="Number of shares held")
    cost_basis: float = Field(description="Average cost per share")
    current_price: Optional[float] = Field(None, description="Current market price")
    sector: Optional[str] = Field(None, description="Sector for concentration analysis")


class Account(BaseModel):
    type: str = Field(
        description="Account type: 401k, IRA, Roth, taxable, hsa, brokerage, 403b"
    )
    balance: float = Field(description="Current total balance")
    expense_ratio: float = Field(
        default=0.0, ge=0.0, description="Fund-level expense ratio (0.001 = 0.1%)"
    )
    holdings: list[Holding] = Field(
        default_factory=list, description="Individual securities"
    )
    fund_tickers: list[str] = Field(
        default_factory=list,
        description="Fund tickers if holding ETFs/mutual funds",
    )

    @property
    def total_value(self) -> float:
        if self.holdings:
            return sum(
                h.shares * (h.current_price or h.cost_basis) for h in self.holdings
            )
        return self.balance


class Income(BaseModel):
    salary: float = Field(description="Annual salary")
    other: float = Field(default=0.0, description="Other annual income")
    frequency: str = Field(default="annual", description="Pay frequency")

    @property
    def total(self) -> float:
        return self.salary + self.other


class Expenses(BaseModel):
    fixed: float = Field(description="Monthly fixed expenses")
    variable: float = Field(default=0.0, description="Monthly variable expenses")

    @property
    def total_monthly(self) -> float:
        return self.fixed + self.variable

    @property
    def total_annual(self) -> float:
        return self.total_monthly * 12


class TaxInfo(BaseModel):
    filing_status: str = Field(description="single, married_joint, head_of_household")
    state: Optional[str] = Field(None, description="US state for state tax")
    deductions_taken: str = Field(
        default="standard", description="standard or itemized"
    )
    itemized_amount: Optional[float] = Field(None)


class PortfolioInput(BaseModel):
    """Top-level input for the finance MoE system."""

    accounts: list[Account]
    income: Income
    expenses: Expenses
    tax: TaxInfo = Field(default_factory=TaxInfo)
    risk_tolerance: float = Field(
        default=0.5, ge=0.0, le=1.0, description="0=conservative, 1=aggressive"
    )
    time_horizon_years: int = Field(default=20, ge=1, le=60)
    retirement_goal: Optional[float] = Field(
        None, description="Target retirement balance"
    )

    @property
    def total_portfolio_value(self) -> float:
        return sum(a.total_value for a in self.accounts)

    @property
    def monthly_savings(self) -> float:
        return self.income.total / 12 - self.expenses.total_monthly

    @property
    def savings_rate(self) -> float:
        return self.monthly_savings / (self.income.total / 12) if self.income.total > 0 else 0

    @property
    def emergency_runway_months(self) -> float:
        taxable = sum(a.balance for a in self.accounts if a.type == "taxable")
        return taxable / self.expenses.total_monthly if self.expenses.total_monthly > 0 else 0

    def model_dump_clean(self) -> dict:
        """Serialize, removing None values."""
        d = self.model_dump()
        return {k: v for k, v in d.items() if v is not None}
