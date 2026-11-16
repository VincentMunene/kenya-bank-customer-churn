"""Request and response schemas for the churn API."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


YesNo = Literal["Yes", "No"]
Text = Annotated[str, Field(min_length=1, max_length=120)]
Amount = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class CustomerInput(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    customer_id: Text
    tenure_months: int = Field(ge=0, strict=True)

    monthly_transaction_value_kes: Amount
    total_relationship_value_kes: Amount

    debit_card: YesNo
    mobile_banking: YesNo
    internet_banking: YesNo
    sms_alerts: YesNo
    standing_orders: YesNo
    loan_product: YesNo
    savings_product: YesNo
    insurance_product: YesNo
    e_statements: YesNo

    primary_account: Text
    customer_segment: Text
    preferred_transaction_channel: Text
    county: Text


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customers: list[CustomerInput] = Field(
        min_length=1,
        max_length=1000,
    )


class CustomerPrediction(BaseModel):
    customer_id: str
    p_churn: float = Field(ge=0, le=1, allow_inf_nan=False)
    risk_band: Literal["Low", "Medium", "High"]


class PredictionResponse(BaseModel):
    predictions: list[CustomerPrediction]