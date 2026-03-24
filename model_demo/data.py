from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

AGE_VALUES = [24, 28, 32, 38, 45, 52]
INCOME_VALUES = [7000, 9000, 11000, 14000]
EXISTING_VALUES = [0, 1]

AGE_BIN_ORDER = ["Under 35", "35 to 49", "50 and above"]
INCOME_BIN_ORDER = ["Under 9k", "9k to 10.9k", "11k to 12.9k", "13k and above"]
EXISTING_STATUS_ORDER = ["New customer", "Existing customer"]

SCENARIO_LABELS = {
    "single_cluster": "Single Positive Region",
    "two_clusters": "Two Positive Regions",
}

SCENARIO_DESCRIPTIONS = {
    "single_cluster": (
        "One strong interaction rule. Positive only if age is under 35 and income is at least 11,000. "
        "Existing-customer status is present in the data but does not drive the target."
    ),
    "two_clusters": (
        "Two different positive regions. Positive if age is under 35 and income is at least 11,000, "
        "or if the customer is existing and income is at most 9,000."
    ),
}


@dataclass(frozen=True)
class ProbeCustomer:
    age: int
    income: int
    existing_customer: int
    age_bin: str
    income_bin: str
    existing_status: str


def existing_status_for(existing_customer: int) -> str:
    return EXISTING_STATUS_ORDER[1] if existing_customer else EXISTING_STATUS_ORDER[0]


def outcome_rule(
        age: float,
        income: float,
        existing_customer: int,
        scenario: str,
) -> int:
    if scenario == "two_clusters":
        return int(
            (age < 35 and income >= 11000)
            or (existing_customer == 1 and income <= 9000)
        )
    return int(age < 35 and income >= 11000)


def age_bin_for(age: float) -> str:
    if age < 35:
        return AGE_BIN_ORDER[0]
    if age < 50:
        return AGE_BIN_ORDER[1]
    return AGE_BIN_ORDER[2]


def income_bin_for(income: float) -> str:
    if income < 9000:
        return INCOME_BIN_ORDER[0]
    if income < 11000:
        return INCOME_BIN_ORDER[1]
    if income < 13000:
        return INCOME_BIN_ORDER[2]
    return INCOME_BIN_ORDER[3]


def build_probe_customer(age: int, income: int, existing_customer: int) -> ProbeCustomer:
    return ProbeCustomer(
        age=age,
        income=income,
        existing_customer=existing_customer,
        age_bin=age_bin_for(age),
        income_bin=income_bin_for(income),
        existing_status=existing_status_for(existing_customer),
    )


def build_training_data(scenario: str = "single_cluster") -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    customer_number = 1

    for age in AGE_VALUES:
        for income in INCOME_VALUES:
            for existing_customer in EXISTING_VALUES:
                accepted = outcome_rule(age, income, existing_customer, scenario)
                rows.append(
                    {
                        "customer_id": f"C{customer_number:02d}",
                        "age": age,
                        "income": income,
                        "existing_customer": existing_customer,
                        "existing_status": existing_status_for(existing_customer),
                        "age_bin": age_bin_for(age),
                        "income_bin": income_bin_for(income),
                        "accepted": accepted,
                        "accepted_label": "Accepted" if accepted else "Rejected",
                    }
                )
                customer_number += 1

    frame = pd.DataFrame(rows)
    frame["existing_status"] = pd.Categorical(
        frame["existing_status"], categories=EXISTING_STATUS_ORDER, ordered=True
    )
    frame["age_bin"] = pd.Categorical(
        frame["age_bin"], categories=AGE_BIN_ORDER, ordered=True
    )
    frame["income_bin"] = pd.Categorical(
        frame["income_bin"], categories=INCOME_BIN_ORDER, ordered=True
    )
    return frame


def build_rule_regions(scenario: str) -> pd.DataFrame:
    rows = [
        {
            "existing_status": "New customer",
            "x1": 22,
            "x2": 34,
            "y1": 10500,
            "y2": 14500,
            "rule_label": "Young + higher income",
        },
        {
            "existing_status": "Existing customer",
            "x1": 22,
            "x2": 34,
            "y1": 10500,
            "y2": 14500,
            "rule_label": "Young + higher income",
        },
    ]

    if scenario == "two_clusters":
        rows.append(
            {
                "existing_status": "Existing customer",
                "x1": 22,
                "x2": 55,
                "y1": 6500,
                "y2": 9000,
                "rule_label": "Existing + lower income",
            }
        )

    return pd.DataFrame(rows)


def pretty_training_table(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rename(
        columns={
            "customer_id": "Customer",
            "age": "Age",
            "income": "Income",
            "existing_status": "Existing customer",
            "age_bin": "Age band",
            "income_bin": "Income band",
            "accepted_label": "Outcome",
        }
    )[
        [
            "Customer",
            "Age",
            "Income",
            "Existing customer",
            "Age band",
            "Income band",
            "Outcome",
        ]
    ]
