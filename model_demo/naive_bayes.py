from __future__ import annotations

from dataclasses import dataclass
from math import exp, log

import pandas as pd

from model_demo.data import ProbeCustomer

FEATURE_ORDER = ["age_bin", "income_bin", "existing_status"]
FEATURE_LABELS = {
    "age_bin": "Age band",
    "income_bin": "Income band",
    "existing_status": "Existing customer",
}


@dataclass
class NaiveBayesModel:
    prior_log_odds: float
    positives: int
    negatives: int
    feature_tables: dict[str, pd.DataFrame]


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + exp(-value))


def train_naive_bayes(frame: pd.DataFrame) -> NaiveBayesModel:
    positives = int(frame["accepted"].sum())
    negatives = int(len(frame) - positives)
    prior_log_odds = log((positives + 0.5) / (negatives + 0.5))

    feature_tables: dict[str, pd.DataFrame] = {}

    for feature in FEATURE_ORDER:
        categories = list(frame[feature].cat.categories)
        n_values = len(categories)
        alpha = 1.0 / n_values
        grouped = (
            frame.groupby([feature, "accepted"], observed=False)
            .size()
            .unstack(fill_value=0)
            .reindex(categories, fill_value=0)
        )

        rows = []
        for value in categories:
            positive_count = int(grouped.loc[value].get(1, 0))
            negative_count = int(grouped.loc[value].get(0, 0))

            positive_prob = (positive_count + alpha) / (positives + alpha * n_values)
            negative_prob = (negative_count + alpha) / (negatives + alpha * n_values)
            contribution = log(positive_prob) - log(negative_prob)

            rows.append(
                {
                    "value": value,
                    "positives": positive_count,
                    "negatives": negative_count,
                    "positive_rate": positive_count / positives if positives else 0.0,
                    "negative_rate": negative_count / negatives if negatives else 0.0,
                    "positive_probability": positive_prob,
                    "negative_probability": negative_prob,
                    "contribution": contribution,
                }
            )

        feature_tables[feature] = pd.DataFrame(rows)

    return NaiveBayesModel(
        prior_log_odds=prior_log_odds,
        positives=positives,
        negatives=negatives,
        feature_tables=feature_tables,
    )


def feature_summary_table(model: NaiveBayesModel, feature: str) -> pd.DataFrame:
    table = model.feature_tables[feature].copy()
    table["Contribution"] = table["contribution"].round(3)
    table["Positives"] = table["positives"]
    table["Negatives"] = table["negatives"]
    table["Positive share"] = table["positive_rate"].round(3)
    table["Negative share"] = table["negative_rate"].round(3)
    return table.rename(columns={"value": FEATURE_LABELS[feature]})[
        [
            FEATURE_LABELS[feature],
            "Positives",
            "Negatives",
            "Positive share",
            "Negative share",
            "Contribution",
        ]
    ]


def score_customer(model: NaiveBayesModel, customer: ProbeCustomer) -> dict[str, object]:
    selected_values = {
        "age_bin": customer.age_bin,
        "income_bin": customer.income_bin,
        "existing_status": customer.existing_status,
    }

    contributions = []
    raw_score = model.prior_log_odds

    for feature in FEATURE_ORDER:
        selected_value = selected_values[feature]
        table = model.feature_tables[feature]
        row = table.loc[table["value"] == selected_value].iloc[0]
        contribution = float(row["contribution"])
        raw_score += contribution
        probability_after_feature = sigmoid(raw_score)
        contributions.append(
            {
                "Feature": FEATURE_LABELS[feature],
                "Selected value": selected_value,
                "Smoothed P(bin|accepted)": float(row["positive_probability"]),
                "Smoothed P(bin|rejected)": float(row["negative_probability"]),
                "Contribution": contribution,
                "Raw score after feature": raw_score,
                "Propensity after feature": probability_after_feature,
                "Positives in bin": int(row["positives"]),
                "Negatives in bin": int(row["negatives"]),
            }
        )

    contribution_frame = pd.DataFrame(contributions)
    probability = sigmoid(raw_score)

    return {
        "prior_log_odds": model.prior_log_odds,
        "raw_score": raw_score,
        "probability": probability,
        "contributions": contribution_frame,
    }
