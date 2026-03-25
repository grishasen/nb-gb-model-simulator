from __future__ import annotations

from dataclasses import dataclass
from math import inf, log, sqrt

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
    base_log_odds: float
    positives: int
    negatives: int
    feature_tables: dict[str, pd.DataFrame]
    active_predictor_count: int
    classifier_table: pd.DataFrame


def z_ratio(
        positive_count: int,
        negative_count: int,
        total_positives: int,
        total_negatives: int,
) -> float:
    if total_positives == 0 or total_negatives == 0:
        return 0.0

    positive_share = positive_count / total_positives
    negative_share = negative_count / total_negatives
    variance = (
        (positive_share * (1.0 - positive_share) / total_positives)
        + (negative_share * (1.0 - negative_share) / total_negatives)
    )
    if variance <= 0:
        if positive_share == negative_share:
            return 0.0
        return float("inf") if positive_share > negative_share else float("-inf")
    return (positive_share - negative_share) / sqrt(variance)


def _format_score_bin(lower_bound: float, upper_bound: float) -> str:
    if lower_bound == -inf:
        return f"< {upper_bound:.3f}"
    if upper_bound == inf:
        return f">= {lower_bound:.3f}"
    return f"[{lower_bound:.3f}, {upper_bound:.3f})"


def _score_frame_with_model(
        feature_tables: dict[str, pd.DataFrame],
        base_log_odds: float,
        active_predictor_count: int,
        frame: pd.DataFrame,
) -> pd.DataFrame:
    scored = frame.copy()
    numerator = pd.Series(base_log_odds, index=scored.index, dtype=float)

    for feature in FEATURE_ORDER:
        contribution_lookup = feature_tables[feature].set_index("value")["contribution"]
        numerator = numerator + scored[feature].map(contribution_lookup).astype(float)

    scored["nb_score"] = numerator / (1 + active_predictor_count)
    return scored


def _build_classifier_table(
        positives: int,
        negatives: int,
        scored_frame: pd.DataFrame,
) -> pd.DataFrame:
    sorted_rows = (
        scored_frame.groupby("nb_score", as_index=False)["accepted"]
        .agg(["sum", "count"])
        .reset_index()
        .rename(columns={"sum": "positives", "count": "count"})
        .sort_values("nb_score")
        .reset_index(drop=True)
    )
    blocks: list[dict[str, float | int]] = []

    for row in sorted_rows.itertuples(index=False):
        block = {
            "positives": int(row.positives),
            "negatives": int(row.count - row.positives),
            "count": int(row.count),
            "min_score": float(row.nb_score),
            "max_score": float(row.nb_score),
        }
        blocks.append(block)

        while len(blocks) >= 2:
            left = blocks[-2]
            right = blocks[-1]
            left_rate = left["positives"] / left["count"]
            right_rate = right["positives"] / right["count"]
            if left_rate <= right_rate:
                break

            merged = {
                "positives": left["positives"] + right["positives"],
                "negatives": left["negatives"] + right["negatives"],
                "count": left["count"] + right["count"],
                "min_score": left["min_score"],
                "max_score": right["max_score"],
            }
            blocks[-2:] = [merged]

    merged_blocks: list[dict[str, float | int]] = []
    for block in blocks:
        if not merged_blocks:
            merged_blocks.append(block)
            continue

        previous = merged_blocks[-1]
        previous_rate = previous["positives"] / previous["count"]
        current_rate = block["positives"] / block["count"]
        if abs(previous_rate - current_rate) < 1e-12:
            merged_blocks[-1] = {
                "positives": previous["positives"] + block["positives"],
                "negatives": previous["negatives"] + block["negatives"],
                "count": previous["count"] + block["count"],
                "min_score": previous["min_score"],
                "max_score": block["max_score"],
            }
        else:
            merged_blocks.append(block)

    blocks = merged_blocks

    classifier_rows: list[dict[str, float | int | str]] = []
    cumulative_count = 0
    cumulative_positives = 0
    total_count = positives + negatives

    for index, block in enumerate(blocks):
        previous_block = blocks[index - 1] if index > 0 else None
        next_block = blocks[index + 1] if index < len(blocks) - 1 else None
        lower_bound = (
            -inf
            if previous_block is None
            else (previous_block["max_score"] + block["min_score"]) / 2
        )
        upper_bound = (
            inf
            if next_block is None
            else (block["max_score"] + next_block["min_score"]) / 2
        )

        block_count = int(block["count"])
        block_positives = int(block["positives"])
        block_negatives = int(block["negatives"])
        cumulative_count += block_count
        cumulative_positives += block_positives

        propensity = block_positives / block_count if block_count else 0.0
        adjusted_propensity = (0.5 + block_positives) / (1 + block_count)
        lift = propensity / (positives / total_count) if positives and total_count else 0.0

        classifier_rows.append(
            {
                "Bin": _format_score_bin(lower_bound, upper_bound),
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "Positives": block_positives,
                "Negatives": block_negatives,
                "Responses %": block_count / total_count if total_count else 0.0,
                "Cum. total %": cumulative_count / total_count if total_count else 0.0,
                "Propensity": propensity,
                "Adjusted propensity": adjusted_propensity,
                "Cum. positives %": (
                    cumulative_positives / positives if positives else 0.0
                ),
                "Z-ratio": z_ratio(
                    block_positives,
                    block_negatives,
                    positives,
                    negatives,
                ),
                "Lift": lift,
            }
        )

    return pd.DataFrame(classifier_rows)


def _match_classifier_row(classifier_table: pd.DataFrame, score: float) -> pd.Series:
    matches = classifier_table.loc[
        (classifier_table["lower_bound"] <= score)
        & (score < classifier_table["upper_bound"])
    ]
    if matches.empty:
        return classifier_table.iloc[-1]
    return matches.iloc[0]


def train_naive_bayes(frame: pd.DataFrame) -> NaiveBayesModel:
    positives = int(frame["accepted"].sum())
    negatives = int(len(frame) - positives)
    total_count = positives + negatives
    base_log_odds = log(positives + 1) - log(negatives + 1)

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
            response_count = positive_count + negative_count

            positive_prob = (positive_count + alpha) / (positives + 1)
            negative_prob = (negative_count + alpha) / (negatives + 1)
            contribution = log(positive_prob) - log(negative_prob)

            rows.append(
                {
                    "value": value,
                    "positives": positive_count,
                    "negatives": negative_count,
                    "responses": response_count,
                    "response_rate": response_count / total_count if total_count else 0.0,
                    "positive_rate_of_total": (
                        positive_count / total_count if total_count else 0.0
                    ),
                    "negative_rate_of_total": (
                        negative_count / total_count if total_count else 0.0
                    ),
                    "positive_share": positive_count / positives if positives else 0.0,
                    "negative_share": negative_count / negatives if negatives else 0.0,
                    "propensity": positive_count / response_count if response_count else 0.0,
                    "adjusted_propensity": (
                        (positive_count + 0.5) / (response_count + 1)
                        if response_count
                        else 0.5
                    ),
                    "z_ratio": z_ratio(
                        positive_count,
                        negative_count,
                        positives,
                        negatives,
                    ),
                    "positive_probability": positive_prob,
                    "negative_probability": negative_prob,
                    "contribution": contribution,
                }
            )

        feature_tables[feature] = pd.DataFrame(rows)

    scored_frame = _score_frame_with_model(
        feature_tables,
        base_log_odds,
        len(FEATURE_ORDER),
        frame,
    )
    classifier_table = _build_classifier_table(positives, negatives, scored_frame)

    return NaiveBayesModel(
        base_log_odds=base_log_odds,
        positives=positives,
        negatives=negatives,
        feature_tables=feature_tables,
        active_predictor_count=len(FEATURE_ORDER),
        classifier_table=classifier_table,
    )


def feature_summary_table(model: NaiveBayesModel, feature: str) -> pd.DataFrame:
    table = model.feature_tables[feature].copy()
    table["Responses"] = table["responses"]
    table["Responses %"] = table["response_rate"].round(3)
    table["Accepted % of total"] = table["positive_rate_of_total"].round(3)
    table["Rejected % of total"] = table["negative_rate_of_total"].round(3)
    table["Accepted % of accepted"] = table["positive_share"].round(3)
    table["Rejected % of rejected"] = table["negative_share"].round(3)
    table["Bin propensity"] = table["propensity"].round(3)
    table["Adjusted propensity"] = table["adjusted_propensity"].round(3)
    table["Z-ratio"] = table["z_ratio"].round(3)
    table["Contribution"] = table["contribution"].round(3)
    return table.rename(columns={"value": FEATURE_LABELS[feature]})[
        [
            FEATURE_LABELS[feature],
            "Responses",
            "Responses %",
            "positives",
            "negatives",
            "Accepted % of total",
            "Rejected % of total",
            "Accepted % of accepted",
            "Rejected % of rejected",
            "Bin propensity",
            "Adjusted propensity",
            "Z-ratio",
            "Contribution",
        ]
    ].rename(columns={"positives": "Positives", "negatives": "Negatives"})


def score_customer(model: NaiveBayesModel, customer: ProbeCustomer) -> dict[str, object]:
    selected_values = {
        "age_bin": customer.age_bin,
        "income_bin": customer.income_bin,
        "existing_status": customer.existing_status,
    }

    contributions = []
    running_numerator = model.base_log_odds
    final_denominator = 1 + model.active_predictor_count

    for feature in FEATURE_ORDER:
        selected_value = selected_values[feature]
        table = model.feature_tables[feature]
        row = table.loc[table["value"] == selected_value].iloc[0]
        contribution = float(row["contribution"])
        running_numerator += contribution
        score_after_feature = running_numerator / final_denominator
        contributions.append(
            {
                "Feature": FEATURE_LABELS[feature],
                "Selected value": selected_value,
                "Smoothed P(bin|accepted)": float(row["positive_probability"]),
                "Smoothed P(bin|rejected)": float(row["negative_probability"]),
                "Bin z-ratio": float(row["z_ratio"]),
                "Contribution": contribution,
                "Running numerator": running_numerator,
                "Score after feature": score_after_feature,
                "Positives in bin": int(row["positives"]),
                "Negatives in bin": int(row["negatives"]),
            }
        )

    contribution_frame = pd.DataFrame(contributions)
    score = running_numerator / final_denominator
    classifier_row = _match_classifier_row(model.classifier_table, score)

    return {
        "base_log_odds": model.base_log_odds,
        "score_numerator": running_numerator,
        "raw_score": score,
        "probability": float(classifier_row["Adjusted propensity"]),
        "classifier_bin": str(classifier_row["Bin"]),
        "classifier_propensity": float(classifier_row["Propensity"]),
        "classifier_adjusted_propensity": float(
            classifier_row["Adjusted propensity"]
        ),
        "classifier_z_ratio": float(classifier_row["Z-ratio"]),
        "classifier_positives": int(classifier_row["Positives"]),
        "classifier_negatives": int(classifier_row["Negatives"]),
        "contributions": contribution_frame,
    }
