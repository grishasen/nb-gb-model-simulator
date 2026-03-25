from __future__ import annotations

from math import log

import pandas as pd

from model_demo.data import build_probe_customer
from model_demo.gradient_boosting import (
    GradientBoostingModel,
    score_customer as score_gradient_boosting,
)
from model_demo.naive_bayes import (
    NaiveBayesModel,
    score_customer as score_naive_bayes,
)


def score_models_on_frame(
        nb_model: NaiveBayesModel,
        gb_model: GradientBoostingModel,
        frame: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []

    for row in frame.itertuples(index=False):
        customer = build_probe_customer(
            int(row.age),
            int(row.income),
            int(row.existing_customer),
        )
        nb_probability = float(score_naive_bayes(nb_model, customer)["probability"])
        gb_probability = float(score_gradient_boosting(gb_model, customer)["probability"])
        rows.append(
            {
                "customer_id": str(row.customer_id),
                "accepted": int(row.accepted),
                "nb_probability": nb_probability,
                "gb_probability": gb_probability,
            }
        )

    return pd.DataFrame(rows)


def binary_log_loss(actual: pd.Series, predicted: pd.Series) -> float:
    epsilon = 1e-12
    clipped = predicted.clip(epsilon, 1 - epsilon)
    return float(
        -(
            actual * clipped.map(log)
            + (1 - actual) * (1 - clipped).map(log)
        ).mean()
    )


def brier_score(actual: pd.Series, predicted: pd.Series) -> float:
    return float(((predicted - actual) ** 2).mean())


def accuracy(actual: pd.Series, predicted: pd.Series) -> float:
    labels = (predicted >= 0.5).astype(int)
    return float((labels == actual).mean())


def roc_auc(actual: pd.Series, predicted: pd.Series) -> float:
    positives = int(actual.sum())
    negatives = int(len(actual) - positives)
    if positives == 0 or negatives == 0:
        return 0.5

    ranked = pd.DataFrame({"actual": actual, "predicted": predicted})
    ranked["rank"] = ranked["predicted"].rank(method="average")
    positive_rank_sum = float(ranked.loc[ranked["actual"] == 1, "rank"].sum())
    return (
        positive_rank_sum - (positives * (positives + 1) / 2)
    ) / (positives * negatives)


def evaluate_predictions(
        actual: pd.Series,
        predicted: pd.Series,
) -> dict[str, float]:
    return {
        "Log loss": binary_log_loss(actual, predicted),
        "Brier score": brier_score(actual, predicted),
        "AUC": roc_auc(actual, predicted),
        "Accuracy": accuracy(actual, predicted),
    }


def build_metric_table(
        nb_model: NaiveBayesModel,
        gb_model: GradientBoostingModel,
        training_frame: pd.DataFrame,
        holdout_frame: pd.DataFrame,
) -> pd.DataFrame:
    train_predictions = score_models_on_frame(nb_model, gb_model, training_frame)
    holdout_predictions = score_models_on_frame(nb_model, gb_model, holdout_frame)

    rows: list[dict[str, float | str]] = []
    for model_label, probability_column in (
            ("Naive Bayes", "nb_probability"),
            ("Gradient boosting", "gb_probability"),
    ):
        train_metrics = evaluate_predictions(
            train_predictions["accepted"],
            train_predictions[probability_column],
        )
        holdout_metrics = evaluate_predictions(
            holdout_predictions["accepted"],
            holdout_predictions[probability_column],
        )
        rows.append(
            {
                "Model": model_label,
                "Train log loss": train_metrics["Log loss"],
                "Holdout log loss": holdout_metrics["Log loss"],
                "Train Brier": train_metrics["Brier score"],
                "Holdout Brier": holdout_metrics["Brier score"],
                "Train AUC": train_metrics["AUC"],
                "Holdout AUC": holdout_metrics["AUC"],
                "Train accuracy": train_metrics["Accuracy"],
                "Holdout accuracy": holdout_metrics["Accuracy"],
            }
        )

    return pd.DataFrame(rows)
