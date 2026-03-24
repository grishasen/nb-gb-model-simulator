from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, inf, log

import pandas as pd

from model_demo.data import ProbeCustomer

FEATURES = ["age", "income", "existing_customer"]
FEATURE_LABELS = {
    "age": "Age",
    "income": "Income",
    "existing_customer": "Existing customer",
}


@dataclass
class TreeNode:
    prediction: float
    sample_count: int
    feature: str | None = None
    threshold: float | None = None
    gain: float = 0.0
    candidates: pd.DataFrame = field(default_factory=pd.DataFrame)
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None

    @property
    def is_leaf(self) -> bool:
        return self.feature is None


@dataclass
class BoostingRound:
    round_index: int
    tree: TreeNode
    learning_rate: float
    snapshot: pd.DataFrame


@dataclass
class GradientBoostingModel:
    base_rate: float
    base_score: float
    rounds: list[BoostingRound]


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + exp(-value))


def squared_error(values: pd.Series) -> float:
    if values.empty:
        return inf
    mean_value = float(values.mean())
    return float(((values - mean_value) ** 2).sum())


def candidate_thresholds(values: pd.Series) -> list[float]:
    unique_values = sorted(float(value) for value in values.dropna().unique())
    return [(left + right) / 2.0 for left, right in zip(unique_values, unique_values[1:])]


def evaluate_split_candidates(
    frame: pd.DataFrame,
    features: list[str],
    min_leaf_size: int,
) -> pd.DataFrame:
    parent_error = squared_error(frame["residual"])
    rows = []

    for feature in features:
        for threshold in candidate_thresholds(frame[feature]):
            left = frame.loc[frame[feature] <= threshold]
            right = frame.loc[frame[feature] > threshold]

            if len(left) < min_leaf_size or len(right) < min_leaf_size:
                continue

            loss = squared_error(left["residual"]) + squared_error(right["residual"])
            gain = parent_error - loss

            rows.append(
                {
                    "feature": feature,
                    "threshold": threshold,
                    "gain": gain,
                    "left_count": len(left),
                    "right_count": len(right),
                    "left_leaf_value": float(left["residual"].mean()),
                    "right_leaf_value": float(right["residual"].mean()),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "feature",
                "threshold",
                "gain",
                "left_count",
                "right_count",
                "left_leaf_value",
                "right_leaf_value",
            ]
        )

    return pd.DataFrame(rows).sort_values("gain", ascending=False).reset_index(drop=True)


def fit_tree(
    frame: pd.DataFrame,
    features: list[str],
    depth: int,
    min_leaf_size: int,
) -> TreeNode:
    node = TreeNode(
        prediction=float(frame["residual"].mean()),
        sample_count=len(frame),
    )

    if depth == 0 or len(frame) < min_leaf_size * 2:
        return node

    candidates = evaluate_split_candidates(frame, features, min_leaf_size)
    node.candidates = candidates

    if candidates.empty or float(candidates.iloc[0]["gain"]) <= 1e-9:
        return node

    best = candidates.iloc[0]
    feature = str(best["feature"])
    threshold = float(best["threshold"])

    left = frame.loc[frame[feature] <= threshold].copy()
    right = frame.loc[frame[feature] > threshold].copy()

    node.feature = feature
    node.threshold = threshold
    node.gain = float(best["gain"])
    node.left = fit_tree(left, features, depth - 1, min_leaf_size)
    node.right = fit_tree(right, features, depth - 1, min_leaf_size)
    return node


def predict_tree(node: TreeNode, row: pd.Series | dict[str, float]) -> float:
    if node.is_leaf:
        return node.prediction
    value = row[node.feature]  # type: ignore[index]
    if value <= node.threshold:
        return predict_tree(node.left, row)  # type: ignore[arg-type]
    return predict_tree(node.right, row)  # type: ignore[arg-type]


def trace_tree(node: TreeNode, customer: ProbeCustomer) -> tuple[list[str], float]:
    if node.is_leaf:
        return [f"Leaf value {node.prediction:+.3f}"], node.prediction

    customer_values = {
        "age": customer.age,
        "income": customer.income,
        "existing_customer": customer.existing_customer,
    }
    value = customer_values[node.feature]
    threshold = node.threshold

    if value <= threshold:
        step = f"{FEATURE_LABELS[node.feature]} <= {threshold:.1f} because {value} <= {threshold:.1f}"
        path, leaf_value = trace_tree(node.left, customer)  # type: ignore[arg-type]
        return [step, *path], leaf_value

    step = f"{FEATURE_LABELS[node.feature]} > {threshold:.1f} because {value} > {threshold:.1f}"
    path, leaf_value = trace_tree(node.right, customer)  # type: ignore[arg-type]
    return [step, *path], leaf_value


def format_tree_as_code(node: TreeNode, indent: int = 0) -> str:
    padding = "    " * indent
    if node.is_leaf:
        return f"{padding}return {node.prediction:+.3f}"

    assert node.feature is not None
    assert node.threshold is not None
    label = FEATURE_LABELS[node.feature].lower()
    left_code = format_tree_as_code(node.left, indent + 1)  # type: ignore[arg-type]
    right_code = format_tree_as_code(node.right, indent + 1)  # type: ignore[arg-type]
    return (
        f"{padding}if {label} <= {node.threshold:.1f}:\n"
        f"{left_code}\n"
        f"{padding}else:\n"
        f"{right_code}"
    )


def tree_to_graphviz(node: TreeNode, customer: ProbeCustomer | None = None) -> str:
    highlight_nodes: set[str] = set()
    highlight_edges: set[tuple[str, str]] = set()

    if customer is not None:
        current = node
        current_id = "n0"
        highlight_nodes.add(current_id)

        while not current.is_leaf:
            assert current.feature is not None
            assert current.threshold is not None
            customer_values = {
                "age": customer.age,
                "income": customer.income,
                "existing_customer": customer.existing_customer,
            }
            go_left = customer_values[current.feature] <= current.threshold
            next_id = f"{current_id}L" if go_left else f"{current_id}R"
            highlight_edges.add((current_id, next_id))
            highlight_nodes.add(next_id)
            current = current.left if go_left else current.right  # type: ignore[assignment]
            current_id = next_id

    lines = [
        "digraph Tree {",
        'graph [rankdir=TB, pad="0.2", nodesep="0.4", ranksep="0.5"];',
        'node [shape=box, style="rounded,filled", color="#94a3b8", fontname="Helvetica", fontsize=11, margin="0.18,0.12"];',
        'edge [color="#64748b", fontname="Helvetica", fontsize=10];',
    ]

    def walk(current: TreeNode, node_id: str) -> None:
        if current.is_leaf:
            fill = "#dcfce7" if node_id in highlight_nodes else "#f8fafc"
            label = (
                "Leaf\\n"
                f"value = {current.prediction:+.3f}\\n"
                f"samples = {current.sample_count}"
            )
            lines.append(
                f'{node_id} [label="{label}", fillcolor="{fill}", shape=ellipse];'
            )
            return

        assert current.feature is not None
        assert current.threshold is not None
        fill = "#dbeafe" if node_id in highlight_nodes else "#f8fafc"
        label = (
            f"{FEATURE_LABELS[current.feature]} <= {current.threshold:.1f}\\n"
            f"gain = {current.gain:.3f}\\n"
            f"samples = {current.sample_count}"
        )
        lines.append(f'{node_id} [label="{label}", fillcolor="{fill}"];')

        left_id = f"{node_id}L"
        right_id = f"{node_id}R"
        left_edge_color = "#16a34a" if (node_id, left_id) in highlight_edges else "#64748b"
        right_edge_color = "#16a34a" if (node_id, right_id) in highlight_edges else "#64748b"
        left_penwidth = "2.4" if (node_id, left_id) in highlight_edges else "1.2"
        right_penwidth = "2.4" if (node_id, right_id) in highlight_edges else "1.2"

        walk(current.left, left_id)  # type: ignore[arg-type]
        walk(current.right, right_id)  # type: ignore[arg-type]
        lines.append(
            f'{node_id} -> {left_id} [label="<=", color="{left_edge_color}", penwidth={left_penwidth}];'
        )
        lines.append(
            f'{node_id} -> {right_id} [label=">", color="{right_edge_color}", penwidth={right_penwidth}];'
        )

    walk(node, "n0")
    lines.append("}")
    return "\n".join(lines)


def train_gradient_boosting_demo(
    frame: pd.DataFrame,
    rounds: int = 2,
    learning_rate: float = 0.9,
    depth: int = 2,
    min_leaf_size: int = 4,
) -> GradientBoostingModel:
    base_rate = float(frame["accepted"].mean())
    base_score = log(base_rate / (1.0 - base_rate))
    working = frame.copy()
    working["raw_score"] = base_score

    boosting_rounds: list[BoostingRound] = []

    for round_index in range(1, rounds + 1):
        working["probability_before"] = working["raw_score"].map(sigmoid)
        working["residual"] = working["accepted"] - working["probability_before"]

        tree = fit_tree(
            working[
                [
                    "customer_id",
                    "age",
                    "income",
                    "existing_customer",
                    "accepted",
                    "residual",
                ]
            ].copy(),
            features=FEATURES,
            depth=depth,
            min_leaf_size=min_leaf_size,
        )

        working["tree_output"] = working.apply(lambda row: predict_tree(tree, row), axis=1)
        working["round_contribution"] = learning_rate * working["tree_output"]
        working["raw_score"] = working["raw_score"] + working["round_contribution"]

        snapshot = working[
            [
                "customer_id",
                "age",
                "income",
                "existing_status",
                "existing_customer",
                "accepted",
                "probability_before",
                "residual",
                "round_contribution",
                "raw_score",
            ]
        ].copy()

        boosting_rounds.append(
            BoostingRound(
                round_index=round_index,
                tree=tree,
                learning_rate=learning_rate,
                snapshot=snapshot,
            )
        )

    return GradientBoostingModel(
        base_rate=base_rate,
        base_score=base_score,
        rounds=boosting_rounds,
    )


def candidate_summary(node: TreeNode) -> pd.DataFrame:
    if node.candidates.empty:
        return node.candidates

    summary = node.candidates.copy()
    summary["Feature"] = summary["feature"].map(FEATURE_LABELS)
    summary["Threshold"] = summary["threshold"].round(1)
    summary["Gain"] = summary["gain"].round(3)
    summary["Left leaf"] = summary["left_leaf_value"].round(3)
    summary["Right leaf"] = summary["right_leaf_value"].round(3)
    return summary[
        ["Feature", "Threshold", "Gain", "left_count", "right_count", "Left leaf", "Right leaf"]
    ].rename(columns={"left_count": "Left count", "right_count": "Right count"})


def score_customer(
    model: GradientBoostingModel, customer: ProbeCustomer
) -> dict[str, object]:
    raw_score = model.base_score
    round_rows = []

    for round_info in model.rounds:
        path, leaf_value = trace_tree(round_info.tree, customer)
        contribution = round_info.learning_rate * leaf_value
        raw_score += contribution
        probability_after_round = sigmoid(raw_score)
        round_rows.append(
            {
                "Round": round_info.round_index,
                "Path": " -> ".join(path[:-1]) if len(path) > 1 else "Leaf only",
                "Leaf value": leaf_value,
                "Contribution": contribution,
                "Raw score after round": raw_score,
                "Propensity after round": probability_after_round,
                "Leaf note": path[-1],
            }
        )

    probability = sigmoid(raw_score)
    return {
        "base_score": model.base_score,
        "raw_score": raw_score,
        "probability": probability,
        "rounds": pd.DataFrame(round_rows),
    }
