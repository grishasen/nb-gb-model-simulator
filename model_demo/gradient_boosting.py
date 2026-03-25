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
    score: float = 0.0
    feature: str | None = None
    split_kind: str | None = None
    threshold: float | None = None
    match_values: tuple[float, ...] | None = None
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


def format_feature_value(feature: str, value: float) -> str:
    if feature == "existing_customer":
        return "Existing" if int(value) == 1 else "New"
    if float(value).is_integer():
        return str(int(value))
    return f"{float(value):.1f}"


def split_rule_text(
        feature: str,
        split_kind: str,
        threshold: float | None = None,
        match_values: tuple[float, ...] | None = None,
) -> str:
    if split_kind == "symbolic":
        assert match_values is not None
        values = ", ".join(format_feature_value(feature, value) for value in match_values)
        return f"in {{{values}}}"
    assert threshold is not None
    return f"<= {threshold:.1f}"


def split_mask(
        frame: pd.DataFrame,
        feature: str,
        split_kind: str,
        threshold: float | None = None,
        match_values: tuple[float, ...] | None = None,
) -> pd.Series:
    if split_kind == "symbolic":
        assert match_values is not None
        return frame[feature].isin(match_values)
    assert threshold is not None
    return frame[feature] <= threshold


def goes_left(
        row: pd.Series | dict[str, float],
        feature: str,
        split_kind: str,
        threshold: float | None = None,
        match_values: tuple[float, ...] | None = None,
) -> bool:
    value = row[feature]  # type: ignore[index]
    if split_kind == "symbolic":
        assert match_values is not None
        return float(value) in match_values
    assert threshold is not None
    return float(value) <= threshold


def evaluate_split_candidates(
        frame: pd.DataFrame,
        features: list[str],
        min_leaf_size: int,
) -> pd.DataFrame:
    parent_error = squared_error(frame["residual"])
    rows = []

    for feature in features:
        if feature == "existing_customer":
            match_values = (1.0,)
            left_mask = split_mask(
                frame,
                feature,
                split_kind="symbolic",
                match_values=match_values,
            )
            left = frame.loc[left_mask]
            right = frame.loc[~left_mask]

            if len(left) >= min_leaf_size and len(right) >= min_leaf_size:
                loss = squared_error(left["residual"]) + squared_error(right["residual"])
                gain = parent_error - loss
                rows.append(
                    {
                        "feature": feature,
                        "split_kind": "symbolic",
                        "threshold": None,
                        "match_values": match_values,
                        "split_text": split_rule_text(
                            feature,
                            "symbolic",
                            match_values=match_values,
                        ),
                        "gain": gain,
                        "left_count": len(left),
                        "right_count": len(right),
                        "left_leaf_value": float(left["residual"].mean()),
                        "right_leaf_value": float(right["residual"].mean()),
                    }
                )
            continue

        for threshold in candidate_thresholds(frame[feature]):
            left_mask = split_mask(
                frame,
                feature,
                split_kind="numeric",
                threshold=threshold,
            )
            left = frame.loc[left_mask]
            right = frame.loc[~left_mask]

            if len(left) < min_leaf_size or len(right) < min_leaf_size:
                continue

            loss = squared_error(left["residual"]) + squared_error(right["residual"])
            gain = parent_error - loss

            rows.append(
                {
                    "feature": feature,
                    "split_kind": "numeric",
                    "threshold": threshold,
                    "match_values": None,
                    "split_text": split_rule_text(
                        feature,
                        "numeric",
                        threshold=threshold,
                    ),
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
                "split_kind",
                "threshold",
                "match_values",
                "split_text",
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
    split_kind = str(best["split_kind"])
    threshold = (
        None if pd.isna(best["threshold"]) else float(best["threshold"])
    )
    match_values = best["match_values"]
    if isinstance(match_values, tuple):
        selected_values = tuple(float(value) for value in match_values)
    elif pd.isna(match_values):
        selected_values = None
    else:
        selected_values = tuple(float(value) for value in match_values)

    left_mask = split_mask(
        frame,
        feature,
        split_kind=split_kind,
        threshold=threshold,
        match_values=selected_values,
    )
    left = frame.loc[left_mask].copy()
    right = frame.loc[~left_mask].copy()

    node.feature = feature
    node.split_kind = split_kind
    node.threshold = threshold
    node.match_values = selected_values
    node.gain = float(best["gain"])
    node.left = fit_tree(left, features, depth - 1, min_leaf_size)
    node.right = fit_tree(right, features, depth - 1, min_leaf_size)
    return node


def apply_learning_rate(node: TreeNode, learning_rate: float) -> None:
    node.score = learning_rate * node.prediction
    if node.left is not None:
        apply_learning_rate(node.left, learning_rate)
    if node.right is not None:
        apply_learning_rate(node.right, learning_rate)


def predict_tree(node: TreeNode, row: pd.Series | dict[str, float]) -> float:
    if node.is_leaf:
        return node.score
    assert node.feature is not None
    assert node.split_kind is not None
    if goes_left(
            row,
            node.feature,
            split_kind=node.split_kind,
            threshold=node.threshold,
            match_values=node.match_values,
    ):
        return predict_tree(node.left, row)  # type: ignore[arg-type]
    return predict_tree(node.right, row)  # type: ignore[arg-type]


def trace_tree(node: TreeNode, customer: ProbeCustomer) -> tuple[list[str], float]:
    if node.is_leaf:
        return [f"Tree score {node.score:+.3f}"], node.score

    customer_values = {
        "age": customer.age,
        "income": customer.income,
        "existing_customer": customer.existing_customer,
    }
    assert node.feature is not None
    assert node.split_kind is not None
    value = customer_values[node.feature]

    if goes_left(
            customer_values,
            node.feature,
            split_kind=node.split_kind,
            threshold=node.threshold,
            match_values=node.match_values,
    ):
        if node.split_kind == "symbolic":
            match_text = split_rule_text(
                node.feature,
                node.split_kind,
                match_values=node.match_values,
            )
            step = (
                f"{FEATURE_LABELS[node.feature]} {match_text} because "
                f"{format_feature_value(node.feature, value)} matched"
            )
        else:
            assert node.threshold is not None
            step = (
                f"{FEATURE_LABELS[node.feature]} <= {node.threshold:.1f} because "
                f"{value} <= {node.threshold:.1f}"
            )
        path, leaf_value = trace_tree(node.left, customer)  # type: ignore[arg-type]
        return [step, *path], leaf_value

    if node.split_kind == "symbolic":
        match_text = split_rule_text(
            node.feature,
            node.split_kind,
            match_values=node.match_values,
        )
        step = (
            f"{FEATURE_LABELS[node.feature]} not {match_text} because "
            f"{format_feature_value(node.feature, value)} did not match"
        )
    else:
        assert node.threshold is not None
        step = (
            f"{FEATURE_LABELS[node.feature]} > {node.threshold:.1f} because "
            f"{value} > {node.threshold:.1f}"
        )
    path, leaf_value = trace_tree(node.right, customer)  # type: ignore[arg-type]
    return [step, *path], leaf_value


def format_tree_as_code(node: TreeNode, indent: int = 0) -> str:
    padding = "    " * indent
    if node.is_leaf:
        return f"{padding}return {node.score:+.3f}"

    assert node.feature is not None
    label = FEATURE_LABELS[node.feature].lower()
    assert node.split_kind is not None
    if node.split_kind == "symbolic":
        assert node.match_values is not None
        values = ", ".join(repr(format_feature_value(node.feature, value)) for value in node.match_values)
        condition = f"{label} in {{{values}}}"
    else:
        assert node.threshold is not None
        condition = f"{label} <= {node.threshold:.1f}"
    left_code = format_tree_as_code(node.left, indent + 1)  # type: ignore[arg-type]
    right_code = format_tree_as_code(node.right, indent + 1)  # type: ignore[arg-type]
    return (
        f"{padding}if {condition}:\n"
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
            customer_values = {
                "age": customer.age,
                "income": customer.income,
                "existing_customer": customer.existing_customer,
            }
            assert current.split_kind is not None
            go_left = goes_left(
                customer_values,
                current.feature,
                split_kind=current.split_kind,
                threshold=current.threshold,
                match_values=current.match_values,
            )
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
                f"score = {current.score:+.3f}\\n"
                f"samples = {current.sample_count}"
            )
            lines.append(
                f'{node_id} [label="{label}", fillcolor="{fill}", shape=ellipse];'
            )
            return

        assert current.feature is not None
        assert current.split_kind is not None
        fill = "#dbeafe" if node_id in highlight_nodes else "#f8fafc"
        label = (
            f"{FEATURE_LABELS[current.feature]} {split_rule_text(current.feature, current.split_kind, threshold=current.threshold, match_values=current.match_values)}\\n"
            f"score = {current.score:+.3f}\\n"
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
        apply_learning_rate(tree, learning_rate)

        working["tree_score"] = working.apply(lambda row: predict_tree(tree, row), axis=1)
        working["raw_score"] = working["raw_score"] + working["tree_score"]

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
                "tree_score",
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


def candidate_summary(node: TreeNode, scale: float = 1.0) -> pd.DataFrame:
    if node.candidates.empty:
        return node.candidates

    summary = node.candidates.copy()
    summary["Feature"] = summary["feature"].map(FEATURE_LABELS)
    summary["Split"] = summary["split_text"]
    summary["Gain"] = summary["gain"].round(3)
    summary["Left score"] = (scale * summary["left_leaf_value"]).round(3)
    summary["Right score"] = (scale * summary["right_leaf_value"]).round(3)
    return summary[
        ["Feature", "Split", "Gain", "left_count", "right_count", "Left score", "Right score"]
    ].rename(columns={"left_count": "Left count", "right_count": "Right count"})


def score_customer(
        model: GradientBoostingModel, customer: ProbeCustomer
) -> dict[str, object]:
    raw_score = model.base_score
    round_rows = []
    running_tree_scores: list[float] = []

    for round_info in model.rounds:
        path, tree_score = trace_tree(round_info.tree, customer)
        raw_score += tree_score
        running_tree_scores.append(tree_score)
        probability_after_round = sigmoid(raw_score)
        round_rows.append(
            {
                "Tree": f"Tree {round_info.round_index}",
                "Path": " -> ".join(path[:-1]) if len(path) > 1 else "Leaf only",
                "Tree score": tree_score,
                "Running mean tree score": sum(running_tree_scores) / len(running_tree_scores),
                "Raw score after tree": raw_score,
                "Propensity after tree": probability_after_round,
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
