import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from model_demo.data import (
    SCENARIO_DESCRIPTIONS,
    SCENARIO_LABELS,
    build_probe_customer,
    build_rule_regions,
    build_training_data,
    pretty_training_table,
)
from model_demo.gradient_boosting import (
    candidate_summary,
    format_tree_as_code,
    score_customer as score_gradient_boosting,
    tree_to_graphviz,
    train_gradient_boosting_demo,
)
from model_demo.naive_bayes import (
    feature_summary_table,
    score_customer as score_naive_bayes,
    train_naive_bayes,
)
from model_demo.plotly_utils import init_plotly_theme

st.set_page_config(
    page_title="Naive Bayes vs Gradient Boosting",
    layout="wide",
    initial_sidebar_state="expanded",
)
init_plotly_theme()


def show_table(
        data: pd.DataFrame | list[dict[str, object]] | dict[str, list[object]],
        *,
        column_config: dict[str, st.column_config.Column] | None = None,
        height: int | str = 'content',
) -> None:
    st.data_editor(
        data,
        hide_index=True,
        width='stretch',
        disabled=True,
        num_rows="fixed",
        column_config=column_config,
        height=height,
    )


def create_nb_contribution_figure(
        contribution_frame: pd.DataFrame,
        limit: float,
) -> go.Figure:
    chart_frame = contribution_frame.copy()
    chart_frame["Color"] = chart_frame["Contribution"].apply(
        lambda value: "#0e6b62" if value >= 0 else "#b86839"
    )

    fig = go.Figure(
        data=[
            go.Bar(
                x=chart_frame["Contribution"],
                y=chart_frame["Feature"],
                orientation="h",
                marker_color=chart_frame["Color"],
                customdata=chart_frame[["Selected value", "Bin z-ratio"]],
                hovertemplate=(
                    "Feature=%{y}<br>"
                    "Selected value=%{customdata[0]}<br>"
                    "Bin z-ratio=%{customdata[1]:.3f}<br>"
                    "Contribution=%{x:.3f}<extra></extra>"
                ),
            )
        ]
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(
        title_text="Contribution",
        range=[-limit, limit],
        zeroline=True,
        zerolinecolor="#64748b",
        showgrid=True,
        gridcolor="#e2e8f0",
    )
    fig.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
    )
    return fig


def create_gb_round_figure(
        base_score: float,
        round_frame: pd.DataFrame,
) -> go.Figure:
    chart_frame = round_frame.copy()
    chart_frame["Color"] = chart_frame["Tree score"].apply(
        lambda value: "#0e6b62" if value >= 0 else "#b86839"
    )
    customdata = chart_frame[["Leaf note", "Path", "Propensity after tree"]]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=chart_frame["Tree"],
            y=chart_frame["Tree score"],
            marker_color=chart_frame["Color"],
            name="Tree score",
            customdata=customdata,
            hovertemplate=(
                "%{x}<br>"
                "Tree score=%{y:.3f}<br>"
                "Scoring node=%{customdata[0]}<br>"
                "Path=%{customdata[1]}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=chart_frame["Tree"],
            y=chart_frame["Running mean tree score"],
            mode="lines+markers",
            name="Running mean tree score",
            line=dict(color="#1d4ed8", width=3),
            marker=dict(size=8),
            hovertemplate="%{x}<br>Running mean=%{y:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=chart_frame["Tree"],
            y=chart_frame["Propensity after tree"],
            mode="lines+markers",
            name="Running propensity",
            line=dict(color="#7c3aed", width=3, dash="dot"),
            marker=dict(size=8),
            yaxis="y2",
            hovertemplate="%{x}<br>Running propensity=%{y:.1%}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_width=1, line_color="#94a3b8")
    fig.update_layout(
        height=480,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        yaxis=dict(title="Tree score"),
        yaxis2=dict(
            title="Propensity",
            overlaying="y",
            side="right",
            tickformat=".0%",
            range=[0, 1],
        ),
        title=dict(
            text=f"Bias score before learned trees: {base_score:+.3f}",
            x=0.06,
            y=0.92,
            xanchor="left",
            font=dict(size=12),
        ),
    )
    return fig


st.sidebar.header("Scenario")
scenario = st.sidebar.selectbox(
    "Dataset scenario",
    options=list(SCENARIO_LABELS.keys()),
    format_func=lambda value: SCENARIO_LABELS[value],
)
st.sidebar.caption(SCENARIO_DESCRIPTIONS[scenario])

st.sidebar.header("Probe Customer")
st.sidebar.caption("Adjust one customer and compare how both models score the same case.")
probe_age = st.sidebar.slider("Age", min_value=22, max_value=55, value=30, step=1)
probe_income = st.sidebar.slider(
    "Income", min_value=6000, max_value=16000, value=12000, step=250
)
probe_existing = st.sidebar.checkbox("Existing customer", value=False)

training_data = build_training_data(scenario)
nb_model = train_naive_bayes(training_data)
gb_model = train_gradient_boosting_demo(training_data)
probe = build_probe_customer(probe_age, probe_income, int(probe_existing))
nb_contribution_limit = max(
    abs(value)
    for table in nb_model.feature_tables.values()
    for value in table["contribution"].tolist()
)
nb_contribution_limit = max(0.25, round(nb_contribution_limit * 1.15, 3))

nb_result = score_naive_bayes(nb_model, probe)
gb_result = score_gradient_boosting(gb_model, probe)
rule_regions = build_rule_regions(scenario)

st.title("Naive Bayes vs Gradient Boosting")
st.markdown(
    """
    - `Naive Bayes` does not create combined rules. It scores each feature independently and adds the evidence.
    - `Gradient boosting` can create paths such as `age < 35` then `income > 10000`.
    - The same change in income can have a fixed effect in NB but a branch-specific effect in GB.
    """
)
st.success(
    SCENARIO_DESCRIPTIONS[scenario]
)

tab_story, tab_nb, tab_gb, tab_compare, tab_wiki = st.tabs(
    [":material/database_search: Story", ":material/star_shine: Naive Bayes", ":material/light_mode: Gradient Boosting",
     ":material/wand_stars: Compare", ":material/description: Wiki"]
)

with tab_story:
    st.header("Learning Setup")
    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        chart_data = training_data.assign(outcome=training_data["accepted_label"])
        chart_spec = {
            "layer": [
                {
                    "data": {"values": rule_regions.to_dict("records")},
                    "mark": {"type": "rect", "opacity": 0.18, "color": "#0e6b62"},
                    "encoding": {
                        "column": {
                            "field": "existing_status",
                            "type": "nominal",
                            "title": "Existing customer",
                            "sort": ["New customer", "Existing customer"],
                        },
                        "x": {"field": "x1", "type": "quantitative", "title": "Age"},
                        "x2": {"field": "x2"},
                        "y": {
                            "field": "y1",
                            "type": "quantitative",
                            "title": "Income",
                        },
                        "y2": {"field": "y2"},
                    },
                },
                {
                    "data": {"values": chart_data.to_dict("records")},
                    "mark": {"type": "circle", "size": 220, "opacity": 0.9},
                    "encoding": {
                        "column": {
                            "field": "existing_status",
                            "type": "nominal",
                            "title": "Existing customer",
                            "sort": ["New customer", "Existing customer"],
                        },
                        "x": {"field": "age", "type": "quantitative", "title": "Age"},
                        "y": {
                            "field": "income",
                            "type": "quantitative",
                            "title": "Income",
                        },
                        "color": {
                            "field": "outcome",
                            "type": "nominal",
                            "scale": {
                                "domain": ["Accepted", "Rejected"],
                                "range": ["#0e6b62", "#b86839"],
                            },
                        },
                        "tooltip": [
                            {"field": "customer_id", "type": "nominal", "title": "Customer"},
                            {"field": "age", "type": "quantitative", "title": "Age"},
                            {"field": "income", "type": "quantitative", "title": "Income"},
                            {
                                "field": "existing_status",
                                "type": "nominal",
                                "title": "Existing customer",
                            },
                            {"field": "outcome", "type": "nominal", "title": "Outcome"},
                        ],
                    },
                },
            ],
            "height": 420,
        }
        st.vega_lite_chart(chart_data, chart_spec, width='stretch')
        st.caption("The shaded block is the hidden rule that generated positive outcomes.")

    with right:
        st.subheader("Current hidden rule")
        if scenario == "two_clusters":
            st.markdown(
                """
                - Positive if `age < 35 and income >= 11000`
                - Or positive if `existing customer = yes and income <= 9000`
                """
            )
        else:
            st.markdown("- Positive if `age < 35 and income >= 11000`")
        st.subheader("Probe Customer")
        show_table(
            pd.DataFrame(
                {
                    "Age": [probe.age],
                    "Income": [probe.income],
                    "Existing customer": [probe.existing_status],
                    "Age band": [probe.age_bin],
                    "Income band": [probe.income_bin],
                }
            ),
            column_config={
                "Age": st.column_config.NumberColumn("Age", format="%d"),
                "Income": st.column_config.NumberColumn("Income", format="$%d"),
            },
        )

    st.subheader("Training Data")
    with st.expander("Example dataset", expanded=True):
        show_table(
            pretty_training_table(training_data),
            column_config={
                "Age": st.column_config.NumberColumn("Age", format="%d"),
                "Income": st.column_config.NumberColumn("Income", format="$%d"),
            }
        )

with tab_nb:
    st.subheader("Naive Bayes: Independent evidence from each feature")
    st.success(
        "Example follows the Pega ADM-style: each feature adds a modified log-odds contribution, the model averages those contributions into a final score, and a classifier table maps that score to the returned propensity.",
    )

    metric_columns = st.columns(3, gap="medium")
    with metric_columns[0]:
        st.metric("Base log-odds term", f"{nb_result['base_log_odds']:+.3f}")
    with metric_columns[1]:
        st.metric("Final ADM score", f"{nb_result['raw_score']:+.3f}")
    with metric_columns[2]:
        st.metric("Returned propensity", f"{nb_result['probability']:.1%}")

    nb_left, nb_right = st.columns(2, gap="large")
    with nb_left:
        st.subheader("How the base term is calculated")
        st.write(
            "Pega's NB score starts from the overall accepted-versus-rejected balance. The total denominator cancels out, so the base term can be written directly from counts."
        )
        st.latex(
            rf"\text{{base term}} = \log(1 + {nb_model.positives}) - \log(1 + {nb_model.negatives}) = {nb_result['base_log_odds']:+.3f}"
        )
        st.latex(
            r"\text{contribution}_p = \log\left(Pos_i + \frac{1}{nBins}\right) - \log\left(Neg_i + \frac{1}{nBins}\right) - \log(1 + TotalPos) + \log(1 + TotalNeg)"
        )
        st.caption(
            "Each bin also gets a z-ratio, which measures how different its accepted-share and rejected-share are relative to sampling noise."
        )

    with nb_right:
        st.subheader("How this probe customer becomes a propensity")
        st.latex(
            r"score = \frac{\log(1 + TotalPositives) – \log(1 + TotalNegatives) + \sum_p contribution_p}{1 + nActivePredictors}"
        )
        nb_contribution_terms = " + ".join(
            [f"({nb_result['base_log_odds']:+.3f})"]
            + [
                f"({value:+.3f})"
                for value in nb_result["contributions"]["Contribution"].tolist()
            ]
        )
        st.latex(
            rf"\text{{score numerator}} = {nb_contribution_terms} = {nb_result['score_numerator']:+.3f}"
        )
        st.latex(
            rf"\text{{score}} = \frac{{{nb_result['score_numerator']:+.3f}}}{{1 + {nb_model.active_predictor_count}}} = {nb_result['raw_score']:+.3f}"
        )
        st.latex(
            rf"\text{{returned propensity}} = \frac{{0.5 + {nb_result['classifier_positives']}}}{{1 + {nb_result['classifier_positives']} + {nb_result['classifier_negatives']}}} = {nb_result['probability']:.3f}"
        )
        st.caption(
            f"Probe score falls into classifier bin `{nb_result['classifier_bin']}` with classifier z-ratio {nb_result['classifier_z_ratio']:+.3f}."
        )

    st.subheader("Contribution chart")
    nb_contribution_figure = create_nb_contribution_figure(
        nb_result["contributions"],
        nb_contribution_limit,
    )
    st.plotly_chart(
        nb_contribution_figure,
        width='stretch',
        key=(
            f"nb_contribution_chart_{scenario}_{probe.age_bin}_"
            f"{probe.income_bin}_{probe.existing_status}"
        ),
    )
    st.caption(
        f"Axis fixed for this scenario: {-nb_contribution_limit:.3f} to {nb_contribution_limit:.3f}. "
        "This chart shows only the currently selected modified log-odds contributions."
    )

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Score breakdown for the probe customer")
        show_table(
            nb_result["contributions"].assign(
                **{
                    "Contribution": lambda frame: frame["Contribution"].round(3),
                    "Smoothed P(bin|accepted)": lambda frame: frame[
                        "Smoothed P(bin|accepted)"
                    ].round(3),
                    "Smoothed P(bin|rejected)": lambda frame: frame[
                        "Smoothed P(bin|rejected)"
                    ].round(3),
                    "Bin z-ratio": lambda frame: frame["Bin z-ratio"].round(3),
                    "Running numerator": lambda frame: frame[
                        "Running numerator"
                    ].round(3),
                    "Score after feature": lambda frame: frame[
                        "Score after feature"
                    ].round(3),
                }
            ),
            column_config={
                "Smoothed P(bin|accepted)": st.column_config.NumberColumn(
                    "Smoothed P(bin|accepted)", format="%.3f"
                ),
                "Smoothed P(bin|rejected)": st.column_config.NumberColumn(
                    "Smoothed P(bin|rejected)", format="%.3f"
                ),
                "Bin z-ratio": st.column_config.NumberColumn(
                    "Bin z-ratio", format="%.3f"
                ),
                "Contribution": st.column_config.NumberColumn(
                    "Contribution", format="%.3f"
                ),
                "Running numerator": st.column_config.NumberColumn(
                    "Running numerator", format="%.3f"
                ),
                "Score after feature": st.column_config.NumberColumn(
                    "Score after feature", format="%.3f"
                ),
                "Positives in bin": st.column_config.NumberColumn(
                    "Positives in bin", format="%d"
                ),
                "Negatives in bin": st.column_config.NumberColumn(
                    "Negatives in bin", format="%d"
                ),
            },
        )

    with right:
        st.subheader("What the model learned from bins")
        st.latex(
            r"\text{bin z-ratio} = \frac{p_{acc} - p_{rej}}{\sqrt{\frac{p_{acc}(1-p_{acc})}{TotalAccepted} + \frac{p_{rej}(1-p_{rej})}{TotalRejected}}}"
        )
        age_table, income_table, existing_table, classifier_table = st.tabs(
            [
                "Age band table",
                "Income band table",
                "Existing customer table",
                "Classifier table",
            ]
        )
        feature_table_config = {
            "Responses": st.column_config.NumberColumn("Responses", format="%d"),
            "Responses %": st.column_config.NumberColumn("Responses %", format="%.3f"),
            "Positives": st.column_config.NumberColumn("Positives", format="%d"),
            "Negatives": st.column_config.NumberColumn("Negatives", format="%d"),
            "Accepted % of total": st.column_config.NumberColumn(
                "Accepted % of total", format="%.3f"
            ),
            "Rejected % of total": st.column_config.NumberColumn(
                "Rejected % of total", format="%.3f"
            ),
            "Accepted % of accepted": st.column_config.NumberColumn(
                "Accepted % of accepted", format="%.3f"
            ),
            "Rejected % of rejected": st.column_config.NumberColumn(
                "Rejected % of rejected", format="%.3f"
            ),
            "Bin propensity": st.column_config.NumberColumn(
                "Bin propensity", format="%.3f"
            ),
            "Adjusted propensity": st.column_config.NumberColumn(
                "Adjusted propensity", format="%.3f"
            ),
            "Z-ratio": st.column_config.NumberColumn("Z-ratio", format="%.3f"),
            "Contribution": st.column_config.NumberColumn(
                "Contribution", format="%.3f"
            ),
        }
        with age_table:
            show_table(
                feature_summary_table(nb_model, "age_bin"),
                column_config=feature_table_config,
            )
        with income_table:
            show_table(
                feature_summary_table(nb_model, "income_bin"),
                column_config=feature_table_config,
            )
        with existing_table:
            show_table(
                feature_summary_table(nb_model, "existing_status"),
                column_config=feature_table_config,
            )
        with classifier_table:
            classifier_display = nb_model.classifier_table.copy()
            classifier_display["Selected"] = classifier_display["Bin"].eq(
                nb_result["classifier_bin"]
            )
            show_table(
                classifier_display[
                    [
                        "Selected",
                        "Bin",
                        "Positives",
                        "Negatives",
                        "Responses %",
                        "Cum. total %",
                        "Propensity",
                        "Adjusted propensity",
                        "Cum. positives %",
                        "Z-ratio",
                        "Lift",
                    ]
                ].assign(
                    **{
                        "Responses %": lambda frame: frame["Responses %"].round(3),
                        "Cum. total %": lambda frame: frame["Cum. total %"].round(3),
                        "Propensity": lambda frame: frame["Propensity"].round(3),
                        "Adjusted propensity": lambda frame: frame[
                            "Adjusted propensity"
                        ].round(3),
                        "Cum. positives %": lambda frame: frame[
                            "Cum. positives %"
                        ].round(3),
                        "Z-ratio": lambda frame: frame["Z-ratio"].round(3),
                        "Lift": lambda frame: frame["Lift"].round(3),
                    }
                ),
                column_config={
                    "Selected": st.column_config.CheckboxColumn("Selected"),
                    "Positives": st.column_config.NumberColumn("Positives", format="%d"),
                    "Negatives": st.column_config.NumberColumn("Negatives", format="%d"),
                    "Responses %": st.column_config.NumberColumn(
                        "Responses %", format="%.3f"
                    ),
                    "Cum. total %": st.column_config.NumberColumn(
                        "Cum. total %", format="%.3f"
                    ),
                    "Propensity": st.column_config.NumberColumn(
                        "Propensity", format="%.3f"
                    ),
                    "Adjusted propensity": st.column_config.NumberColumn(
                        "Adjusted propensity", format="%.3f"
                    ),
                    "Cum. positives %": st.column_config.NumberColumn(
                        "Cum. positives %", format="%.3f"
                    ),
                    "Z-ratio": st.column_config.NumberColumn(
                        "Z-ratio", format="%.3f"
                    ),
                    "Lift": st.column_config.NumberColumn("Lift", format="%.3f"),
                },
            )
            st.caption(
                "This simplified classifier table is built with a pool-adjacent-violators monotonic mapping so the final score is translated to propensity in the same spirit as the Pega ADM classifier."
            )

    if scenario == "two_clusters":
        st.warning(
            "Naive Bayes can add evidence from age, income, and existing-customer status, but it still does not create a path like "
            "`if existing customer and low income then positive`. It sums separate feature evidence instead."
        )
    else:
        st.warning(
            "If income moves from 9,000 to 14,000, Naive Bayes applies the same income evidence no matter whether the customer is 28 or 45. That is the independence assumption in action.",
        )

with tab_gb:
    st.subheader("Gradient Boosting: Conditional rules built from split paths")
    st.success(
        "This simplified view now uses a more Pega-like scoring story: start from a bias score, take one scoring node from each tree, sum those tree scores, and then convert the final score to propensity.",
    )
    if scenario == "two_clusters":
        st.info(
            "This scenario was designed to leave behind a second strong residual pattern. Tree 1 usually explains the young-higher-income region, and tree 2 often pivots to `Existing customer` to pick up the lower-income existing-customer region."
        )
    else:
        st.info(
            "The trees look similar because the toy dataset has one very strong, clean rule: younger customers with higher income are positive. "
            "After the first tree, the remaining residuals are smaller, but they still concentrate in the same region, so the second tree often picks the same split pattern with smaller leaf values."
        )

    metric_columns = st.columns(3, gap="medium")
    with metric_columns[0]:
        st.metric("Bias score", f"{gb_result['base_score']:+.3f}")
    with metric_columns[1]:
        st.metric("Final raw score", f"{gb_result['raw_score']:+.3f}")
    with metric_columns[2]:
        st.metric("Final probability", f"{gb_result['probability']:.1%}")

    base_left, base_right = st.columns(2, gap="large")
    positive_count = int(training_data["accepted"].sum())
    total_count = len(training_data)
    with base_left:
        st.subheader("How the bias score is calculated")
        st.write(
            "Before any learned trees fire, the model starts from the overall population balance."
        )
        st.latex(
            rf"\text{{accepted rate}} = \frac{{{positive_count}}}{{{total_count}}} = {gb_model.base_rate:.3f}"
        )
        st.latex(
            rf"\text{{bias score}} = \log\left(\frac{{{gb_model.base_rate:.3f}}}{{1-{gb_model.base_rate:.3f}}}\right) = {gb_model.base_score:+.3f}"
        )
    with base_right:
        st.subheader("How this probe customer becomes a propensity")
        contribution_terms = " + ".join(
            [f"({gb_model.base_score:+.3f})"]
            + [
                f"({value:+.3f})"
                for value in gb_result["rounds"]["Tree score"].tolist()
            ]
        )
        st.latex(
            rf"\text{{final score}} = {contribution_terms} = {gb_result['raw_score']:+.3f}"
        )
        st.latex(
            rf"\text{{propensity}} = \sigma(\text{{final score}}) = \frac{{1}}{{1 + e^{{-({gb_result['raw_score']:+.3f})}}}} = {gb_result['probability']:.3f}"
        )

    st.subheader("Pega-style per-tree contribution view")
    gb_round_figure = create_gb_round_figure(
        gb_result["base_score"],
        gb_result["rounds"],
    )
    st.plotly_chart(
        gb_round_figure,
        width='stretch',
        key=f"gb_round_chart_{scenario}_{probe.age}_{probe.income}_{probe.existing_customer}"
    )
    st.caption(
        "Bars show the score contributed by the visited node in each tree. The blue line tracks the running mean tree score, and the dotted line shows the running propensity after adding each tree to the bias score."
    )

    st.subheader("Tree-by-tree scoring trace")
    show_table(
        gb_result["rounds"].assign(
            **{
                "Tree score": lambda frame: frame["Tree score"].round(3),
                "Running mean tree score": lambda frame: frame[
                    "Running mean tree score"
                ].round(3),
                "Raw score after tree": lambda frame: frame[
                    "Raw score after tree"
                ].round(3),
                "Propensity after tree": lambda frame: frame[
                    "Propensity after tree"
                ].round(3),
            }
        ),
        column_config={
            "Tree score": st.column_config.NumberColumn(
                "Tree score", format="%.3f"
            ),
            "Running mean tree score": st.column_config.NumberColumn(
                "Running mean tree score", format="%.3f"
            ),
            "Raw score after tree": st.column_config.NumberColumn(
                "Raw score after tree", format="%.3f"
            ),
            "Propensity after tree": st.column_config.NumberColumn(
                "Propensity after tree", format="%.3f"
            ),
        },
    )

    for round_info in gb_model.rounds:
        with st.expander(
                f"Boosting round {round_info.round_index}",
                expanded=round_info.round_index == 1,
        ):
            left, right = st.columns([1.05, 0.95], gap="large")
            with left:
                st.markdown("**Top split candidates at the root**")
                show_table(
                    candidate_summary(
                        round_info.tree,
                        scale=round_info.learning_rate,
                    ).head(6),
                    column_config={
                        "Split": st.column_config.TextColumn("Split"),
                        "Gain": st.column_config.NumberColumn("Gain", format="%.3f"),
                        "Left count": st.column_config.NumberColumn(
                            "Left count", format="%d"
                        ),
                        "Right count": st.column_config.NumberColumn(
                            "Right count", format="%d"
                        ),
                        "Left score": st.column_config.NumberColumn(
                            "Left score", format="%.3f"
                        ),
                        "Right score": st.column_config.NumberColumn(
                            "Right score", format="%.3f"
                        ),
                    },
                )
                st.markdown("**Residual snapshot before this tree**")
                snapshot = round_info.snapshot.copy()
                snapshot["probability_before"] = snapshot["probability_before"].round(3)
                snapshot["residual"] = snapshot["residual"].round(3)
                snapshot["tree_score"] = snapshot["tree_score"].round(3)
                snapshot["raw_score"] = snapshot["raw_score"].round(3)
                snapshot = snapshot.drop(columns=["existing_customer"])
                show_table(
                    snapshot,
                    column_config={
                        "age": st.column_config.NumberColumn("age", format="%d"),
                        "income": st.column_config.NumberColumn(
                            "income", format="$%d"
                        ),
                        "existing_status": st.column_config.TextColumn(
                            "existing_status"
                        ),
                        "accepted": st.column_config.NumberColumn(
                            "accepted", format="%d"
                        ),
                        "probability_before": st.column_config.NumberColumn(
                            "probability_before", format="%.3f"
                        ),
                        "residual": st.column_config.NumberColumn(
                            "residual", format="%.3f"
                        ),
                        "tree_score": st.column_config.NumberColumn(
                            "tree_score", format="%.3f"
                        ),
                        "raw_score": st.column_config.NumberColumn(
                            "raw_score", format="%.3f"
                        ),
                    },
                    height=320,
                )

            with right:
                st.markdown("**Tree learned in this round**")
                tree_tab, code_tab = st.tabs(["Visual tree", "Pseudo-code"])
                with tree_tab:
                    st.graphviz_chart(tree_to_graphviz(round_info.tree, probe))
                    st.caption("The green path is the route taken by the probe customer.")
                with code_tab:
                    st.code(format_tree_as_code(round_info.tree), language="python")
                round_row = gb_result["rounds"].loc[
                    gb_result["rounds"]["Tree"] == f"Tree {round_info.round_index}"
                    ].iloc[0]
                st.markdown("**Probe customer path through this tree**")
                st.write(round_row["Path"])
                st.caption(round_row["Leaf note"])

    st.info(
        "The split search should discover interaction rules such as `Age <= 35` followed by `Income > 10000`, or symbolic rules such as `Existing customer in {Existing}`. That is exactly the kind of conditional logic Naive Bayes does not create.",
    )

with tab_compare:
    st.header("Compare the Models")
    st.subheader("Same feature change, different model behavior")

    if scenario == "two_clusters":
        comparison_cases = [
            ("Young + high income + new", build_probe_customer(28, 14000, 0)),
            ("Young + low income + new", build_probe_customer(28, 7000, 0)),
            ("Older + low income + new", build_probe_customer(45, 7000, 0)),
            ("Older + low income + existing", build_probe_customer(45, 7000, 1)),
        ]
    else:
        comparison_cases = [
            ("Young + low income", build_probe_customer(28, 9000, 0)),
            ("Young + high income", build_probe_customer(28, 14000, 0)),
            ("Older + low income", build_probe_customer(45, 9000, 0)),
            ("Older + high income", build_probe_customer(45, 14000, 0)),
        ]

    rows = []
    for label, customer in comparison_cases:
        nb_score = score_naive_bayes(nb_model, customer)
        gb_score = score_gradient_boosting(gb_model, customer)
        rows.append(
            {
                "Scenario": label,
                "Naive Bayes probability": f"{nb_score['probability']:.1%}",
                "Gradient boosting probability": f"{gb_score['probability']:.1%}",
            }
        )

    comparison_frame = pd.DataFrame(rows)
    show_table(
        comparison_frame,
    )

    delta_left, delta_right = st.columns(2, gap="large")
    if scenario == "two_clusters":
        nb_income_delta = score_naive_bayes(nb_model, build_probe_customer(28, 14000, 0))[
                              "probability"
                          ] - score_naive_bayes(nb_model, build_probe_customer(28, 7000, 0))["probability"]
        nb_existing_delta = score_naive_bayes(nb_model, build_probe_customer(45, 7000, 1))[
                                "probability"
                            ] - score_naive_bayes(nb_model, build_probe_customer(45, 7000, 0))["probability"]

        gb_income_delta = score_gradient_boosting(
            gb_model, build_probe_customer(28, 14000, 0)
        )["probability"] - score_gradient_boosting(
            gb_model, build_probe_customer(28, 7000, 0)
        )["probability"]
        gb_existing_delta = score_gradient_boosting(
            gb_model, build_probe_customer(45, 7000, 1)
        )["probability"] - score_gradient_boosting(
            gb_model, build_probe_customer(45, 7000, 0)
        )["probability"]

        with delta_left:
            st.subheader("Naive Bayes deltas")
            st.metric("Income jump at age 28", f"{nb_income_delta:+.1%}")
            st.metric("Existing toggle at age 45, income 7k", f"{nb_existing_delta:+.1%}")
            st.caption(
                "Naive Bayes adds both effects independently. It does not build a separate second rule path."
            )
        with delta_right:
            st.subheader("Gradient boosting deltas")
            st.metric("Income jump at age 28", f"{gb_income_delta:+.1%}")
            st.metric("Existing toggle at age 45, income 7k", f"{gb_existing_delta:+.1%}")
            st.caption(
                "Gradient boosting can use one tree for the first region and another tree for the leftover existing-customer region."
            )
    else:
        nb_young_delta = score_naive_bayes(nb_model, build_probe_customer(28, 14000, 0))[
                             "probability"
                         ] - score_naive_bayes(nb_model, build_probe_customer(28, 9000, 0))["probability"]
        nb_older_delta = score_naive_bayes(nb_model, build_probe_customer(45, 14000, 0))[
                             "probability"
                         ] - score_naive_bayes(nb_model, build_probe_customer(45, 9000, 0))["probability"]

        gb_young_delta = score_gradient_boosting(
            gb_model, build_probe_customer(28, 14000, 0)
        )["probability"] - score_gradient_boosting(
            gb_model, build_probe_customer(28, 9000, 0)
        )["probability"]
        gb_older_delta = score_gradient_boosting(
            gb_model, build_probe_customer(45, 14000, 0)
        )["probability"] - score_gradient_boosting(
            gb_model, build_probe_customer(45, 9000, 0)
        )["probability"]

        with delta_left:
            st.subheader("Naive Bayes deltas")
            st.metric("Income jump at age 28", f"{nb_young_delta:+.1%}")
            st.metric("Income jump at age 45", f"{nb_older_delta:+.1%}")
            st.caption(
                "Those movements stay very similar because income evidence is added independently."
            )
        with delta_right:
            st.subheader("Gradient boosting deltas")
            st.metric("Income jump at age 28", f"{gb_young_delta:+.1%}")
            st.metric("Income jump at age 45", f"{gb_older_delta:+.1%}")
            st.caption(
                "Those movements differ because income matters inside a learned branch."
            )

    st.warning(
        "If you added a third feature that was almost a copy of age, Naive Bayes algorithm would still count it as separate evidence by design (though Pega ADM will detect correlation and rule out correlated feature). A tree model can treat such a feature as redundant and simply not split on it if it adds no gain.",
    )
with tab_wiki:
    with open(os.path.join(os.path.dirname(__file__), "docs/DEMO_WIKI.md"), "r") as f:
        readme_line = f.readlines()
        readme_buffer = []

    for line in readme_line:
        readme_buffer.append(line)

    st.markdown("".join(readme_buffer))
