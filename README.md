# Model Simulator Demo

This project is a small Streamlit app that teaches the difference between:

- Naive Bayes style scoring with independent feature contributions
- Tree-based gradient boosting with conditional split rules

The demo now includes a scenario selector and three simple features:

- `age`
- `income`
- `existing_customer`

There are two teaching scenarios:

- `Single Positive Region`
  - positive if `age < 35` and `income >= 11000`
- `Two Positive Regions`
  - positive if `age < 35` and `income >= 11000`
  - or positive if `existing_customer = yes` and `income <= 9000`

This supports two different lessons:

- Naive Bayes can add evidence from age and income, but it does not create a learned rule that says "income matters only inside the younger segment"
- Gradient boosting can learn exactly that kind of path with a tree
- With the second scenario, a later tree can pivot to a different feature and explain a second leftover cluster
## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Notes

- The Naive Bayes section is intentionally simplified but uses binned, smoothed log-odds contributions inspired by the Pega ADM explanation.
- The gradient boosting section is also simplified for education: it fits shallow regression trees to logistic residuals so the split logic stays transparent.
- A presenter-facing walkthrough lives in `docs/DEMO_WIKI.md`.
