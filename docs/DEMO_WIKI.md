# Demo Wiki

This page is a presenter-friendly walkthrough for the Streamlit demo.

It covers both teaching scenarios:

- `Single Positive Region`
- `Two Positive Regions`

The goal is to make it easy to explain:

- what Naive Bayes is doing
- what gradient boosting is doing
- why those behaviors differ

## App Structure

The app is organized into four tabs:

1. `Story`
2. `Naive Bayes`
3. `Gradient Boosting`
4. `Compare`

Use the sidebar to control:

- dataset scenario
- probe customer age
- probe customer income
- whether the probe customer is an existing customer

## Scenario 1: Single Positive Region

### Hidden Rule

Positive only if:

- `age < 35`
- and `income >= 11000`

This is the cleanest scenario for teaching a single interaction.

### Recommended Live Walkthrough

1. Select `Single Positive Region`.
2. Set the probe customer to:
   - `Age = 30`
   - `Income = 12000`
   - `Existing customer = no`
3. Open `Story`.
4. Explain that the positive region sits in one corner of the feature space.
5. Open `Naive Bayes`.
6. Point out that the score is:
   - prior
   - plus age-band contribution
   - plus income-band contribution
   - plus existing-customer contribution
7. Emphasize that those pieces are added independently.
8. Open `Gradient Boosting`.
9. Show that the first tree learns the age then income rule.
10. Show that the second tree looks similar, but with smaller leaf values.
11. Open `Compare`.
12. Show that changing income has a similar effect in Naive Bayes no matter the age, while in gradient boosting the effect depends on the branch.

### What To Say

- “Naive Bayes treats each feature as a separate source of evidence.”
- “Gradient boosting can build a path like `if age is young, then check income`.”
- “That is why GB can express interaction rules directly.”

### Expected Takeaway

This scenario is best for explaining:

- independence in Naive Bayes
- one strong interaction in gradient boosting
- why later trees may look like smaller corrections of the same rule

## Scenario 2: Two Positive Regions

### Hidden Rule

Positive if either:

- `age < 35 and income >= 11000`
- or `existing_customer = yes and income <= 9000`

This scenario is designed to make a later tree useful for a different leftover pattern.

### Recommended Live Walkthrough

1. Select `Two Positive Regions`.
2. Open `Story`.
3. Explain that there are now two different reasons to be positive:
   - a young and higher-income cluster
   - an existing-customer and lower-income cluster
4. Start with probe customer:
   - `Age = 30`
   - `Income = 12000`
   - `Existing customer = no`
5. Open `Gradient Boosting`.
6. Show that tree 1 usually focuses on the young and higher-income cluster.
7. Then change the probe customer to:
   - `Age = 45`
   - `Income = 7000`
   - `Existing customer = yes`
8. Show that tree 2 can pivot to the `Existing customer` feature and explain the second positive region.
9. Open `Naive Bayes`.
10. Show that Naive Bayes can give positive evidence for `existing customer` and for `low income`, but it still adds them as separate terms.
11. Open `Compare`.
12. Highlight that gradient boosting can use different trees to cover different leftover regions, while Naive Bayes still stays additive.

### What To Say

- “The second cluster gives boosting a new job after the first tree.”
- “Tree 1 handles the main pattern; tree 2 goes after what tree 1 left behind.”
- “Naive Bayes still does not create a second explicit rule path. It just adds more evidence.”

### Expected Takeaway

This scenario is best for explaining:

- why a second tree can look different from the first
- how boosting handles residual structure sequentially
- how a later tree may start from another feature

## Suggested Probe Customers

These probes work well during a live demo.

### Single Positive Region

- `30, 12000, new`
- `30, 9000, new`
- `45, 12000, new`
- `45, 9000, new`

### Two Positive Regions

- `30, 12000, new`
- `30, 7000, new`
- `45, 7000, new`
- `45, 7000, existing`

## Fast Explanation Script

If you only have two to three minutes:

1. Start with `Single Positive Region`.
2. Show that Naive Bayes adds independent feature evidence.
3. Show that gradient boosting learns a split path.
4. Switch to `Two Positive Regions`.
5. Show that tree 2 can pivot to a different feature.
6. Close with:
   - Naive Bayes: “sum of independent evidence”
   - Gradient boosting: “sum of sequential rule corrections”

## Teaching Notes

- Use `Single Positive Region` first because it is easier to understand.
- Use `Two Positive Regions` second because it answers the natural question: “Why do we need another tree at all?”
- If the audience is non-technical, spend more time on the `Compare` tab than on the split candidate tables.
- If the audience is technical, use the `Gradient Boosting` tab to connect residuals, leaf values, raw score, and sigmoid propensity.
