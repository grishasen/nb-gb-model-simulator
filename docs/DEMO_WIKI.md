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

## What Changed In The Current Version

The current demo differs from the earlier version in two important ways.

### Naive Bayes now follows the Pega ADM flow more closely

The NB tab now shows:

- a `Base log-odds term`
- modified per-bin contributions
- a `Final ADM score`
- a classifier-table mapping from score to `Returned propensity`
- per-bin and classifier-bin `z-ratio` values

The presenter story is now:

- start from overall accepted vs rejected balance
- add one contribution per active predictor
- average into a final score
- map that score to adjusted propensity through the classifier table

### Gradient boosting now uses a more Pega-like scoring story

The GB tab now shows:

- a `Bias score`
- one visited `Tree score` per tree
- a `Pega-style per-tree contribution view`
- symbolic split text for categorical logic such as `Existing customer in {Existing}`

The presenter story is now:

- start from the bias score
- take one score from each visited tree
- sum those tree scores with the bias score
- apply sigmoid to get the final propensity

Important caveat:

- the GB training-side split search is still intentionally simplified for education
- the scoring explanation is now closer to the Pega mental model than the training algorithm itself

## Scenario 1: Single Positive Region

<img alt="image" src="https://github.com/user-attachments/assets/a634f551-bacb-406f-ab26-00e09574f274" />


### Hidden Rule

Positive only if:

- `age < 35`
- and `income >= 11000`

This is the cleanest scenario for teaching a one strong interaction.

<img alt="NB Overview" src="https://github.com/user-attachments/assets/a2a6cf0c-9f3d-4a0f-a60f-18c660f8f78a" />


### Recommended Live Walkthrough

1. Select `Single Positive Region`.
2. Set the probe customer to:
   - `Age = 30`
   - `Income = 12000`
   - `Existing customer = no`
3. Open `Story`.
4. Explain that there is one positive region in the feature space.
5. Open `Naive Bayes`.
6. Point to the three headline metrics:
   - `Base log-odds term`
   - `Final ADM score`
   - `Returned propensity`
7. Explain the formula block:
   - base term from overall accepted vs rejected balance
   - one contribution from age band
   - one contribution from income band
   - one contribution from existing-customer status
   - divide by `1 + number of active predictors`
   - map the score through the classifier table
8. Use the `Contribution chart` to show that the current NB contribution for a feature is fixed once the customer is in a bin.
9. Use `Score breakdown for the probe customer` to show:
   - `Smoothed P(bin|accepted)`
   - `Smoothed P(bin|rejected)`
   - `Bin z-ratio`
   - `Contribution`
10. Use `What the model learned from bins` to show that each feature is scored separately.
11. Open the `Classifier table` tab and explain that the final score is converted to adjusted propensity there.
12. Open `Gradient Boosting`.
13. Point to the three headline metrics:
   - `Bias score`
   - `Final raw score`
   - `Final probability`
14. Explain the formula block:
   - bias score from overall accepted rate
   - add one `Tree score` per tree
   - apply sigmoid to get the final probability
15. Show the `Pega-style per-tree contribution view`.
16. Open `Boosting round 1` and show that the first tree learns the age then income interaction.
17. Open `Boosting round 2` and explain that it often looks similar but with a smaller correction because it is fixing what is still left over.
18. Open `Compare`.
19. Show that changing income has a similar NB effect regardless of age band, while in GB the effect depends on the branch path.

### What To Say

- “Naive Bayes adds independent evidence from each predictor.”
- “Once a customer falls into an NB bin, that bin has a fixed contribution and a fixed z-ratio.”
- “Gradient boosting follows rule paths, so income can matter differently inside different branches.”
- “In this scenario, the second tree often reinforces the same main pattern with a smaller score.”

### Expected Takeaway

This scenario is best for explaining:

- independence in Naive Bayes
- one strong interaction in gradient boosting
- why later GB trees may look like smaller corrections of the same rule

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
   - a young and higher-income region
   - an existing-customer and lower-income region
4. Start with probe customer:
   - `Age = 30`
   - `Income = 12000`
   - `Existing customer = no`
5. Open `Gradient Boosting`.
6. Show the `Pega-style per-tree contribution view`.
7. In `Boosting round 1`, show that the model usually focuses on the young and higher-income pattern first.
8. Then switch the probe customer to:
   - `Age = 45`
   - `Income = 7000`
   - `Existing customer = yes`
9. Show that a later tree can pivot to the existing-customer region.
10. Call out the symbolic split text if it appears:
   - `Existing customer in {Existing}`
11. Explain that this is exactly the kind of explicit conditional rule that NB does not create.
12. Open `Naive Bayes`.
13. Show that NB can assign positive evidence to:
   - the `Existing customer` bin
   - the lower-income bin
14. Then emphasize that NB still adds them as separate contributions rather than creating one combined path.
15. Use the `Classifier table` to explain that a final NB score still becomes propensity only after score-to-bin mapping.
16. Open `Compare`.
17. Highlight that GB can use different trees to cover different leftover regions, while NB remains additive.

<img alt="Tree 1" src="https://github.com/user-attachments/assets/4cad8695-a759-4a88-93db-b3344120901a" />

<img alt="Tree 2" src="https://github.com/user-attachments/assets/6ae47a55-0405-43f1-b616-951847a8d3dc" />

### What To Say

- “The second positive region gives boosting a second job.”
- “One tree can explain the main region, and another tree can chase the leftover region.”
- “Naive Bayes can add separate evidence from `existing customer` and `low income`, but it still does not create one explicit rule path for that combination.”

### Expected Takeaway

This scenario is best for explaining:

- why a second tree can look different from the first
- how boosting handles leftover structure sequentially
- how symbolic or alternative root splits can appear later

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
2. In `Naive Bayes`, say:
   - “NB adds one contribution per predictor, averages those contributions into a final score, then maps that score to propensity through a classifier table.”
3. In `Gradient Boosting`, say:
   - “GB starts from a bias score, takes one score from each visited tree, and turns the final score into probability.”
4. Switch to `Two Positive Regions`.
5. Show that a later GB tree can pivot to another rule pattern, including `Existing customer`.
6. Close with:
   - Naive Bayes: “sum of independent evidence, then classifier mapping”
   - Gradient boosting: “sum of rule-based tree scores”

## Teaching Notes

- Use `Single Positive Region` first because it is easier to understand.
- Use `Two Positive Regions` second because it answers the natural question: “Why do we need another tree at all?”
- If the audience is non-technical, spend more time on the `Compare` tab and the contribution charts than on the root-candidate tables.
- If the audience is technical, use the NB `Classifier table`, the NB `z-ratio` fields, the GB `Pega-style per-tree contribution view`, and the per-tree scoring trace.
- In the NB tab, avoid saying “sigmoid” because the current explanation is classifier-table based.
- In the GB tab, avoid saying “leaf value times learning rate” because the current explanation is tree-score based.
