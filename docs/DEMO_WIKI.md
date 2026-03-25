# Demo Wiki

This page is a presenter-friendly walkthrough for the Streamlit demo.

It covers both teaching scenarios:

- `Single Positive Region`
- `Two Positive Regions`
- `Messy Real-World Pattern`

The goal is to make it easy to explain:

- what Naive Bayes is doing
- what gradient boosting is doing
- why those behaviors differ
- why GB often has an advantage once the data gets noisy and locally irregular

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

The `Compare` tab now has two roles:

- in the first two scenarios, it highlights score differences on hand-picked probe customers
- in the third scenario, it also shows train-versus-holdout quality metrics

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
- sidebar controls for `Trees`, `Depth`, and `Learning rate`
- a `Raw sigmoid probability`
- a `Calibrated probability`
- symbolic split text for categorical logic such as `Existing customer in {Existing}`

The presenter story is now:

- start from the bias score
- take one score from each visited tree
- sum those tree scores with the bias score
- convert the raw score to a raw sigmoid probability
- apply a simple calibration function on top of that raw score

Important caveat:

- the GB training-side split search is still intentionally simplified for education
- the scoring explanation is now closer to the Pega mental model than the training algorithm itself
- the current version is more realistic than before because tree leaves now use logistic gradient/hessian information instead of simple mean residuals

## Scenario 3: Messy Real-World Pattern

### Hidden Pattern

This is not a perfect rule-based dataset.

Instead, the probability of acceptance is higher in some regions and lower in others:

- higher for very young customers with sufficiently high income
- higher for some existing customers with lower income
- lower in a few local exception pockets

This scenario also includes:

- overlap between accepted and rejected outcomes
- train and holdout datasets
- broad bins that hide finer thresholds from Naive Bayes

### Why This Scenario Exists

The first two scenarios teach structure.

This third scenario teaches generalization:

- NB still uses the same broad bins and independent contributions
- GB can insert extra thresholds inside those broad bins
- a holdout set lets you show that this matters on unseen data, not just on one probe customer

### Recommended Live Walkthrough

1. Select `Messy Real-World Pattern`.
2. Open `Story`.
3. Explain that the shaded regions are now only higher-propensity regions, not deterministic rules.
4. Point out that there is a `Training Data` section and a `Holdout dataset` section.
5. Open `Compare`.
6. Start with the score table and call out these two pairs:
   - `Age 22 + 14.5k + new`
   - `Age 34 + 14.5k + new`
   - `Age 22 + 12.5k + new`
   - `Age 34 + 12.5k + new`
7. Explain the key point:
   - NB gives each pair the same score because both customers fall into the same broad age and income bins
   - GB separates them because it can place extra age thresholds inside those bins
8. Stay in `Compare` and show the `Generalization on unseen data` table.
9. Emphasize that `Holdout AUC` is the main metric to watch here.
10. Point out that the GB sidebar defaults are stronger here:
   - more trees
   - slightly deeper trees
   - lower learning rate
11. Explain that the updated GB demo now uses logistic-style leaf scores plus a simple calibration layer, so the probabilities are more realistic than before.
12. Open `Gradient Boosting`.
13. Show the tree paths and say:
   - “These extra thresholds are exactly what let GB pull apart customers who looked identical to Naive Bayes.”
14. Point to the `Raw sigmoid probability` and `Calibrated probability` metrics and explain the difference.
15. Open `Naive Bayes`.
16. Show that NB still has one contribution per broad bin and cannot create those finer pockets inside a bin.

### What To Say

- “Real data is rarely a perfect rectangle.”
- “Naive Bayes has to summarize each broad band with one contribution.”
- “Gradient boosting can keep splitting inside that band and create local pockets.”
- “That is why GB often wins once the data has overlap, exceptions, and hidden finer thresholds.”

### Expected Takeaway

This scenario is best for explaining:

- why GB often ranks real customers better on messy data
- why independent bin contributions can be too coarse
- why a holdout set matters when comparing models

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
   - `Raw sigmoid probability`
   - `Calibrated probability`
14. Explain the formula block:
   - bias score from overall accepted rate
   - add one `Tree score` per tree
   - apply sigmoid to get a raw probability
   - apply the calibration function to get the displayed probability
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

### Messy Real-World Pattern

- `22, 14500, new`
- `34, 14500, new`
- `22, 12500, new`
- `34, 12500, new`

## Fast Explanation Script

If you only have two to three minutes:

1. Start with `Single Positive Region`.
2. In `Naive Bayes`, say:
   - “NB adds one contribution per predictor, averages those contributions into a final score, then maps that score to propensity through a classifier table.”
3. In `Gradient Boosting`, say:
   - “GB starts from a bias score, takes one score from each visited tree, and turns the final score into probability.”
4. Switch to `Messy Real-World Pattern`.
5. Show that two customers in the same NB bins can get different GB scores.
6. Point to the holdout `AUC` comparison.
7. Close with:
   - Naive Bayes: “sum of independent evidence, then classifier mapping”
   - Gradient boosting: “sum of rule-based tree scores”

## Teaching Notes

- Use `Single Positive Region` first because it is easier to understand.
- Use `Two Positive Regions` second because it answers the natural question: “Why do we need another tree at all?”
- Use `Messy Real-World Pattern` third because it shows why the structural difference matters on holdout data.
- If the audience is non-technical, spend more time on the `Compare` tab and the contribution charts than on the root-candidate tables.
- If the audience is technical, use the NB `Classifier table`, the NB `z-ratio` fields, the GB `Pega-style per-tree contribution view`, and the per-tree scoring trace.
- In the third scenario, emphasize `Holdout AUC` before `log loss`, because the simplified GB demo is most convincing there as a ranking model.
- In the current GB tab, explicitly mention the difference between `Raw sigmoid probability` and `Calibrated probability`.
- In the NB tab, avoid saying “sigmoid” because the current explanation is classifier-table based.
- In the GB tab, avoid saying “leaf value times learning rate” because the current explanation is tree-score based.
