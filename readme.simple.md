# Domain Adaptation for Finance - Simple Explanation

## What is Domain Adaptation? (The Simple Version)

Imagine you learned to ride a bike in your quiet neighborhood.
You know the streets, the turns, and where all the bumps are.
One day, your family goes on vacation to a completely different country.
You rent a bike there and try to ride it.

The bike is the same. Your legs still work. You still know how to pedal.
But everything around you is different!
The roads are narrower. People drive on the other side.
The traffic signs look weird. The pavement feels different.

You can still ride the bike, but you need to **adapt** to the new place.

That is exactly what **domain adaptation** is about in the world of computers.

A computer program (we call it a "model") learns patterns from one set of data.
We call that set the **source domain** -- think of it as your home neighborhood.
Then we want the model to work on a different set of data.
We call that the **target domain** -- think of it as the foreign country.

The tricky part?
The two sets of data look different on the surface,
but the underlying rules and patterns are often similar.

Domain adaptation is the set of tricks and techniques we use
to help the model perform well in the new, unfamiliar data
without having to learn everything from scratch.


---


## Why Do We Need This for Trading?

Trading is all about predicting what happens next in financial markets.
People build computer models that look at past prices, volumes, and news
to guess whether a stock will go up or down.

But here is the problem: **financial data changes all the time.**

Think of it like weather.
If you learn to predict weather in Arizona (sunny almost every day),
your predictions would be terrible in London (rain, clouds, surprise sunshine).
Same planet, same atmosphere, but very different patterns!

In finance, these "weather changes" happen because of many reasons:

### Different Markets
- A model trained on **US stocks** might not work on **Japanese stocks**.
- A model trained on **stocks** might fail on **cryptocurrency**.
- A model trained on **large companies** might not apply to **small companies**.

### Different Time Periods
- Markets in 2010 behaved very differently from markets in 2020.
- A calm market (like a quiet lake) looks nothing like a crisis (like a stormy sea).
- Rules that worked before a big event (like COVID) might not work after.

### Different Data Sources
- Some data comes from stock exchanges, some from social media.
- Some data is clean and organized, some is messy and noisy.
- Different brokers provide data in slightly different formats.

Without domain adaptation, your model is like a student who memorized
answers for one test and then gets a completely different test.
The student knows the subject, but the questions look unfamiliar.

Domain adaptation helps the model say:
"Okay, this looks different, but I can still figure it out."


---


## The Main Ideas

There are three big ideas in domain adaptation.
Let us walk through each one with a fun analogy.


---


### Idea 1: Making Features Look the Same (DANN)

**DANN** stands for **Domain-Adversarial Neural Network**.
Big words! But the idea is simple.

#### The Uniform Analogy

Imagine two schools: School A and School B.
Kids in School A wear red shirts and blue pants.
Kids in School B wear green shirts and yellow pants.

If you see a kid, you can instantly tell which school they come from
just by looking at their clothes.

Now imagine we make ALL kids wear the same uniform -- a plain white shirt
and gray pants. Suddenly, you cannot tell which school a kid comes from
just by looking at them!

**That is what DANN does with data.**

It takes data from two different sources (like stocks and crypto)
and transforms them so that a computer cannot tell
which source the data came from.

#### How Does DANN Actually Work?

DANN has three parts, like a team of three workers:

1. **The Feature Extractor** (the Transformer)
   - This worker takes the raw data and converts it into useful features.
   - Think of it as a translator who reads the data and writes a summary.

2. **The Label Predictor** (the Smart One)
   - This worker tries to predict the answer (will the price go up or down?).
   - It reads the summary from the Transformer and makes a guess.

3. **The Domain Classifier** (the Spy)
   - This worker tries to figure out where the data came from.
   - "Was this from stocks or from crypto?"

Here is the clever trick:

The Transformer is trained to do TWO things at once:
- Help the Smart One make good predictions (this is the main job).
- **Fool the Spy** so it CANNOT tell where the data came from.

It is like a game! The Spy gets better at detecting differences,
and the Transformer gets better at hiding them.
Over time, the Transformer learns to create summaries that are
useful for prediction but hide the source of the data.

This is called **adversarial training** -- the two parts compete
against each other, and both get better!

```
Raw Data (stocks) --\
                     --> [Feature Extractor] --> Features --> [Label Predictor] --> Up/Down?
Raw Data (crypto) --/                              |
                                                   v
                                          [Domain Classifier]
                                          "Which source?" --> CONFUSED! (good!)
```

When the Domain Classifier is confused and cannot tell the difference,
it means the features are truly universal -- they work for both domains!


---


### Idea 2: Matching Distributions (MMD)

**MMD** stands for **Maximum Mean Discrepancy**.
Another set of big words, but the idea is beautiful and simple.

#### The Paint Mixing Analogy

Imagine you have two buckets of paint.
Bucket A has a reddish-orange color.
Bucket B has a yellowish-orange color.

They are both "orange," but they look slightly different.

Your goal is to adjust both buckets until they look EXACTLY the same.
You add a tiny bit of red to Bucket B, a tiny bit of yellow to Bucket A,
and keep mixing until you cannot tell the buckets apart.

**That is what MMD does with data.**

It measures how different two groups of data are,
and then adjusts the model until the two groups look the same
from the model's perspective.

#### How Does MMD Work?

1. **Take a bunch of data points from Source A** (e.g., stock market data).
2. **Take a bunch of data points from Source B** (e.g., crypto market data).
3. **Transform both groups** through the model.
4. **Calculate the average** (mean) of each group after transformation.
5. **Measure the gap** between the two averages.
6. **Adjust the model** to make that gap smaller.
7. **Repeat** until the gap is tiny.

The "discrepancy" in MMD is just a fancy word for "difference."
We are measuring the maximum difference between the means (averages)
of the two groups and trying to shrink it.

```
Source Data:  [3, 7, 5, 8, 2]   Average = 5.0
Target Data:  [10, 14, 12, 15, 9]  Average = 12.0

Gap = 12.0 - 5.0 = 7.0  (too big!)

After transformation:
Source Features:  [6, 8, 7, 9, 5]   Average = 7.0
Target Features:  [7, 9, 8, 10, 6]  Average = 8.0

Gap = 8.0 - 7.0 = 1.0  (much better!)
```

The smaller the gap, the more similar the two groups are,
and the better the model will work on both.

Actually, MMD uses a clever mathematical trick called a **kernel function**
that captures not just the average but also the shape and spread of the data.
But the core idea remains: measure the difference and shrink it.


---


### Idea 3: Aligning Statistics (CORAL)

**CORAL** stands for **CORrelation ALignment**.
This one has a great analogy.

#### The Photo Editing Analogy

Imagine you took photos on two different cameras.
Camera A takes photos that are very bright and warm (yellowish).
Camera B takes photos that are dark and cool (bluish).

If you put these photos side by side, they look very different,
even though they might show the same things!

What do professional photographers do?
They adjust the **brightness**, **contrast**, and **color balance**
of all photos so they look consistent.

**That is what CORAL does with data.**

It looks at the **statistics** of each data source:
- How spread out are the numbers? (variance)
- How do different features relate to each other? (correlation)

Then it adjusts one dataset so its statistics match the other.

#### How Does CORAL Work?

1. **Calculate the covariance matrix** for the source data.
   - This is like measuring the brightness and contrast of Camera A photos.
   - It tells you how each feature varies and how features relate to each other.

2. **Calculate the covariance matrix** for the target data.
   - Same measurement, but for Camera B photos.

3. **Adjust the source data** so its covariance matches the target.
   - Like editing Camera A photos to look like Camera B photos.

Think of covariance as a "fingerprint" of the data.
Two datasets might have very different numbers,
but if their fingerprints match, a model can treat them the same way.

```
Source data "fingerprint":          Target data "fingerprint":
| 4.0  1.5 |                       | 3.8  1.6 |
| 1.5  2.0 |                       | 1.6  2.1 |

These are close! After CORAL alignment, they become even closer.
The model sees both datasets as having the same "shape."
```

CORAL is popular because it is simple and fast.
You do not need complex neural networks -- just some matrix math!


---


## How Does It Work in Trading?

Let us walk through a real-world scenario step by step.

### The Scenario

You work at an investment company.
You built an amazing model that predicts stock prices in the US market.
Now your boss says: "Make it work for European stocks too!"

But European stocks are different:
- They trade in different currencies (euros, pounds, francs).
- They follow different economic cycles.
- They react to different news events.
- Trading hours are different.

Your model was trained on US data. It has never seen European data.
What do you do?

### Step 1: Collect Data from Both Domains

You gather data from both markets:

| Feature          | US Stocks (Source) | European Stocks (Target) |
|------------------|--------------------|--------------------------|
| Daily return     | -2% to +3%         | -1.5% to +2%            |
| Trading volume   | 1M - 100M shares   | 500K - 50M shares       |
| Volatility       | 10% - 40%          | 8% - 35%                |
| Price-to-Earnings| 10 - 50            | 8 - 40                  |

Notice how the ranges are similar but not identical?
That is the "domain shift" we need to handle.

### Step 2: Choose a Domain Adaptation Method

You pick one (or a combination) of our three methods:
- **DANN** if you want the model to automatically learn domain-invariant features.
- **MMD** if you want to directly minimize the gap between the two distributions.
- **CORAL** if you want a quick, simple statistical alignment.

### Step 3: Train with Both Datasets

Here is the key insight: **you use labeled data from the source
and unlabeled (or partially labeled) data from the target.**

For US stocks, you know the answers (did the price go up or down?).
For European stocks, you might not have labels yet,
but you still feed the raw data into the model.

The domain adaptation loss pushes the model to treat both datasets similarly.

### Step 4: Fine-Tune and Validate

After training, you test the model on European stocks.
You compare it to:
- A model trained ONLY on US data (no adaptation) -- usually performs poorly.
- A model trained ONLY on European data -- might be great but needs lots of labeled data.
- Your adapted model -- should perform well even with limited European labels!

### Step 5: Deploy and Monitor

You put the model into production, but you keep watching.
Markets change over time (concept drift), so you might need to
re-adapt the model periodically.

```
Training Pipeline:

  US Stock Data -----> [Shared Feature Extractor] ----> [Predictor] ----> Buy/Sell Signal
  (with labels)             |                              ^
                            |                              |
  EU Stock Data -----> [Shared Feature Extractor] ----> [Predictor] ----> Buy/Sell Signal
  (few/no labels)           |
                            v
                    [Domain Adaptation Loss]
                    "Make both domains look the same!"
```


---


## A Simple Example

Let us walk through a tiny, concrete example with actual numbers.

### The Setup

We have a very simple trading model.
It looks at just TWO features to decide whether to buy or sell:

- **Feature 1**: Price momentum (how fast the price has been moving)
- **Feature 2**: Volume change (how much trading activity changed)

We trained this model on **stock market** data (Source Domain).
Now we want it to work on **cryptocurrency** data (Target Domain).

### The Data

**Stock Market Data (Source):**

| Day | Momentum | Volume Change | Label     |
|-----|----------|---------------|-----------|
| 1   | 0.5      | 0.3           | Buy       |
| 2   | -0.8     | 0.1           | Sell      |
| 3   | 0.3      | 0.7           | Buy       |
| 4   | -0.6     | -0.2          | Sell      |
| 5   | 0.9      | 0.5           | Buy       |

**Cryptocurrency Data (Target):**

| Day | Momentum | Volume Change | Label     |
|-----|----------|---------------|-----------|
| 1   | 2.5      | 1.8           | ???       |
| 2   | -3.1     | 0.9           | ???       |
| 3   | 1.7      | 3.2           | ???       |
| 4   | -2.4     | -1.1          | ???       |
| 5   | 4.1      | 2.7           | ???       |

Notice something?
- Stock data has small numbers (between -1 and 1).
- Crypto data has big numbers (between -3 and 4).
- Crypto is much more volatile (wild swings)!

If we just run our stock model on crypto data, it will be confused.
It has never seen numbers this big!

### Without Domain Adaptation

The model was trained on stocks where:
- Momentum > 0.2 usually means "Buy"
- Momentum < -0.2 usually means "Sell"

For crypto, momentum of 2.5 is "normal" -- but the model thinks
it is EXTREME because it has only seen values up to 0.9!

Result: The model panics and makes bad decisions.

### With CORAL (Statistical Alignment)

Let us apply CORAL to fix this.

**Step 1: Calculate statistics of both domains.**

Stock momentum: mean = 0.06, standard deviation = 0.68
Crypto momentum: mean = 0.56, standard deviation = 2.87

**Step 2: Transform crypto data to match stock statistics.**

For each crypto value, we apply this formula:

```
adjusted_value = (crypto_value - crypto_mean) / crypto_std * stock_std + stock_mean
```

Let us adjust crypto Day 1 momentum:

```
adjusted = (2.5 - 0.56) / 2.87 * 0.68 + 0.06
adjusted = (1.94) / 2.87 * 0.68 + 0.06
adjusted = 0.676 * 0.68 + 0.06
adjusted = 0.46 + 0.06
adjusted = 0.52
```

Now 0.52 is in the same range as stock data!
The model can understand it.

**After adjustment, crypto data looks like:**

| Day | Adjusted Momentum | Original Momentum |
|-----|--------------------|--------------------|
| 1   | 0.52               | 2.5               |
| 2   | -0.81              | -3.1              |
| 3   | 0.33               | 1.7               |
| 4   | -0.64              | -2.4              |
| 5   | 0.90               | 4.1               |

Look at that! The adjusted values match the stock data range perfectly.
And the patterns are preserved:
- Days with positive momentum are still positive.
- Days with negative momentum are still negative.
- The relative ordering is the same.

Now our stock model can make sensible predictions on crypto data!

### With MMD (Distribution Matching)

Instead of manually adjusting statistics, MMD trains the neural network
to automatically transform the data.

**Step 1: Pass both datasets through the feature extractor.**

Stock features:   [0.5, -0.8, 0.3, -0.6, 0.9]  -> average = 0.06
Crypto features:  [2.5, -3.1, 1.7, -2.4, 4.1]  -> average = 0.56

**Step 2: Calculate MMD loss.**

```
MMD = |average(stock_features) - average(crypto_features)|
MMD = |0.06 - 0.56| = 0.50
```

**Step 3: Train the model to reduce MMD while keeping predictions accurate.**

After several rounds of training:

Stock features:   [0.4, -0.6, 0.3, -0.5, 0.7]  -> average = 0.06
Crypto features:  [0.5, -0.7, 0.3, -0.5, 0.8]  -> average = 0.08

```
MMD = |0.06 - 0.08| = 0.02   (much smaller!)
```

The model learned to map both domains into a common feature space!

### With DANN (Adversarial Training)

DANN adds a "spy" network that tries to guess "is this stocks or crypto?"

**Round 1:**
- Feature Extractor produces features for both domains.
- The Spy easily tells them apart (98% accuracy). "Easy! Big numbers = crypto."
- The Feature Extractor is punished and adjusts itself.

**Round 2:**
- Feature Extractor makes features more similar.
- The Spy can still tell them apart but with lower accuracy (75%).
- Feature Extractor keeps adjusting.

**Round 10:**
- Features from both domains look very similar.
- The Spy is just guessing randomly (50% accuracy -- like flipping a coin).
- The Feature Extractor wins the game!

At this point, the features are truly domain-invariant.
The predictor can use them for both stocks and crypto.


---


## Types of Domain Adaptation

Before we wrap up, let us briefly look at the different flavors
of domain adaptation you might encounter:

### By Label Availability

| Type                  | Source Labels | Target Labels | Example                          |
|-----------------------|---------------|---------------|----------------------------------|
| Unsupervised          | Yes           | No            | Trained on stocks, apply to crypto |
| Semi-supervised       | Yes           | A few         | Some crypto labels available     |
| Supervised            | Yes           | Yes           | Labels for both, but different distributions |

### By What Changes

| Type                  | What Changes              | Example                               |
|-----------------------|---------------------------|---------------------------------------|
| Covariate Shift       | Input distribution        | Same features, different ranges       |
| Prior Probability Shift| Label distribution       | More "buy" days in one market         |
| Concept Drift         | Relationship between X and Y | Rules of the market change over time |

### By Number of Sources

| Type                  | Description                           |
|-----------------------|---------------------------------------|
| Single-source         | Adapt from one domain to another      |
| Multi-source          | Combine knowledge from many domains   |


---


## When Does Domain Adaptation NOT Work?

Domain adaptation is not magic. It fails in some situations:

1. **The domains are too different.**
   If source and target have absolutely nothing in common,
   no amount of adaptation will help.
   Like trying to use a cooking recipe to fix a car -- just too different.

2. **The source model is bad.**
   If your model does not even work well on the source data,
   adapting it to a new domain will not help.
   You need a good foundation first.

3. **The features do not overlap.**
   If the source uses features A, B, C and the target uses X, Y, Z
   with no connection between them, adaptation cannot bridge the gap.

4. **Negative transfer.**
   Sometimes, trying to adapt actually makes things WORSE.
   This happens when the model forces incorrect similarities
   between domains that are genuinely different.
   It is like a student who applies math rules to an English essay --
   forcing a connection that should not be there.


---


## Key Takeaways

Here are the most important things to remember:

- **Domain adaptation** helps models work on new, unseen data
  by bridging the gap between training data (source) and real-world data (target).

- **Why it matters for trading**: Financial markets change across regions,
  time periods, and asset classes. Models need to adapt to survive.

- **Three main techniques**:
  - **DANN**: Uses a "spy" game to force features to be domain-invariant.
    The model learns to hide where the data came from.
  - **MMD**: Measures and minimizes the statistical gap between two domains.
    Like mixing paint until two buckets match.
  - **CORAL**: Aligns the covariance (shape and spread) of two datasets.
    Like adjusting photo settings so all pictures look consistent.

- **The key insight**: You do not need lots of labeled data from the new domain.
  You can transfer knowledge from a domain where you DO have labels.

- **It is not magic**: Domain adaptation works best when the domains
  share some underlying structure. Completely unrelated domains cannot be adapted.

- **Real-world trading applications**:
  - Adapt stock models to work on cryptocurrency.
  - Transfer models across different country markets.
  - Handle regime changes (bull market to bear market).
  - Apply models trained on large-cap stocks to small-cap stocks.

- **Always validate**: Test your adapted model carefully before risking real money.
  Domain adaptation improves things, but it is not a guarantee.

- **Monitor continuously**: Markets evolve, so domain adaptation
  is not a one-time fix. Keep watching and re-adapting as needed.


---


## Quick Glossary

| Term                  | Simple Meaning                                            |
|-----------------------|-----------------------------------------------------------|
| Domain                | A specific dataset or environment (like "US stocks")      |
| Source Domain          | The data you trained on (your "home turf")               |
| Target Domain          | The new data you want to work on (the "foreign country") |
| Domain Shift           | The difference between source and target                 |
| Feature Extractor      | Part of the model that converts raw data to useful features |
| Domain-Invariant       | Features that look the same regardless of the domain     |
| Adversarial Training   | Training two parts of a model to compete against each other |
| Covariance Matrix      | A table showing how features vary and relate to each other |
| Transfer Learning      | The broad idea of reusing knowledge from one task to another |
| Negative Transfer      | When adaptation makes performance worse instead of better |


---


*Remember: domain adaptation is like being a good traveler.
You take the skills you already have and adjust them
to work in a new place. The better you adapt,
the more successful you will be -- whether traveling or trading!*
