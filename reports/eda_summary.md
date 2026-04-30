# What We Found in the Data

## About This Dataset

We analyzed **150,000 U.S. domestic flights**, combining flight schedule and outcome records from the Bureau of Transportation Statistics with daily weather data at each flight's departure and arrival airport.

For each flight we tracked: when it was scheduled to depart, which airline operated it, the route, the distance, and whether conditions at both airports were clear, rainy, hot, cold, or windy. Our goal was to predict whether a flight would arrive **more than 15 minutes late** — the industry-standard threshold for a reportable delay.

A few things about the dataset worth knowing up front:

- Flights that were cancelled or diverted are excluded — we're only predicting delays for flights that actually operated.
- Weather data wasn't available for every flight, so roughly 8–9% of rows are missing some weather fields. We fill those gaps with typical values.
- The dataset spans multiple months across different seasons, which helps the model learn seasonal patterns.

## Key Numbers at a Glance

| | Departure Hour | Month | Day of Week | Distance (miles) | High Temp (°C) | Precipitation (mm) |
|---|---|---|---|---|---|---|
| Average | 1:26 PM | May–June | Wednesday | 804 | 22°C | 2mm |
| Shortest / Earliest | 12:03 AM | January | Sunday | 31 mi | -32°C | 0mm |
| Longest / Latest | 11:59 PM | October | Saturday | 5,095 mi | 51°C | 208mm |

A few things these numbers reveal:

- The average flight departs early-to-mid afternoon, though flights run across all 24 hours.
- Most flights are relatively short hops — the median distance is around 650 miles — but a long tail of cross-country routes stretches up to 5,000+ miles.
- Weather varies enormously across the dataset, from freezing winter days to sweltering summer ones.

## How Often Do Delays Actually Happen?

**About 1 in 5 flights in our dataset is delayed** — 20.6% to be precise, versus 79.4% that arrive on time or with only a small delay.

This matters a lot for how we measure success. If a model just guessed "on time" for every single flight, it would be right 80% of the time — but it would never catch a single delay. That's not useful. So instead of measuring simple accuracy, we evaluate how well the model catches real delays without crying wolf too often on flights that would have been fine.

## When Are Delays Most Likely?

The time you're scheduled to depart is one of the strongest predictors of whether you'll be delayed. Early morning flights — around 6 AM — have the lowest delay rates, around 10–12%. But delay risk climbs steadily through the day, peaking near **7 PM at about 29%**.

Why? Late-day flights inherit problems from earlier in the day. If an aircraft is running behind at 2 PM, that ripple effect shows up in its 5 PM, 7 PM, and 9 PM legs too. Airports and airlines also get more congested as the day goes on.

This pattern was clear enough that we treated time of day as one of our most important inputs — and we encoded it carefully to make sure the model understands that 11 PM and midnight are close together, not far apart.

## Does the Airline Matter?

Yes — significantly. We looked at the 12 busiest carriers in the dataset and found a spread of about **15 percentage points** between the most and least punctual airlines. The best performers run close to a 10–15% delay rate; the worst are closer to 25–30%.

This isn't random noise. It reflects real operational differences: how aggressively airlines schedule their aircraft, how much buffer time they build into their schedules, and how quickly they recover from disruptions. Knowing which airline you're flying with meaningfully shifts your delay risk.

## Do Features Overlap Too Much With Each Other?

Before building the model, we checked whether any of our inputs were essentially telling the model the same thing twice. For example, maximum and minimum daily temperature are often closely related — on a hot day, both go up together.

We found no problematic overlaps that needed to be removed. A couple of weather variables track closely, so our pipeline automatically filters those out during training to keep the model from over-weighting redundant information.

## What About Unusual Flights and Extreme Weather?

Every dataset has edge cases — unusually long routes, record-breaking weather days. We identified these by looking at how far each value falls from the typical range.

- About **5.6% of flights** fall outside the typical distance range (primarily very long cross-country routes).
- About **1.2% of weather readings** reflect genuinely extreme temperature days.
- **Departure time** had no outliers — flights are scheduled throughout the day in a predictable spread.

We chose to keep these extreme values in the model rather than removing them. Long flights and harsh weather are real scenarios that travelers face, and a model that's never seen them would give unreliable predictions for exactly the cases where good guidance matters most.

## What This All Means

The data exploration confirmed a clear story: **delay risk is shaped by time, airline, and route — not by chance**. The patterns are consistent and predictable enough that a model trained on historical data can meaningfully estimate risk for a new flight. Weather adds further signal, especially for flights at airports prone to snow, ice, or heavy rain.

These findings shaped every decision we made in building the model: which features to include, how to handle missing weather data, and how to measure whether our predictions are actually useful.
