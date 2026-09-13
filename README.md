# Volatility Forecasting & Risk-Aware Portfolio Allocation

This project tries to predict how *risky* a stock is going to be (its
volatility), instead of trying to predict whether the price will go up or
down. Then it uses those risk predictions to decide how to split money
across a few different stocks. I picked this angle on purpose because I
want to go into risk/portfolio management, and this project is basically
a small, hands-on version of the kind of thing that field actually cares
about.

## Why volatility and not just "will the stock go up"?

Predicting whether a stock's price goes up or down tomorrow is really,
really hard, almost to the point of basically being random (this is
related to something called the efficient market hypothesis, which
roughly says stock prices already reflect all the info people have, so
there's not much "free" pattern left to find). So instead of trying to
predict price direction, I predict **volatility** — basically, "how much
is this stock about to bounce around," not "which way will it go."
Volatility turns out to be a lot easier to predict than direction,
because calm periods tend to stay calm and wild periods tend to stay
wild. This is called "volatility clustering," and there are already old
formulas from statistics (GARCH and EWMA) that are pretty good at
capturing it.

So the real question I was trying to answer wasn't "can a machine
learning model predict volatility at all" (yes, kind of trivially, since
volatility is predictable in general). The real question was: **can my
LSTM (a type of neural network) actually beat those older, simpler
formulas, and can I trust its risk estimates enough to actually use them?**
I tried to answer that honestly, even in the parts where the answer ended
up being "not really" or "only sometimes."

## What's in each file

```
data_pipeline.py     # loads stock price data (real data from yfinance, or fake/synthetic
                      #   data if you don't have internet) and builds the input features
baseline_models.py   # the two "old school" volatility formulas: EWMA and GARCH
ml_model.py          # my LSTM model - one version predicts volatility, another
                      #   version predicts a "worst case" return (used for VaR, see below)
backtest.py          # takes the predictions and turns them into a portfolio,
                      #   then simulates how that portfolio would have done
evaluate.py          # checks how good the predictions actually were
main.py              # runs everything above, start to finish
```

## How to set it up

```bash
pip install torch arch scikit-learn matplotlib yfinance pandas numpy
```

## How to run it

```bash
# Quick test with made-up data, no internet needed
python main.py --demo --tickers AAPL MSFT JPM

# Real run using actual stock data (needs internet)
python main.py --tickers AAPL MSFT JPM XOM PG --start 2015-01-01 --end 2024-01-01
```

This saves a chart (`results.png`) and a CSV file
(`forecast_leaderboard.csv`) comparing how accurate each model was.

![Training curve and portfolio comparison](results.png)

## A quick explanation of what's actually going on

**There are two different things being predicted, and that's on purpose:**
- One target is the stock's volatility over the next few weeks. This is
  what gets compared against the GARCH/EWMA formulas.
- The other target is literally tomorrow's return, and this is used to
  build something called VaR (Value at Risk) — basically an estimate of
  "how bad could tomorrow realistically get." I originally mixed these
  two up early on (used the wrong one for VaR) and had to go back and fix
  it, since they're not actually the same thing — predicting "volatility
  will be low" isn't the same as predicting "there's only a 5% chance of
  a big loss tomorrow."

**I made sure not to let the model "cheat" by seeing the future.** When
you're working with time-based data like stock prices, it's really easy
to accidentally let your model peek at information from the future during
training, which makes your results look way better than they actually
are. I split everything so training data always comes strictly before
test data, with no shuffling.

**I checked if my "risk warning" was actually trustworthy.** If a model
says "there's only a 5% chance of a bad day tomorrow," that claim should
actually be true about 5% of the time if you check it against real
history. I built a check for this (`evaluate.var_calibration`) instead of
just assuming the model's confidence numbers meant anything. Sometimes it
was accurate, sometimes it wasn't — I tried to report that honestly
instead of hiding the times it didn't work.

**I built an actual portfolio out of the predictions, with realistic
friction.** Once I have a risk prediction, I use it to decide how much
money to put into each stock (put less money into whatever looks
riskier). I also added trading costs and made sure you can't trade using
information from the same day it becomes available (that would be
unrealistic - you'd need at least a day to act on it).

## What I actually found when I ran it on real data

**Setup:** I used 5 big US banks (JPM, BAC, WFC, GS, MS), pulling data
from 2018 to 2024. I tested the models on data from March 2022 to
November 2023 (data they hadn't seen during training).

**Which model predicted volatility best:** My LSTM won pretty clearly. It
beat both GARCH and EWMA on accuracy for all 5 stocks, which was
consistent across multiple runs.

**Was the "risk warning" (VaR) actually trustworthy:** Kind of hit or
miss, honestly. In one run, only 1 of the 5 banks had a well-calibrated
VaR estimate — the other 4 were too cautious, meaning the model said "5%
chance of a bad day" when the real chance was closer to 1-3%. I think
this happened because I made the model simpler to stop it from
overfitting (more on that below), which may have also made it too
conservative. I didn't fully solve this, and I'm being upfront about
that instead of pretending it's not an issue.

**Did the better predictions actually make a better portfolio? No, and
this was the most interesting part.** Even though my LSTM was the best
at predicting volatility, using those predictions to build a portfolio
actually did *worse* than just splitting the money equally across all 5
banks with no fancy math at all. At first I assumed this was because my
strategy (called "risk parity," which shifts money away from whatever
looks risky) must be failing specifically during the 2023 banking crisis,
since that's when banks are correlated and crash together. So I checked
that directly by breaking the results into three time chunks:

| Time period | My LSTM | GARCH | EWMA | Just split evenly |
|---|---|---|---|---|
| 2022 rate-hike crash | -1.01 | -1.08 | -1.10 | -1.16 |
| 2023 banking crisis | -2.20 | -2.16 | -2.24 | -2.17 |
| Recovery after | 0.81 | 0.82 | 0.86 | **1.68** |

(These numbers are Sharpe ratios — basically, return per unit of risk.
Higher is better.)

Turns out my first guess was wrong. During the actual crisis, every
single strategy did about equally badly — so risk-parity wasn't uniquely
failing there. Where the real gap showed up was **afterward, during the
recovery.** Splitting money evenly did way better during the bounce-back
(+34% annualized) than any of the "smart" strategies (+16-17%).

**Why that happens, once I thought about it more:** risk-parity keeps
pulling money away from whatever looks the riskiest at that moment. But
right after a crash, the stocks that look "riskiest" are usually the same
ones that crashed the hardest — and those are often exactly the stocks
that bounce back the hardest too. So risk-parity was basically punishing
itself by staying underweight in the stocks that ended up leading the
recovery.

**My takeaway:** having a more accurate volatility forecast didn't
actually translate into a better investment outcome here. The problem
wasn't that risk-parity handled the crash badly (everyone did badly then)
— it's that risk-parity's own rule (avoid what's risky) works against you
once the market starts recovering. If I kept developing this, the fix I'd
try first is having the strategy "loosen up" its risk-avoidance once a
crisis looks like it's ending, instead of applying the same rule no
matter what's going on in the market.

## Things I know are limitations (I'd rather say this myself than have someone else point it out)

- The GARCH model only re-trains itself every 21 days instead of every
  single day, mostly because re-training it daily is slow. A more
  "for real" version would probably do it daily.
- The way I calculate correlations between stocks for the min-variance
  version (an alternative to risk-parity, also in `backtest.py`) is a
  simplified version. Real risk teams usually use fancier techniques.
- The fake/synthetic data mode is only meant for quickly testing that the
  code runs — it's not real market behavior, so I only trust the results
  I got from actual real data.
- I only tested this on one time window, and a 5% "bad day" doesn't
  happen very often, so my calibration numbers are based on a fairly
  small sample. I wouldn't treat them as super precise, more like a
  rough signal.
- My results aren't perfectly identical every time I run this, since the
  neural network starts from random values each time. The overall
  pattern (LSTM most accurate, equal-split winning on real portfolio
  performance) has stayed consistent, but exact numbers shift a bit
  between runs.

## What I'd add if I kept working on this

- Add more types of input features (like sector info or momentum) instead
  of only looking at each stock completely on its own.
- Try fancier versions of GARCH that handle the fact that volatility often
  reacts differently to a stock going down vs going up.
- Instead of one VaR number, predict a handful of different risk levels
  at once to get a fuller picture of what could happen.
- Test this same idea across other rough patches in market history (like
  2020's COVID crash) to see if the "recovery lag" pattern I found shows
  up again, or if it was specific to this one banking crisis.
