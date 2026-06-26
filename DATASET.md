# Dataset

The dataset for this challenge is published by HackerRank in the official
starter repository:

> https://github.com/interviewstreet/hackerrank-orchestrate-june26

It contains:
- `dataset/claims.csv` — 44 unlabeled claims to run your system on
- `dataset/sample_claims.csv` — 20 labeled claims for evaluation
- `dataset/user_history.csv` — historical claim counts and risk context
- `dataset/evidence_requirements.csv` — minimum image evidence rules
- `dataset/images/` — 82 images (sample/ and test/ subfolders)

The full task description and 14-column output schema are in:
> https://github.com/interviewstreet/hackerrank-orchestrate-june26/blob/main/problem_statement.md

## Setup

Clone the official repo alongside this one and link its `dataset/` folder:

```bash
git clone https://github.com/interviewstreet/hackerrank-orchestrate-june26 hr-starter
ln -s $(pwd)/hr-starter/dataset ./dataset        # macOS / Linux
# Windows: xcopy hr-starter\dataset\ dataset\ /E /I
```

Then follow the Quick start in [README.md](README.md).
