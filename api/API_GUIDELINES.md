# API Guidelines

This folder contains three API modules for frontend integration:

- `demo_api.py`: run pipeline stages for demo workflow.
- `tweet_api.py`: score a user-provided tweet.
- `company_api.py`: collect Reddit data for a company and return a final score.

## 1) Demo API (`demo_api.py`)

Purpose: Trigger Stage 1A through Stage 3A from a demo button.

### Routes

- `GET /`
  - Health check for this service.

- `POST /api/demo/run`
  - Starts background execution of pipeline stages.
  - Request body:
    ```json
    {
      "include_granger": true
    }
    ```
  - Runs:
    - `1A`, `1B`, `1C`, `2A`, `2C`, `3A`
    - plus `2B` when `include_granger=true`
  - Response includes `job_id`.

- `GET /api/demo/run/{job_id}`
  - Poll job status and latest logs.
  - Returns `queued`, `running`, `completed`, or `failed`.

- `DELETE /api/demo/run/{job_id}`
  - Removes stored job metadata and logs from memory.

### Frontend call flow

1. User clicks "Run Demo".
2. Frontend calls `POST /api/demo/run`.
3. Frontend stores `job_id`.
4. Frontend polls `GET /api/demo/run/{job_id}` every 2-5 seconds.
5. When status is `completed` or `failed`, stop polling and render logs/results.

## 2) Tweet API (`tweet_api.py`)

Purpose: Score one tweet and return main score plus component scores.

### Models used

- FinBERT engine from existing pipeline (`src/sentiment/finbert.py`).
- `models/v4_full_model.pkl` for final market-direction probability.
- `models/scaler.pkl` for feature scaling.

### Route

- `POST /api/tweet/score`
  - Request body:
    ```json
    {
      "tweet": "NVIDIA earnings look very strong this quarter",
      "ticker": "NVDA",
      "user_credibility": 1.0
    }
    ```
  - Returns:
    - `main_score` (v4 UP probability)
    - `predicted_direction`
    - FinBERT outputs (`label`, `confidence`, class scores)
    - `weighted_sentiment_score`

### Internal scoring flow

1. Clean tweet text.
2. Run FinBERT inference for sentiment label/confidence.
3. Build sentiment features (single-text proxy values).
4. Inject these into a latest feature template from output CSVs.
5. Run `v4_full_model.pkl` inference to compute final probability.

## 3) Company API (`company_api.py`)

Purpose: User provides company name, API collects Reddit data, then computes final score.

### Route

- `POST /api/company/score`
  - Request body:
    ```json
    {
      "company_name": "Reliance",
      "posts_per_query": 30,
      "fetch_comments": true
    }
    ```

### Internal flow

1. Resolve company name to ticker using `ticker_reddit_collector.resolve_ticker`.
2. Collect Reddit posts/comments using `collect_for_ticker`.
3. Clean text and run FinBERT on collected messages.
4. Aggregate sentiment metrics across all fetched posts.
5. Run final score using `models/v4_full_model.pkl`.
6. Return:
  - `main_score`
  - `predicted_direction`
  - distribution of FinBERT labels
  - mention volume and average confidence

## Running each API module

From project root:

```bash
uvicorn api.demo_api:app --host 0.0.0.0 --port 8010 --reload
uvicorn api.tweet_api:app --host 0.0.0.0 --port 8011 --reload
uvicorn api.company_api:app --host 0.0.0.0 --port 8012 --reload
```

## Suggested frontend wiring order

1. Build demo page integration first with `demo_api.py` polling flow.
2. Add tweet scoring form with `tweet_api.py`.
3. Add company scoring form with `company_api.py`.
4. Normalize all score cards to display:
  - main score
  - predicted direction
  - confidence/supporting signals

## Notes

- `company_api.py` depends on live Reddit HTTP calls and can be slow.
- `tweet_api.py` and `company_api.py` rely on `models/v4_full_model.pkl` and `models/scaler.pkl` being present.
- The v4 score is returned as UP probability in range [0, 1].
