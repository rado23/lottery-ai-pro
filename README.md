# 🎯 Lottery AI Prediction API

This Flask-based API provides endpoints for lottery number prediction, trend analysis, and strategy building across multiple UK lottery games.

---

## 🎮 Supported Games

- EuroMillions
- Thunderball
- Lotto
- Set for Life

---

## 📡 API Endpoints

### ✅ Predictions

#### 🔮 Heuristic Predictions
```
GET /predict/<game>
```
- Description: Returns 10 statistically inferred prediction sets.
- Example: `/predict/euromillions`
- Response:
```json
{
  "heuristic": [
    {
      "main_numbers": [10, 17, 27, 35, 49],
      "lucky_stars": [2, 6]
    },
    ...
  ]
}
```

#### 🧠 ML Predictions
```
GET /predict/<game>-ml
```
- Description: Uses machine learning models to return a single most likely prediction.
- Example: `/predict/thunderball-ml`
- Response:
```json
{
  "ml": {
    "main_numbers": [1, 12, 21, 28, 33],
    "thunderball": 5
  }
}
```

---

### 📊 Trends

```
GET /trends/<game>
```
- Description: Shows frequency, heatmap, intervals, and hot/cold number stats.
- Example: `/trends/setforlife`
- Response:
```json
{
  "trends": {
    "hot_numbers": {...},
    "cold_numbers": {...},
    "frequency": {...},
    "intervals": {...},
    "heatmap": {...},
    "draw_count": 100
  }
}
```

Optional query:
- `draws=<int>` to customize window (e.g., `/trends/lotto?draws=50`)

---

### 🧠 Strategy Builder

```
POST /strategy
```
- Description: Given a fund and game preferences, returns an optimized play strategy.
- Request Body:
```json
{
  "funds": 50,
  "focus": "highest_win",
  "max_draws": 4,
  "games": ["euromillions", "thunderball"]
}
```
- Response:
```json
{
  "strategy": [
    {
      "game": "euromillions",
      "tickets": 2,
      "per_draw": 1,
      "draws": 2,
      "cost": 5
    },
    ...
  ],
  "total_cost": 50
}
```

---

## 🚀 Running Locally

```bash
PYTHONPATH=. flask run
# or
python3 backend/app.py
```

Make sure the following folders exist: `data/`, `models/`.

---

## 🧪 Development

- Python 3.9+
- Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 📬 Author

@Rado