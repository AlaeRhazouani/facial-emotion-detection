# Data

## Download FER2013

1. Go to https://www.kaggle.com/datasets/msambare/fer2013
2. Download `fer2013.csv`
3. Place it in `data/raw/fer2013.csv`

> Training is done on Kaggle. The local path is for development only.

## Structure

| Column | Description |
|--------|-------------|
| `emotion` | Integer label 0–6 |
| `pixels` | Space-separated pixel values (48×48 = 2304 values) |
| `Usage` | `Training`, `PublicTest`, or `PrivateTest` |

## Emotion labels

| Index | Emotion |
|-------|---------|
| 0 | Angry |
| 1 | Disgust |
| 2 | Fear |
| 3 | Happy |
| 4 | Sad |
| 5 | Surprise |
| 6 | Neutral |

## Important

`data/raw/` and `data/processed/` are gitignored — never commit raw data.