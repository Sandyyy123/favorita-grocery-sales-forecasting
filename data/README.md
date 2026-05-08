# Data - Store Sales Time Series Forecasting (Corporacion Favorita)

## Kaggle competition

Slug: `store-sales-time-series-forecasting`
URL: https://www.kaggle.com/competitions/store-sales-time-series-forecasting

This is a Kaggle competition dataset and requires accepting the competition rules on the Kaggle website at least once for the account whose API token is in use. Total compressed size is roughly 30 MB.

## Download command

```bash
cd /root/AI/liora_projects/09_favorita_sales/data
kaggle competitions download -c store-sales-time-series-forecasting
unzip -o store-sales-time-series-forecasting.zip
rm store-sales-time-series-forecasting.zip
```

If `kaggle` is not on PATH, install it once with `pip install --user kaggle` and ensure `~/.kaggle/kaggle.json` exists with mode `600`. If the download fails with `403 Forbidden`, log into kaggle.com once in a browser, accept the competition rules at the URL above, then re-run the command.

Alternative manual route: download the ZIP from the competition page, drop it into this folder, and run the unzip step above.

## Files (after extraction)

| File | Rows (approx) | Purpose |
|------|---------------|---------|
| `train.csv` | 3,000,888 | Daily unit sales per store-family from 2013-01-01 to 2017-08-15. Columns: `id`, `date`, `store_nbr`, `family`, `sales`, `onpromotion`. |
| `test.csv` | 28,512 | Forecast horizon 2017-08-16 to 2017-08-31 (16 days x 54 stores x 33 families). Columns same as train minus `sales`. |
| `stores.csv` | 54 | Store metadata: `store_nbr`, `city`, `state`, `type`, `cluster`. |
| `oil.csv` | 1,218 | Daily WTI oil price (`dcoilwtico`). Some weekday gaps; forward-fill or interpolate. |
| `holidays_events.csv` | 350 | National, regional, and local holidays. Columns: `date`, `type`, `locale`, `locale_name`, `description`, `transferred`. |
| `transactions.csv` | 83,488 | Daily transaction count per store (proxy for footfall). Sparser than train (many missing store-days). |
| `sample_submission.csv` | 28,512 | Submission template. |

## Schema notes for downstream code

- Parse `date` columns with `pd.to_datetime`. The training horizon ends 2017-08-15 inclusive; the public test set covers 2017-08-16 to 2017-08-31 inclusive (16 days).
- `family` has 33 distinct categorical values. Use one-hot or target encoding; LightGBM handles native categoricals via `pd.Categorical`.
- `onpromotion` is integer count of items on promotion in that store-family-day.
- `holidays_events.transferred = True` means the legal holiday was moved to a different date and the original date is a working day. Build the effective-holiday calendar by filtering out transferred rows and adding `Transfer`-typed rows.
- The April 2016 earthquake (Pedernales, Ecuador) disrupts April-June 2016 transactions in many stores. Several Kaggle winners flag this with a binary `is_post_quake` window; the advanced model in this scaffold sets up the column but does not tune the window.

## Citation

Alexis Cook, DanB, inversion, Ryan Holbrook. Store Sales - Time Series Forecasting. Kaggle competition. 2021. https://kaggle.com/competitions/store-sales-time-series-forecasting

## Provenance

Corporacion Favorita is the largest grocery retailer in Ecuador. The data is anonymised store-day-family unit sales released by the company under the Kaggle competition rules. No PII.
