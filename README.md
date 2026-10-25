# Graph features for money laundering detection

Money mules and laundering accounts rarely act alone: money moves through chains of accounts.
This project tests how much the **transfer network** helps to find new laundering accounts,
compared with looking at each account alone.

## Setup
- **Data**: [IBM Transactions for Anti Money Laundering](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml) (synthetic bank transfers), `HI-Small_Trans.csv`.
- **Time split**: features use only transfers before a cutoff T. Accounts already involved in laundering before T are "known".
  The target is accounts that are **not known yet** and get involved in laundering **after T**: *who should analysts look at next?*
- **Three models** (LightGBM), same accounts:
  - A. account only: amounts, counts, payment types, pass-through
  - B. + network structure: counterparties, fan-in / fan-out, PageRank
  - C. + closeness to known cases: known counterparties, money from / to known accounts, steps to the nearest known account
- **EBM** to see how the network features change the risk.

## Results
_To be added after running the notebook._

## How to run
```bash
pip install -r requirements.txt
# put HI-Small_Trans.csv in data/raw/ibm_aml/
cd notebooks
jupyter notebook 01_graph_features_aml.ipynb
```
Without the data file, the notebook runs on a small synthetic file (only to check the code).

## Files
- `src/graph_utils.py`: loading, USD conversion, time split, account / network features, evaluation
- `notebooks/01_graph_features_aml.ipynb`: the analysis
- `make_nb.py`: builds the notebook

## Limitations
- Synthetic data. Real networks are messier, and one bank sees only part of the network.
- Only accounts active before T are scored.
- "Known" cases are only the ones that were found. Some laundering is never found (a PU learning problem).
