# Features

All features use only transfers **before the cutoff T**.
Amounts are converted to **US dollars**.

## A. Account only (19 features)

Each account's own behaviour. No network, no labels.

| Feature | What it means | Why it may matter |
|---|---|---|
| `n_sent` | Number of transfers the account sent | Very active accounts can be used to move money |
| `sum_sent_usd` | Total amount sent | Large amounts going out |
| `max_sent_usd` | Largest single transfer sent | One unusually large transfer |
| `median_sent_usd` | Typical size of a sent transfer | Laundering often uses similar, round amounts |
| `n_recv` | Number of transfers the account received | Many incoming transfers can mean collecting money |
| `sum_recv_usd` | Total amount received | Large amounts coming in |
| `max_recv_usd` | Largest single transfer received | One unusually large transfer |
| `median_recv_usd` | Typical size of a received transfer | Same as above, for incoming money |
| `pass_through` | How close money in and money out are (0 to 1). 1 = the same amount goes in and out | Mules receive money and send it on quickly |
| `share_sent_ach` | Share of sent transfers paid by ACH (bank transfer) | Some payment types are used more for laundering |
| `share_sent_bitcoin` | Share paid in Bitcoin | Harder to trace |
| `share_sent_cash` | Share paid in cash | Harder to trace |
| `share_sent_cheque` | Share paid by cheque | Payment type pattern |
| `share_sent_credit_card` | Share paid by credit card | Payment type pattern |
| `share_sent_reinvestment` | Share that is reinvestment (often to the same account) | Normal savings behaviour |
| `share_sent_wire` | Share paid by wire transfer | Fast, often cross-border |
| `n_currencies_sent` | Number of different currencies the account paid in | Using many currencies can hide the trail |
| `share_cross_currency` | Share of sent transfers where the paid and received currencies differ | Currency exchange is a common laundering step |
| `active_days` | Number of days with at least one transfer | Short bursts of activity can be suspicious |

## B. Network structure (5 features)

The account's position in the transfer network. No labels used.

| Feature | What it means | Why it may matter |
|---|---|---|
| `fan_out` | Number of different accounts it sent money to | Spreading money to many accounts |
| `fan_in` | Number of different accounts it received money from | Collecting money from many accounts |
| `n_counterparties` | Number of different accounts it is connected to, in either direction | Size of its network |
| `avg_counterparty_degree` | Average number of connections of its counterparties | Connected to hubs, or to small isolated accounts |
| `pagerank` | How central the account is: high when it receives from accounts that also receive a lot. 1 = average | Money tends to flow into central accounts |

## C. Closeness to known cases (6 features)

"Known" = accounts already involved in laundering before T (already found).

| Feature | What it means | Why it may matter |
|---|---|---|
| `n_known_counterparties` | Number of direct counterparties that are known cases | Mules often work with other mules |
| `share_known_counterparties` | Share of its counterparties that are known cases | Same, adjusted for network size |
| `n_counterparties_near_known` | Number of counterparties that are connected to a known case (two steps away) | Being close to a laundering group |
| `usd_from_known` | Total amount received from known cases | Money coming from a known laundering account |
| `usd_to_known` | Total amount sent to known cases | Money going to a known laundering account |
| `hops_to_known` | Steps to the nearest known case. 1 = direct counterparty, 7 = not connected within 6 steps | The closer, the higher the risk |

## How they are built (in one line each)

- **A**: group the transfers by sender or receiver, then count, sum and take shares.
- **B**: build a table of "who sent money to whom", then count rows and columns, and run PageRank.
- **C**: multiply that table by a list of known cases (1 = known, 0 = not). One multiplication = one step away, two = two steps away. `hops_to_known` uses a breadth-first search from all known cases.
