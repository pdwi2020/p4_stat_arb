# P4 Future Work

## Data Upgrades

- Replace current-constituent S&P 500 membership with point-in-time historical constituent data to remove the biggest survivorship-bias source.
- Expand the ETF taxonomy beyond the fixed v1 list and support richer sector/style/theme mappings.
- Add stronger cache provenance metadata so public-source refreshes are easier to audit.

## Research Extensions

- Explore broader candidate families only after preserving the full tested-strategy family for deflation reporting.
- Add richer bootstrap variants and alternative dependence assumptions for White RC and SPA sensitivity checks.
- Compare alternative OU estimation choices and spread-normalization schemes on the same walk-forward splits.

## Execution and Capacity

- Add more granular cost stress tests and alternative borrow assumptions.
- Separate signal quality from capacity failure by reporting pre-capacity and post-capacity portfolio stats side by side.

## Documentation

- Add a persistent fixture-demo config if a stable basket-heavy offline example is needed outside the pytest integration path.
