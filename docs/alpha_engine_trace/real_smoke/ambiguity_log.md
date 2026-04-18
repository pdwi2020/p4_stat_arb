# P4 Ambiguity Log

- Survivorship: v1 uses current S&P 500 constituents, not historical constituent membership.
- Universe mixing: equity-ETF pairs are limited to sector-matched relationships; broad-market ETFs are mainly used in ETF-only families.
- Basket scope: baskets are restricted to 3 assets and generated from prefiltered pairs rather than exhaustive triplet enumeration.
- Multiple-testing bootstrap: the family-wise bootstrap uses a common block length derived from the median `2 x half_life`, clipped to `[5, 60]`.
