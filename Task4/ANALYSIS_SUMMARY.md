# Task4: NeurIPS-Clinical Trials Cross-Reference Analysis

## Executive Summary

This analysis identifies connections between clinical trial sponsors (pharmaceutical companies) and AI research through NeurIPS conference author affiliations.

### Key Findings

- **66 unique author-sponsor matches** found across 17 clinical trial sponsors
- **25 authors (38%)** had company affiliations overlapping with clinical trial years (2008-2021)
- **4,374 total clinical trials** linked to matched sponsors

## Methodology

### Data Sources
1. **NeurIPS Papers**: 13,405 papers (2008-2022) with 24,125 unique authors
2. **Clinical Trials**: 9,428 trials with 691 unique sponsors
3. **OpenAlex API**: Author affiliation data with employment history

### Matching Process
1. Direct text matching of sponsor names in paper titles/abstracts (156 matches, mostly tech companies)
2. Author-based matching via OpenAlex company affiliations (66 pharma matches)

## Results by Sponsor

| Sponsor | Authors | NeurIPS Papers | Clinical Trials | Overlap |
|---------|---------|----------------|-----------------|---------|
| Novartis | 8 | 133 | 53 | 0 |
| Pfizer | 8 | 31 | 721 | 5 |
| Bristol-Myers Squibb | 7 | 60 | 221 | 4 |
| Amgen | 6 | 92 | 194 | 4 |
| Merck (ArQule) | 5 | 16 | 7 | 1 |
| Gilead Sciences | 4 | 48 | 158 | 0 |
| AstraZeneca | 4 | 9 | 602 | 2 |
| Sanofi | 4 | 21 | 280 | 0 |
| GlaxoSmithKline | 4 | 27 | 543 | 2 |
| Hoffmann-La Roche | 3 | 4 | 443 | 1 |
| Regeneron | 2 | 15 | 60 | 2 |
| Eli Lilly | 2 | 3 | 562 | 2 |
| Biogen | 2 | 23 | 118 | 1 |
| Bayer | 2 | 17 | 258 | 1 |
| Allergan | 2 | 3 | 86 | 0 |
| Vertex | 2 | 12 | 52 | 0 |
| Novo Nordisk | 1 | 3 | 176 | 0 |

## Top Authors by NeurIPS Publication Count

| Author | Papers | Sponsor | Affiliation Years |
|--------|--------|---------|-------------------|
| Andreas Krause | 64 | Novartis | 2005-2007 |
| Ping Li | 39 | Novartis | 2002 |
| Qiang Liu | 35 | Amgen | 1999 |
| Wei Chen | 25 | Bristol-Myers Squibb | 2018 |
| Matthias Hein | 24 | Gilead Sciences | 2024 |
| Xi Chen | 23 | Amgen | 2021 |
| Tommi Jaakkola | 22 | Amgen/Novartis | 2019 |
| Devavrat Shah | 21 | Biogen | 2025 |
| Chunhua Shen | 18 | Bristol-Myers Squibb | 2020 |
| Klaus-Robert Muller | 16 | Bayer | 2007-2009 |

## Output Files

1. **author_sponsor_matches.csv** - 66 author-sponsor pairs with OpenAlex IDs
2. **author_trial_cross_reference.csv** - Detailed author-trial linkage data
3. **firm_year_panel.csv** - Panel data (214 sponsor-year observations)
4. **NeurIPS_Clinical_Trials.csv** - Direct text matches for database upload

## Observations

1. **Tech companies dominate direct mentions**: Google (73), Microsoft (31), Amazon (30), NVIDIA (16)
2. **Pharma connections are indirect**: Found via author affiliations, not direct paper mentions
3. **Many affiliations are brief**: Most are 1-year consulting/advisory roles
4. **Time lag exists**: Some affiliations occurred before/after clinical trial periods
5. **High-profile ML researchers**: Several well-known ML researchers have had pharma affiliations

## Database Upload

The following SQL creates the results table (requires admin privileges):

```sql
CREATE TABLE NeurIPS_Clinical_Trials (
    match_id INT AUTO_INCREMENT PRIMARY KEY,
    conference_name VARCHAR(50),
    year INT,
    paper_title TEXT,
    abstract TEXT,
    authors JSON,
    paper_url VARCHAR(500),
    matched_firm VARCHAR(255),
    match_location VARCHAR(50),
    matched_text TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---
Generated: 2026-03-25
