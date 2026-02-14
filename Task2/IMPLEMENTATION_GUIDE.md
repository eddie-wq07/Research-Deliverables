# Task #2 Implementation Guide
## Merging PatentsView, DISCERN, and Clinical Trials

**Author:** Edward Jung
**Date:** February 14, 2026
**Deliverable:** Research Assistant Task #2

---

## Table of Contents

1. [Overview](#overview)
2. [Data Architecture](#data-architecture)
3. [Memory Efficiency Best Practices](#memory-efficiency-best-practices)
4. [AI Classification Strategy](#ai-classification-strategy)
5. [DISCERN 2 Integration](#discern-2-integration)
6. [Troubleshooting](#troubleshooting)
7. [Validation Checklist](#validation-checklist)
8. [References](#references)

---

## Overview

### Objective
Construct a firm-year dataset of AI-related patent applications for biopharma firms conducting clinical trials.

### Key Requirements
- **Data Source:** PatentsView bulk downloads
- **Focus:** Patent APPLICATIONS (not just granted patents)
- **Time Period:** 2000-2025
- **Firm Identification:** Map to GVKEY using DISCERN 2
- **AI Classification:** CPC codes + keyword filtering
- **Output:** Firm-year dataset with AI patent metrics

### Deliverables
1. `task2_patents_discern_merge.ipynb` - Main analysis notebook
2. `patent_level_dataset.csv` - Patent-level data with AI flags
3. `firm_year_patents.csv` - Firm-year aggregated patent metrics
4. `firm_year_merged.csv` - Patents + clinical trials combined

---

## Data Architecture

### Two-Layer Design

#### Layer 1: Patent-Level Dataset
**Purpose:** Intermediate dataset for validation and flexibility

**Structure:**
```
One row per patent application

Columns:
- application_id: Unique identifier
- patent_id: Patent ID if granted (NULL if pending)
- filing_date: Application filing date
- filing_year: Year extracted from filing_date
- gvkey: Firm identifier (from DISCERN 2)
- applicant_organization: Raw applicant name
- is_ai: Binary AI flag (1 = AI-related)
- ai_method: Classification method ('cpc', 'keyword', 'both')
- ai_cpc_codes: AI-related CPC codes found
- ai_keywords: Keywords matched
- title: Patent title
- abstract: Patent abstract
```

**Advantages:**
- Enables manual validation and spot-checking
- Flexible for adding new metrics
- Reproducible aggregation to firm-year level
- Supports patent-level analysis if needed

#### Layer 2: Firm-Year Dataset
**Purpose:** Final dataset for regression analysis

**Structure:**
```
One row per gvkey-year combination

Columns:
- gvkey: Firm identifier
- year: Calendar year
- total_applications: Count of all applications
- ai_applications: Count of AI-related applications
- ai_share: ai_applications / total_applications
- ai_dummy: 1 if ai_applications > 0, else 0
- num_trials: Count of clinical trials (from merge)
- avg_phase: Average trial phase (from merge)
```

**Advantages:**
- Ready for regression analysis
- Compact and efficient
- Standard panel data format
- Easy to merge with other firm-year datasets

---

## Memory Efficiency Best Practices

### Problem: Large PatentsView Files

**File Sizes:**
- `g_application.tsv`: ~2-3 GB
- `g_cpc_current.tsv`: ~4 GB
- `pg_applicant_not_disambiguated.tsv`: ~1-2 GB
- `g_patent_abstract.tsv`: ~6 GB

**Total uncompressed:** ~15 GB

### Strategy 1: Use DuckDB for Pre-Filtering

**Why DuckDB?**
- In-process database (no server setup)
- Handles files larger than RAM
- SQL interface for complex filtering
- Fast columnar storage

**Example:**
```python
import duckdb

con = duckdb.connect('patents.ddb')

# Import with automatic optimization
con.execute("""
    CREATE TABLE applications AS
    SELECT * FROM read_csv('g_application.tsv',
                           delim='\t',
                           all_varchar=true)
    WHERE filing_date BETWEEN '2000-01-01' AND '2025-12-31'
""")

# Filter BEFORE loading into pandas
df = con.execute("""
    SELECT * FROM applications
    WHERE filing_year >= 2000
    LIMIT 100000
""").df()
```

### Strategy 2: Chunked Reading

**When to use:** Files too large for DuckDB or need custom processing

```python
import pandas as pd

chunk_size = 100000
chunks = []

for chunk in pd.read_csv('large_file.tsv',
                         sep='\t',
                         chunksize=chunk_size):
    # Filter each chunk
    filtered = chunk[chunk['year'] >= 2000]
    chunks.append(filtered)

df = pd.concat(chunks, ignore_index=True)
```

### Strategy 3: Optimize Data Types

**Default pandas dtypes waste memory:**
- `object` dtype for strings (inefficient)
- `int64` for small integers (overkill)
- `float64` for simple metrics (excessive precision)

**Optimized dtypes:**
```python
dtypes = {
    'gvkey': 'category',           # Repetitive strings
    'year': 'int16',               # Small integers
    'is_ai': 'bool',               # Binary flags
    'ai_share': 'float32',         # Sufficient precision
    'application_id': 'string'     # Unique strings
}

df = pd.read_csv('data.csv', dtype=dtypes)
```

**Memory savings:** 50-70% reduction typical

### Strategy 4: Year-by-Year Processing

**For extremely large datasets:**

```python
years = range(2000, 2026)
yearly_results = []

for year in years:
    print(f"Processing {year}...")

    # Load only one year at a time
    year_data = con.execute(f"""
        SELECT * FROM applications
        WHERE filing_year = {year}
    """).df()

    # Process
    year_ai = classify_ai(year_data)
    yearly_results.append(year_ai)

    # Free memory
    del year_data

final_df = pd.concat(yearly_results)
```

### Strategy 5: Drop Columns Early

**Keep only what you need:**

```python
# BAD: Load everything
df = pd.read_csv('large_file.tsv', sep='\t')

# GOOD: Select columns upfront
needed_cols = ['application_id', 'filing_date', 'title']
df = pd.read_csv('large_file.tsv',
                 sep='\t',
                 usecols=needed_cols)
```

### Strategy 6: Use Generators for Text Processing

**For keyword search on abstracts:**

```python
def process_abstracts(filepath, keywords):
    """Generator to avoid loading all abstracts into memory."""
    for chunk in pd.read_csv(filepath, chunksize=10000):
        for idx, row in chunk.iterrows():
            if any(kw in row['abstract'].lower() for kw in keywords):
                yield row

# Use generator
ai_patents = list(process_abstracts('abstracts.tsv', AI_KEYWORDS))
```

### Memory Monitoring

```python
import psutil
import os

def print_memory_usage():
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 / 1024
    print(f"Memory usage: {mem_mb:.1f} MB")

# Check periodically
print_memory_usage()  # Before operation
df = load_large_file()
print_memory_usage()  # After operation
```

---

## AI Classification Strategy

### Two-Pronged Approach

#### Method 1: CPC Classification Codes (High Precision)

**AI-Related CPC Codes:**
```
G06N - Computing based on specific computational models
├── G06N3 - Neural networks
│   ├── G06N3/04 - Architecture (general)
│   ├── G06N3/08 - Learning methods
│   └── G06N3/09 - Supervised learning
├── G06N5 - Knowledge-based models
│   └── G06N5/01 - Expert systems
├── G06N7 - Probabilistic/fuzzy logic
│   └── G06N7/01 - Bayesian networks
├── G06N10 - Quantum computing
└── G06N20 - Machine learning
    ├── G06N20/00 - General ML
    └── G06N20/10 - Ensemble methods
```

**Advantages:**
- Examiner-assigned (authoritative)
- Standardized internationally
- High precision (~90-95%)
- No false positives from unrelated uses

**Limitations:**
- Only available for GRANTED patents
- Not available for pending applications
- May miss emerging AI applications
- Classification lag (1-2 years)

**Implementation:**
```python
AI_CPC_PATTERNS = [
    'G06N3',   # Neural networks
    'G06N5',   # Knowledge-based
    'G06N7',   # Probabilistic
    'G06N10',  # Quantum
    'G06N20',  # Machine learning
]

def is_ai_cpc(cpc_codes):
    """Check if any CPC code matches AI patterns."""
    if pd.isna(cpc_codes):
        return False
    for pattern in AI_CPC_PATTERNS:
        if any(code.startswith(pattern) for code in cpc_codes):
            return True
    return False
```

#### Method 2: Keyword-Based Filtering (High Recall)

**AI Keyword Categories:**

```python
AI_KEYWORDS = {
    # Core ML terms
    'core': [
        'machine learning', 'deep learning', 'neural network',
        'artificial intelligence', 'ai model', 'ml model'
    ],

    # Learning paradigms
    'learning': [
        'supervised learning', 'unsupervised learning',
        'reinforcement learning', 'semi-supervised',
        'transfer learning', 'meta-learning'
    ],

    # Models
    'models': [
        'random forest', 'gradient boosting',
        'support vector machine', 'decision tree',
        'bayesian network', 'lstm', 'transformer',
        'generative adversarial', 'gan'
    ],

    # Applications
    'applications': [
        'computer vision', 'natural language processing',
        'nlp', 'image recognition', 'speech recognition',
        'predictive model'
    ],

    # Techniques
    'techniques': [
        'feature extraction', 'dimensionality reduction',
        'classification algorithm', 'regression algorithm',
        'clustering algorithm', 'optimization algorithm'
    ]
}
```

**Advantages:**
- Works for pending applications
- Captures emerging terminology
- Higher recall
- Flexible and updatable

**Limitations:**
- May include false positives
- Requires careful keyword curation
- Context-dependent (e.g., "network" alone is ambiguous)
- Lower precision (~70-80%)

**Best Practices:**
1. Use multi-word phrases (not single words)
2. Search in both title AND abstract
3. Case-insensitive matching
4. Exclude obvious false positives

**Implementation:**
```python
def contains_ai_keywords(text, keywords):
    """Check if text contains AI keywords."""
    if pd.isna(text):
        return False, []

    text_lower = str(text).lower()
    matched = []

    for keyword in keywords:
        # Use word boundaries to avoid partial matches
        pattern = r'\b' + re.escape(keyword) + r'\b'
        if re.search(pattern, text_lower):
            matched.append(keyword)

    return len(matched) > 0, matched
```

### Hybrid Approach (Recommended)

**Combine both methods:**
```python
is_ai = (is_ai_cpc OR is_ai_keyword)

# Track which method detected AI
ai_method = {
    'cpc': is_ai_cpc and not is_ai_keyword,
    'keyword': is_ai_keyword and not is_ai_cpc,
    'both': is_ai_cpc and is_ai_keyword,
    None: not is_ai
}
```

**Interpretation:**
- **CPC only:** High confidence, granted patents
- **Keyword only:** Moderate confidence, may need validation
- **Both:** Very high confidence
- **Neither:** Not AI-related

### Validation Strategy

**Sample-based validation:**
1. Randomly sample 50-100 patents from each method
2. Manually review titles/abstracts
3. Calculate precision = true positives / total sampled
4. Refine keywords based on false positives

**Cross-validation:**
1. Compare to known AI patent portfolios (IBM, Google, etc.)
2. Check temporal trends (AI should increase over time)
3. Validate against external datasets (Google Patents, Lens.org)

---

## DISCERN 2 Integration

### Overview

**DISCERN 2** (Disambiguation and Integration System for Compustat-Enabled Research using Name-matching)

**Purpose:** Maps patent assignees to Compustat GVKEY

**Download:** https://zenodo.org/records/13619821

### Key Files

According to the DISCERN 2 data dictionary:

1. **assignee_gvkey_mapping.csv**
   - Links patent assignee IDs to GVKEY
   - Handles name variations and disambiguation

2. **time_varying_mapping.csv**
   - Tracks firm changes over time (M&A, spin-offs)
   - Important for long time series

3. **match_quality.csv**
   - Confidence scores for matches
   - Helps filter low-quality matches

### Integration Steps

#### Step 1: Download DISCERN 2

```bash
# Download from Zenodo
wget https://zenodo.org/records/13619821/files/DISCERN2.zip

# Extract
unzip DISCERN2.zip -d DISCERN2/
```

#### Step 2: Load Mapping Files

```python
import pandas as pd

# Load main mapping
discern_mapping = pd.read_csv('DISCERN2/assignee_gvkey_mapping.csv')

# Load time-varying mapping
discern_time = pd.read_csv('DISCERN2/time_varying_mapping.csv')

# Load match quality scores
match_quality = pd.read_csv('DISCERN2/match_quality.csv')
```

#### Step 3: Join with Patent Data

**Option A: Join on Assignee ID (if available)**
```python
# If PatentsView assignee_id is available
patents_with_gvkey = patents.merge(
    discern_mapping[['assignee_id', 'gvkey']],
    on='assignee_id',
    how='left'
)
```

**Option B: Fuzzy Name Matching**
```python
from rapidfuzz import fuzz, process

def fuzzy_match_gvkey(org_name, discern_names, threshold=85):
    """
    Match organization name to DISCERN database using fuzzy matching.

    Parameters:
    -----------
    org_name : str
        Organization name to match
    discern_names : dict
        Dictionary of {clean_name: gvkey}
    threshold : int
        Minimum similarity score (0-100)

    Returns:
    --------
    gvkey or None
    """
    if pd.isna(org_name):
        return None

    # Clean input name
    org_clean = clean_org_name(org_name)

    # Find best match
    match = process.extractOne(
        org_clean,
        discern_names.keys(),
        scorer=fuzz.ratio
    )

    if match and match[1] >= threshold:
        matched_name = match[0]
        return discern_names[matched_name]

    return None

# Create DISCERN lookup dictionary
discern_lookup = dict(zip(
    discern_mapping['clean_name'],
    discern_mapping['gvkey']
))

# Apply fuzzy matching
patents['gvkey'] = patents['applicant_organization'].apply(
    lambda x: fuzzy_match_gvkey(x, discern_lookup)
)
```

#### Step 4: Handle Time-Varying Mappings

**For firms that undergo M&A:**

```python
def get_gvkey_for_year(assignee_id, year, time_mapping):
    """
    Get correct GVKEY for a given year, accounting for M&A.

    Parameters:
    -----------
    assignee_id : str
        Patent assignee identifier
    year : int
        Year of patent application
    time_mapping : DataFrame
        DISCERN 2 time-varying mapping table

    Returns:
    --------
    gvkey : str or None
    """
    matches = time_mapping[
        (time_mapping['assignee_id'] == assignee_id) &
        (time_mapping['year_start'] <= year) &
        (time_mapping['year_end'] >= year)
    ]

    if len(matches) > 0:
        return matches.iloc[0]['gvkey']
    return None

# Apply time-varying mapping
patents['gvkey'] = patents.apply(
    lambda row: get_gvkey_for_year(
        row['assignee_id'],
        row['filing_year'],
        discern_time
    ),
    axis=1
)
```

### Fallback: Clinical Trials Mapping

**If DISCERN 2 is unavailable:**

Use the existing sponsor_name → gvkey mapping from clinical trials as a proxy:

```python
# Extract from clinical trials
sponsor_lookup = dict(zip(
    clinical_trials['sponsor_name'].apply(clean_org_name),
    clinical_trials['gvkey_sponsor']
))

# Apply to patents
patents['gvkey'] = patents['applicant_organization'].apply(
    lambda x: sponsor_lookup.get(clean_org_name(x))
)
```

**Limitations:**
- Lower coverage (only trial sponsors)
- May miss firms without trials
- No time-varying mappings

---

## Troubleshooting

### Issue 1: File Download Failures

**Symptom:** `urlretrieve()` times out or fails

**Solutions:**
```python
# Option 1: Increase timeout
import socket
socket.setdefaulttimeout(300)  # 5 minutes

# Option 2: Use requests library
import requests
response = requests.get(url, stream=True, timeout=300)
with open(filepath, 'wb') as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)

# Option 3: Manual download
# Download via browser, then point code to local file
```

### Issue 2: Memory Errors

**Symptom:** `MemoryError` or kernel crash

**Solutions:**
1. Use DuckDB instead of pandas
2. Process year-by-year
3. Reduce chunk size
4. Use categorical dtypes
5. Close Jupyter notebooks you're not using
6. Restart kernel periodically

```python
# Check available memory
import psutil
mem = psutil.virtual_memory()
print(f"Available: {mem.available / 1024**3:.1f} GB")
```

### Issue 3: Low GVKEY Match Rate

**Symptom:** <50% of patents matched to gvkey

**Causes & Solutions:**

1. **Name variations:**
   - Improve `clean_org_name()` function
   - Add more legal suffix patterns
   - Handle international characters

2. **Missing from DISCERN:**
   - Check if firms are public (DISCERN only covers public firms)
   - Use alternative identifier (e.g., Org ID from PatentsView)

3. **Wrong table:**
   - Verify using correct DISCERN table
   - Check for year-specific mappings

**Debugging:**
```python
# Find unmatched applicants
unmatched = patents[patents['gvkey'].isna()]['applicant_organization'].value_counts()
print("Top unmatched applicants:")
print(unmatched.head(20))

# Manual review reveals patterns
```

### Issue 4: No AI Patents Found

**Symptom:** `ai_applications == 0` for all firms

**Causes & Solutions:**

1. **CPC table not joined:**
   - Verify `g_cpc_current` loaded correctly
   - Check join keys match

2. **Wrong CPC pattern:**
   - Verify using correct CPC prefixes
   - Check CPC code format in data

3. **Keyword search too strict:**
   - Reduce minimum keyword count
   - Add more keyword variations
   - Check case sensitivity

**Debugging:**
```python
# Check CPC codes present
sample_cpc = con.execute("""
    SELECT DISTINCT cpc_group
    FROM g_cpc_current
    WHERE cpc_group LIKE 'G06N%'
    LIMIT 10
""").df()
print(sample_cpc)

# Test keyword matching
test_text = "machine learning model for drug discovery"
is_ai, keywords = contains_ai_keywords(test_text)
print(f"AI: {is_ai}, Keywords: {keywords}")
```

### Issue 5: Duplicate Applications

**Symptom:** Same application appears multiple times

**Causes:**
- Multiple assignees per patent
- Multiple CPC codes per patent
- Duplicate rows in source data

**Solutions:**
```python
# Deduplicate by application_id
patents = patents.drop_duplicates(subset='application_id')

# Or keep one assignee per patent (first listed)
patents = patents.groupby('application_id').first().reset_index()

# Or aggregate to application level
patents_agg = patents.groupby('application_id').agg({
    'filing_date': 'first',
    'gvkey': lambda x: ','.join(x.dropna().unique()),  # Multiple gvkeys
    'is_ai': 'max'  # AI if any assignee is flagged
}).reset_index()
```

---

## Validation Checklist

### Data Quality Checks

- [ ] **File integrity:**
  - [ ] All TSV files downloaded and extracted
  - [ ] File sizes match expected values
  - [ ] No corrupted/truncated files

- [ ] **Import validation:**
  - [ ] Row counts match expected values
  - [ ] No duplicate primary keys
  - [ ] Date formats parsed correctly
  - [ ] Missing values documented

- [ ] **Merge quality:**
  - [ ] Join keys validated (no spurious matches)
  - [ ] Match rates documented
  - [ ] Unmatched records investigated

### AI Classification Validation

- [ ] **CPC classification:**
  - [ ] Sample 50 CPC-flagged patents manually reviewed
  - [ ] Precision calculated: ____%
  - [ ] False positives investigated
  - [ ] CPC code distribution sensible

- [ ] **Keyword classification:**
  - [ ] Sample 50 keyword-flagged patents reviewed
  - [ ] Precision calculated: ____%
  - [ ] False positives addressed
  - [ ] Keyword list refined

- [ ] **Temporal trends:**
  - [ ] AI patents increase over time (2000-2025)
  - [ ] No suspicious spikes or drops
  - [ ] Consistent with external sources

### Output Validation

- [ ] **Patent-level dataset:**
  - [ ] Unique application_id per row
  - [ ] All flags (is_ai, ai_method) populated
  - [ ] No missing gvkey for known firms
  - [ ] Year range: 2000-2025

- [ ] **Firm-year dataset:**
  - [ ] One row per gvkey-year
  - [ ] Counts match patent-level data
  - [ ] ai_share = ai_applications / total_applications
  - [ ] No negative values

- [ ] **Merged dataset:**
  - [ ] Clinical trials + patents aligned
  - [ ] Merge statistics documented
  - [ ] Missing values handled appropriately

### Reproducibility

- [ ] **Code:**
  - [ ] All cells run without errors
  - [ ] Random seeds set (if applicable)
  - [ ] File paths parameterized
  - [ ] Dependencies documented

- [ ] **Documentation:**
  - [ ] Methods clearly explained
  - [ ] Assumptions stated
  - [ ] Limitations acknowledged
  - [ ] Data sources cited

---

## References

### Data Sources

1. **PatentsView**
   - Bulk Downloads: https://patentsview.org/download/data-download-tables
   - API Documentation: https://search.patentsview.org/docs/
   - Data Dictionary: https://patentsview.org/download/data-download-dictionary

2. **DISCERN 2**
   - Zenodo Repository: https://zenodo.org/records/13619821
   - Original Paper: Arora et al. (2021)
   - User Guide: Included in download

3. **ClinicalTrials.gov**
   - Website: https://clinicaltrials.gov/
   - API: https://clinicaltrials.gov/api/

### CPC Classification

1. **Cooperative Patent Classification**
   - Official Website: https://www.cooperativepatentclassification.org/
   - CPC Code Lookup: https://www.uspto.gov/web/patents/classification/cpc/html/cpc.html
   - AI-Related Codes: https://www.wipo.int/about-ip/en/frontier_technologies/

2. **WIPO Technology Concordance**
   - AI Technologies: https://www.wipo.int/export/sites/www/about-ip/en/frontier_technologies/pdf/ai.pdf

### AI Patent Research

1. **Fujii & Managi (2018):** "Trends and priority shifts in artificial intelligence technology invention"
   - DOI: 10.1016/j.econmod.2018.08.013

2. **Babina et al. (2024):** "Artificial Intelligence, Firm Growth, and Product Innovation"
   - DOI: 10.1016/j.jfineco.2024.103824

3. **OECD AI Patents Report (2021)**
   - Link: https://www.oecd.org/sti/measuring-innovation-in-artificial-intelligence.pdf

### Python Libraries

```python
# Core data processing
pandas>=1.5.0
numpy>=1.23.0

# Large file handling
duckdb>=0.9.0
pyarrow>=10.0.0

# Text processing
rapidfuzz>=2.0.0  # Fuzzy string matching

# Memory profiling
psutil>=5.9.0
memory_profiler>=0.60.0

# Utilities
tqdm>=4.65.0  # Progress bars
```

### Recommended Reading

1. **Hall, Jaffe, Trajtenberg (2001):** "The NBER Patent Citations Data File"
   - Foundational work on patent data for research

2. **Marx & Fuegi (2020):** "Reliance on Science by Inventors"
   - Methods for linking patents to scientific publications

3. **Cockburn et al. (2016):** "Patents and the Global Diffusion of New Drugs"
   - Pharmaceutical patent analysis methods

---

## Appendix: Alternative Approaches

### Alternative 1: Use PatentsView API

**Pros:**
- No need to download large files
- Always up-to-date
- Filtered queries reduce data transfer

**Cons:**
- Rate limits (10 requests/second)
- Max 1,000 records per query
- Slower for bulk analysis
- Requires API key

**Example:**
```python
import requests

def query_patentsview(criteria, fields):
    """
    Query PatentsView API.

    Example:
    criteria = {"filing_date": {"gte": "2020-01-01"}}
    fields = ["application_id", "filing_date", "title"]
    """
    url = "https://search.patentsview.org/api/v1/application/"

    payload = {
        "q": criteria,
        "f": fields,
        "per_page": 1000
    }

    response = requests.post(url, json=payload)
    return response.json()
```

### Alternative 2: Use Google BigQuery

**PatentsView data available on Google BigQuery:**
- Dataset: `patents-public-data`
- No download required
- SQL interface
- Scales to full dataset

**Pros:**
- No local storage needed
- Fast queries on full dataset
- Joins pre-optimized

**Cons:**
- Requires Google Cloud account
- Costs for queries >1TB/month
- Learning curve for BigQuery SQL

**Example:**
```sql
SELECT
    a.application_id,
    a.filing_date,
    c.cpc_group,
    p.applicant_organization
FROM `patents-public-data.patents.applications` a
LEFT JOIN `patents-public-data.patents.cpc_current` c
    ON a.patent_id = c.patent_id
LEFT JOIN `patents-public-data.patents.applicants` p
    ON a.application_id = p.application_id
WHERE a.filing_date BETWEEN '2000-01-01' AND '2025-12-31'
    AND c.cpc_group LIKE 'G06N%'
```

### Alternative 3: Pre-Built AI Patent Datasets

**WIPO AI Patent Dataset:**
- https://www.wipo.int/tech_trends/en/artificial_intelligence/
- Pre-classified AI patents
- 2000-2019 coverage

**Pros:**
- Already classified by experts
- High precision
- Free to use

**Cons:**
- Only through 2019
- Doesn't include applications (only grants)
- No biopharma focus

---

**Last Updated:** February 14, 2026
**Version:** 1.0
**Contact:** Edward Jung
