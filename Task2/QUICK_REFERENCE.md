# Task #2 Quick Reference

## What Your Supervisor Wants

```
┌─────────────────────────────────────────────────┐
│  FIRM-YEAR AGGREGATED TABLE                    │
│  (gvkey × year)                                 │
│                                                 │
│  NOT patent-level details                      │
│  NOT a list of individual patents              │
│  Just clean summary counts per firm per year   │
└─────────────────────────────────────────────────┘
```

---

## The Deliverable

### File: `firm_year_patents.csv`

```csv
gvkey,year,total_applications,ai_applications,ai_share,ai_dummy
8530,2015,342,12,0.035,1
8530,2016,389,18,0.046,1
8530,2017,401,24,0.060,1
9775,2015,298,5,0.017,1
9775,2016,312,8,0.026,1
```

### Columns Explained

- **gvkey**: Firm identifier (from Compustat)
- **year**: Application year (2000-2025)
- **total_applications**: How many patents the firm applied for that year
- **ai_applications**: How many of those were AI-related
- **ai_share**: Percentage that are AI (ai / total)
- **ai_dummy**: 1 if firm has any AI patents that year, 0 otherwise

---

## How It's Built (Internal Process)

```
Step 1: Get Patent Applications (2000-2025)
          ↓
Step 2: Map to GVKEY (firm identifier)
          ↓
Step 3: Classify AI Patents
        - CPC codes (G06N*)
        - Keywords (machine learning, neural network, etc.)
        - Mark is_ai = True if either method finds it
          ↓
Step 4: AGGREGATE to Firm-Year ⭐
        - Group by gvkey + year
        - Count total applications
        - Count AI applications
        - Calculate ai_share and ai_dummy
          ↓
Step 5: Export firm_year_patents.csv
```

---

## What Gets Submitted

### 1. Primary Output
**File:** `firm_year_patents.csv`
**What:** Clean aggregated table, one row per firm-year
**Size:** ~500-2000 rows (depending on firms matched)

### 2. Methodology
**File:** `task2_patents_discern_merge.ipynb`
**What:** Jupyter notebook showing complete workflow
**Purpose:** Reproducible code + documentation

### 3. Documentation (optional)
**Files:** README.md, IMPLEMENTATION_GUIDE.md
**What:** Explains methodology, best practices, troubleshooting

---

## Notebook Already Does This!

The notebook I created (`task2_patents_discern_merge.ipynb`) follows exactly this structure:

✅ **Section 1-5:** Load data, classify AI patents (internal)
✅ **Section 6:** Aggregate to firm-year (deliverable)
✅ **Section 7:** Merge with clinical trials (optional)
✅ **Section 8:** Export firm_year_patents.csv

**You just need to run it!**

---

## Common Questions

### Q: Do I submit the patent-level dataset?
**A:** No, unless your supervisor specifically asks for it. The primary deliverable is the **aggregated firm-year table**.

### Q: How do I show my work?
**A:** The Jupyter notebook shows the complete methodology. It documents how you went from raw patent data → AI classification → firm-year aggregation.

### Q: What about validation?
**A:** You can optionally export the patent-level dataset to spot-check specific patents and validate your AI classification. But the final deliverable is still the aggregated table.

### Q: Can I include clinical trials in the output?
**A:** Yes! Use `firm_year_merged.csv` instead, which includes both patent metrics AND trial counts.

---

## File Structure After Running Notebook

```
Task2/
├── clinical_trial_sample (1).csv          [INPUT]
├── task2_patents_discern_merge.ipynb      [METHODOLOGY] ⭐
│
├── firm_year_patents.csv                  [OUTPUT] ⭐⭐⭐
├── firm_year_merged.csv                   [OUTPUT - alternative]
│
├── patent_level_dataset.csv               [OPTIONAL - validation]
├── task2_patents.ddb                      [INTERNAL - DuckDB database]
│
└── Documentation/
    ├── README.md
    ├── IMPLEMENTATION_GUIDE.md
    ├── DELIVERABLES_SUMMARY.md
    └── QUICK_REFERENCE.md (this file)
```

---

## Summary

**Supervisor wants:** Firm-year aggregated table (gvkey × year)

**You have:** A notebook that builds this exactly

**Next step:** Run the notebook and submit:
1. `firm_year_patents.csv` (the aggregated table)
2. `task2_patents_discern_merge.ipynb` (reproducible methodology)

**That's it!**

---

## AI Classification Methods

For your reference, here's how patents are flagged as AI:

### Method 1: CPC Codes (High Precision)
```
G06N3  - Neural networks
G06N5  - Knowledge-based models
G06N7  - Probabilistic/fuzzy logic
G06N10 - Quantum computing
G06N20 - Machine learning
```

### Method 2: Keywords (High Recall)
```
- machine learning
- deep learning
- neural network
- artificial intelligence
- reinforcement learning
- computer vision
- natural language processing
- random forest
- gradient boosting
... and 20+ more terms
```

### Combined Approach
```
is_ai = (has_ai_cpc_code OR contains_ai_keywords)
```

A patent is marked as AI if **either** method flags it.

---

**Last Updated:** February 14, 2026
