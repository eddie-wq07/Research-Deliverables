# Task #2 Deliverables Summary

## What Your Supervisor Wants

**PRIMARY DELIVERABLE:** A clean **firm-year aggregated table** with one row per firm-year (gvkey × year)

**NOT:** A detailed patent-level file

---

## The Firm-Year Table

### Structure
```
One row per gvkey × year combination
```

### Required Columns

| Column | Type | Description |
|--------|------|-------------|
| `gvkey` | string | Firm identifier from Compustat |
| `year` | integer | Calendar year (2000-2025) |
| `total_applications` | integer | Total patent applications filed by firm in year |
| `ai_applications` | integer | AI-related patent applications |
| `ai_share` | float | AI applications / total applications |
| `ai_dummy` | integer | 1 if firm has ≥1 AI patent that year, 0 otherwise |

### Example Data

```
gvkey    year    total_applications    ai_applications    ai_share    ai_dummy
8530     2015    342                   12                 0.035       1
8530     2016    389                   18                 0.046       1
9775     2015    298                   5                  0.017       1
9775     2016    312                   8                  0.026       1
```

---

## Workflow (Internal - For Your Understanding)

Even though the deliverable is aggregated, you internally need to:

### Step 1: Patent-Level Classification (Internal)
Build a working dataset to classify AI patents:
- Load patent applications (2000-2025)
- Map applicants to gvkey
- Classify as AI using:
  - **CPC codes** (G06N*)
  - **Keywords** (machine learning, neural network, etc.)
- Mark `is_ai = True` if either method flags it

### Step 2: Aggregate to Firm-Year (Deliverable)
```python
firm_year = (
    patent_data
    .groupby(['gvkey', 'year'])
    .agg({
        'application_id': 'count',    # total_applications
        'is_ai': 'sum'                # ai_applications
    })
)

firm_year['ai_share'] = ai_applications / total_applications
firm_year['ai_dummy'] = (ai_applications > 0).astype(int)
```

### Step 3: Export
```python
firm_year.to_csv('firm_year_patents.csv', index=False)
```

---

## What to Submit

### 1. Firm-Year Dataset (CSV)
**File:** `firm_year_patents.csv` or `firm_year_merged.csv`

**Description:** Clean summary table, one row per gvkey-year

**Columns:** gvkey, year, total_applications, ai_applications, ai_share, ai_dummy

### 2. Reproducible Notebook (.ipynb)
**File:** `task2_patents_discern_merge.ipynb`

**Contents:**
- Data import and processing
- AI classification logic (CPC + keywords)
- Firm-year aggregation code
- Documentation of methodology

### Optional (if supervisor wants transparency):
- Patent-level dataset for validation
- Data dictionary explaining classification

---

## Key Points to Remember

✅ **Deliverable = aggregated firm-year table** (not patent-level details)

✅ **Patent-level processing is internal** (just for classification)

✅ **Notebook shows reproducible methodology** (how you got from raw data to firm-year table)

✅ **AI classification uses dual approach:** CPC codes + keywords

✅ **Time period:** 2000-2025 application years

✅ **Firms:** Those conducting clinical trials (from clinical_trial_sample.csv)

---

## Output Files from Notebook

When you run `task2_patents_discern_merge.ipynb`, you will generate:

### Primary:
1. **`firm_year_patents.csv`** ⭐
   - Firm-year patent metrics only
   - ~500-2000 rows (depending on firms matched)

2. **`firm_year_merged.csv`** ⭐
   - Same as above + clinical trial counts
   - Useful if supervisor wants trials merged in

### Optional (can be suppressed):
3. `patent_level_dataset.csv`
   - Detailed patent-level data
   - Only export if you want to validate AI classification
   - Can set `EXPORT_PATENT_LEVEL = False` in notebook to skip

---

## Summary

**Your supervisor confirmed:**
> "The deliverable should be a firm-year dataset (gvkey × year), not patent-level details."

**The notebook already does this!**
- Builds patent-level internally
- Aggregates to firm-year
- Exports clean summary table

**You're all set.** Just run the notebook and submit the firm-year CSV + notebook.

---

**Last Updated:** February 14, 2026
