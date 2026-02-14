# Biopharma AI Capabilities Research

Research repository for analyzing AI capabilities in biopharma firms through patent applications and clinical trials.

## Repository Structure

```
research_task1/
├── Task1/                      # PatentsView Database Exploration
│   └── task1_patentsview_exploration.ipynb
├── Task2/                      # Patents, DISCERN, and Clinical Trials Integration
│   ├── task2_patents_discern_merge.ipynb
│   ├── README.md
│   ├── QUICK_REFERENCE.md
│   ├── DELIVERABLES_SUMMARY.md
│   ├── IMPLEMENTATION_GUIDE.md
│   └── RA Task #2.pdf
└── README.md (this file)
```

---

## Task 1: PatentsView Database Exploration

**Objective:** Explore PatentsView database and build a pipeline for identifying AI-related patents.

**Key Features:**
- Downloads 4 core PatentsView tables directly from S3 (~2 GB for 2021 data)
- Identifies AI patents using CPC codes (G06N family)
- Enables biopharma domain filtering and firm-level analysis
- 2021 test dataset: 363,829 patents, 10,782 AI-related

**Usage:**
```bash
cd Task1/
jupyter notebook task1_patentsview_exploration.ipynb
```

---

## Task 2: Merging PatentsView, DISCERN, and Clinical Trials

**Objective:** Construct a firm-year dataset of AI-related patent applications for firms conducting clinical trials.

**Primary Deliverable:** Firm-year aggregated table (gvkey × year) with columns:
- `gvkey` - Firm identifier
- `year` - Application year (2000-2025)
- `total_applications` - Total patent applications filed
- `ai_applications` - AI-related applications
- `ai_share` - Percentage of AI patents (ai / total)
- `ai_dummy` - Binary indicator (1 if firm has ≥1 AI patent)

**Key Features:**
- Uses **patent applications** (not just granted patents) for temporal alignment
- AI classification via dual approach: CPC codes (G06N*) + keyword filtering
- Maps applicants to GVKEY using DISCERN 2 database
- Memory-efficient processing using DuckDB
- Time period: 2000-2025

**Quick Start:**
```bash
cd Task2/
# Read QUICK_REFERENCE.md first for overview
jupyter notebook task2_patents_discern_merge.ipynb
```

**Documentation:**
- **QUICK_REFERENCE.md** - Visual summary and key concepts
- **DELIVERABLES_SUMMARY.md** - What gets submitted
- **IMPLEMENTATION_GUIDE.md** - Technical deep dive (24 KB)
- **README.md** - Detailed task documentation

**Output:**
- `firm_year_patents.csv` - Clean aggregated table (primary deliverable)
- `firm_year_merged.csv` - Includes clinical trial metrics
- Reproducible Jupyter notebook with complete methodology

---

## Data Sources

1. **PatentsView** - Patent application data (2000-2025)
   - URL: https://patentsview.org/download/data-download-tables
   - Tables: g_application, pg_applicant_not_disambiguated, g_cpc_current, g_patent_abstract

2. **DISCERN 2** - Patent assignee to GVKEY mapping
   - URL: https://zenodo.org/records/13619821
   - Enables firm identification and tracking

3. **ClinicalTrials.gov** - Clinical trial data
   - Includes NCT IDs, sponsor names, start years, phases

---

## Research Context

**Paper:** "How Does AI Change Drug Development? Evidence from Clinical Trial Phases and Drug Types"
- Authors: Angela Kwon, Jaecheol Park, Gene Moo Lee
- Focus: Impact of AI in downstream drug development (clinical trials)
- Finding: Firms' AI capability is positively associated with more refinement trials for biologics in early phases

**Research Goal:** Expand and triangulate firm-level AI capability measures using:
1. **AI-related patents** at the firm level (PatentsView) ← Task 2
2. Trial outcomes (ClinicalTrials.gov → Publications/Conference abstracts) ← Future work

---

## Technical Stack

- **Python 3.9+**
- **DuckDB** - Memory-efficient large file processing
- **Pandas** - Data manipulation and analysis
- **Jupyter** - Interactive notebook environment

**Optional:**
- **RapidFuzz** - Fuzzy string matching (for DISCERN 2 integration)
- **psutil** - Memory monitoring

---

## Notes

- Large data files (.csv, .tsv, .ddb) are excluded from git via .gitignore
- PatentsView tables will be auto-downloaded by notebooks (~10-15 GB total)
- DISCERN 2 data must be downloaded separately (see Task2/IMPLEMENTATION_GUIDE.md)

---

**Last Updated:** February 14, 2026
**Author:** Edward Jung
