"""
Cross-Reference Analysis: NeurIPS Authors to Clinical Trials
Links author-sponsor matches with clinical trial data.

Usage: python 03_cross_reference_analysis.py
"""

import pandas as pd
import json
from collections import defaultdict
from datetime import datetime

def main():
    print("=" * 70)
    print("CROSS-REFERENCE ANALYSIS: NeurIPS Authors to Clinical Trials")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    df_matches = pd.read_csv('author_sponsor_matches.csv')
    df_trials = pd.read_csv('final_sample_clinical_trials_information.csv')

    with open('data/neurips.json/neurips.json', 'r') as f:
        neurips_data = json.load(f)

    # Filter NeurIPS papers to 2008-2022
    def get_year(p):
        y = p.get('year', 0)
        return int(y) if isinstance(y, int) else (int(y) if str(y).isdigit() else 0)

    papers = [p for p in neurips_data if 2008 <= get_year(p) <= 2022]

    print(f"Author-sponsor matches: {len(df_matches)}")
    print(f"Clinical trials: {len(df_trials)}")
    print(f"NeurIPS papers (2008-2022): {len(papers)}")

    # Build author -> papers mapping
    author_papers = defaultdict(list)
    for paper in papers:
        year = get_year(paper)
        for author in paper.get('authors', []):
            author_papers[author].append({
                'title': paper.get('title'),
                'year': year,
                'url': paper.get('url')
            })

    # Create sponsor name mapping (normalize clinical trial sponsor names)
    sponsor_mapping = {
        'Novartis': ['Novartis'],
        'Pfizer': ['Pfizer'],
        'Hoffmann-La Roche': ['Hoffmann-La Roche', 'Roche', 'Genentech'],
        'Bristol-Myers Squibb': ['Bristol-Myers Squibb', 'Bristol Myers Squibb'],
        'Amgen': ['Amgen'],
        'Sanofi': ['Sanofi', 'Sanofi-Aventis'],
        'AstraZeneca': ['AstraZeneca'],
        'GlaxoSmithKline': ['GlaxoSmithKline', 'GSK'],
        'Eli Lilly and Company': ['Eli Lilly', 'Lilly'],
        'Regeneron Pharmaceuticals': ['Regeneron'],
        'Gilead Sciences': ['Gilead'],
        'Biogen': ['Biogen'],
        'Allergan': ['Allergan'],
        'Bayer': ['Bayer'],
        'Novo Nordisk A/S': ['Novo Nordisk'],
        'Vertex Pharmaceuticals Incorporated': ['Vertex'],
    }

    # Reverse mapping for lookup
    name_to_canonical = {}
    for canonical, variants in sponsor_mapping.items():
        for v in variants:
            name_to_canonical[v.lower()] = canonical

    # Analyze each sponsor
    print("\n" + "=" * 70)
    print("SPONSOR ANALYSIS")
    print("=" * 70)

    results = []

    # Get unique sponsors from matches
    sponsors_in_matches = df_matches['clinical_trial_sponsor'].unique()

    for sponsor in sponsors_in_matches:
        # Find trials for this sponsor
        sponsor_trials = df_trials[
            df_trials['sponsor_name'].str.contains(sponsor.split()[0], case=False, na=False)
        ]

        # Find authors matched to this sponsor
        sponsor_authors = df_matches[df_matches['clinical_trial_sponsor'] == sponsor]

        if len(sponsor_trials) > 0:
            trial_years = sponsor_trials['start_year'].dropna().astype(int)
            trial_year_range = f"{trial_years.min()}-{trial_years.max()}" if len(trial_years) > 0 else "N/A"

            print(f"\n{sponsor}")
            print("-" * 50)
            print(f"  Clinical trials in sample: {len(sponsor_trials)}")
            print(f"  Trial year range: {trial_year_range}")
            print(f"  NeurIPS authors affiliated: {len(sponsor_authors)}")

            # List authors with their papers
            for _, author_row in sponsor_authors.iterrows():
                author_name = author_row['author_name']
                company = author_row['company']
                year_start = author_row['company_year_start']
                year_end = author_row['company_year_end']
                neurips_count = author_row['neurips_papers']

                print(f"\n    {author_name}")
                print(f"      Company: {company}")
                print(f"      Affiliation years: {year_start}-{year_end}")
                print(f"      NeurIPS papers: {neurips_count}")

                # Check overlap with clinical trials
                if pd.notna(year_start) and pd.notna(year_end):
                    overlapping_trials = sponsor_trials[
                        (sponsor_trials['start_year'] >= year_start) &
                        (sponsor_trials['start_year'] <= year_end)
                    ]
                    if len(overlapping_trials) > 0:
                        print(f"      Overlapping trials: {len(overlapping_trials)}")

                results.append({
                    'sponsor': sponsor,
                    'author_name': author_name,
                    'company': company,
                    'affiliation_start': year_start,
                    'affiliation_end': year_end,
                    'neurips_papers': neurips_count,
                    'total_sponsor_trials': len(sponsor_trials),
                    'trial_year_min': trial_years.min() if len(trial_years) > 0 else None,
                    'trial_year_max': trial_years.max() if len(trial_years) > 0 else None
                })

    # Create summary DataFrame
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    df_results = pd.DataFrame(results)

    # Sponsors with most author connections
    print("\nSponsors by Author Connections:")
    sponsor_author_counts = df_results.groupby('sponsor').size().sort_values(ascending=False)
    for sponsor, count in sponsor_author_counts.items():
        trials = df_results[df_results['sponsor'] == sponsor]['total_sponsor_trials'].iloc[0]
        print(f"  {sponsor}: {count} authors, {trials} trials")

    # Authors with most NeurIPS papers
    print("\nTop Authors by NeurIPS Publication Count:")
    top_authors = df_results.nlargest(10, 'neurips_papers')[['author_name', 'sponsor', 'neurips_papers', 'affiliation_start', 'affiliation_end']]
    for _, row in top_authors.iterrows():
        print(f"  {row['author_name']}: {row['neurips_papers']} papers ({row['sponsor']}, {row['affiliation_start']}-{row['affiliation_end']})")

    # Year overlap analysis
    print("\nAffiliation-Trial Year Overlap Analysis:")
    overlap_count = 0
    for _, row in df_results.iterrows():
        if pd.notna(row['affiliation_start']) and pd.notna(row['trial_year_min']):
            aff_start = int(row['affiliation_start'])
            aff_end = int(row['affiliation_end']) if pd.notna(row['affiliation_end']) else aff_start
            trial_min = int(row['trial_year_min'])
            trial_max = int(row['trial_year_max']) if pd.notna(row['trial_year_max']) else trial_min

            # Check if ranges overlap
            if aff_start <= trial_max and aff_end >= trial_min:
                overlap_count += 1

    print(f"  Authors with affiliation overlapping trial years: {overlap_count}/{len(df_results)}")

    # Save detailed results
    df_results.to_csv('author_trial_cross_reference.csv', index=False)
    print(f"\nSaved detailed results to: author_trial_cross_reference.csv")

    # Create firm-year panel for matched sponsors
    print("\n" + "=" * 70)
    print("FIRM-YEAR PANEL DATA")
    print("=" * 70)

    panel_data = []

    for sponsor in sponsors_in_matches:
        sponsor_trials = df_trials[
            df_trials['sponsor_name'].str.contains(sponsor.split()[0], case=False, na=False)
        ]
        sponsor_authors = df_matches[df_matches['clinical_trial_sponsor'] == sponsor]

        # Get all years from trials
        trial_years = sponsor_trials['start_year'].dropna().unique()

        for year in sorted(trial_years):
            year = int(year)
            if 2008 <= year <= 2022:
                # Count trials that year
                trials_that_year = len(sponsor_trials[sponsor_trials['start_year'] == year])

                # Count NeurIPS papers by affiliated authors that year
                neurips_that_year = 0
                authors_that_year = []

                for _, author_row in sponsor_authors.iterrows():
                    author_name = author_row['author_name']
                    # Find papers by this author in this year
                    for orig_name, paper_list in author_papers.items():
                        if author_name.lower() in orig_name.lower() or orig_name.lower() in author_name.lower():
                            papers_year = [p for p in paper_list if p['year'] == year]
                            if papers_year:
                                neurips_that_year += len(papers_year)
                                authors_that_year.append(author_name)

                panel_data.append({
                    'sponsor': sponsor,
                    'year': year,
                    'clinical_trials': trials_that_year,
                    'neurips_papers': neurips_that_year,
                    'affiliated_authors': len(set(authors_that_year))
                })

    if panel_data:
        df_panel = pd.DataFrame(panel_data)
        df_panel = df_panel.sort_values(['sponsor', 'year'])
        df_panel.to_csv('firm_year_panel.csv', index=False)
        print(f"Created firm-year panel with {len(df_panel)} observations")
        print(f"Saved to: firm_year_panel.csv")

        # Summary by sponsor
        print("\nFirm-Year Panel Summary:")
        summary = df_panel.groupby('sponsor').agg({
            'clinical_trials': 'sum',
            'neurips_papers': 'sum',
            'year': ['min', 'max']
        })
        print(summary)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("\nOutput files:")
    print("  - author_trial_cross_reference.csv (author-sponsor-trial details)")
    print("  - firm_year_panel.csv (firm-year panel data)")


if __name__ == "__main__":
    main()
