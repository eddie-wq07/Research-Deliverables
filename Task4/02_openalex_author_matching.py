"""
OpenAlex Author-to-Company Matching Script
Matches NeurIPS authors to clinical trial sponsors via company affiliations.

Usage:
    python 02_openalex_author_matching.py [--limit N] [--resume]

Options:
    --limit N   Process only N authors (default: all)
    --resume    Resume from where we left off using cache
"""

import requests
import json
import time
import os
import sys
import pandas as pd
from collections import defaultdict
from datetime import datetime

# Configuration
EMAIL = "edward.h.jung07@gmail.com"
CACHE_FILE = "openalex_author_cache.json"
RESULTS_FILE = "author_sponsor_matches.csv"
BASE_URL = "https://api.openalex.org"
RATE_LIMIT_DELAY = 0.12  # ~8 requests/second, well under 100k/day

# Clinical trial sponsor keywords for matching
SPONSOR_KEYWORDS = {
    'novartis': 'Novartis',
    'pfizer': 'Pfizer',
    'roche': 'Hoffmann-La Roche',
    'merck': 'Merck',
    'bayer': 'Bayer',
    'sanofi': 'Sanofi',
    'astrazeneca': 'AstraZeneca',
    'glaxosmithkline': 'GlaxoSmithKline',
    'gsk': 'GlaxoSmithKline',
    'bristol-myers': 'Bristol-Myers Squibb',
    'bristol myers': 'Bristol-Myers Squibb',
    'johnson & johnson': 'Johnson & Johnson',
    'janssen': 'Johnson & Johnson',
    'eli lilly': 'Eli Lilly and Company',
    'lilly': 'Eli Lilly and Company',
    'abbvie': 'AbbVie',
    'amgen': 'Amgen',
    'gilead': 'Gilead Sciences',
    'biogen': 'Biogen',
    'regeneron': 'Regeneron',
    'takeda': 'Takeda',
    'boehringer ingelheim': 'Boehringer Ingelheim',
    'boehringer': 'Boehringer Ingelheim',
    'novo nordisk': 'Novo Nordisk A/S',
    'allergan': 'Allergan',
    'genentech': 'Genentech, Inc.',
    'celgene': 'Celgene',
    'alexion': 'Alexion',
    'vertex': 'Vertex Pharmaceuticals',
    'illumina': 'Illumina',
    'ibm': 'IBM',
    'google': 'Google',
    'deepmind': 'Google',
    'microsoft': 'Microsoft',
    'nvidia': 'NVIDIA',
    'facebook': 'Facebook',
    'meta': 'Facebook',
    'amazon': 'Amazon',
    'intel': 'Intel',
    'apple': 'Apple',
}


def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_cache(cache):
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)


def get_author_affiliations(author_name, cache):
    """Get author's company affiliations from OpenAlex."""
    cache_key = author_name.lower().strip()

    # Return cached result if exists
    if cache_key in cache:
        cached = cache[cache_key]
        if isinstance(cached, dict) and 'queried' in cached:
            return cached

    try:
        # Search for author
        response = requests.get(
            f"{BASE_URL}/authors",
            params={
                'filter': f'display_name.search:{author_name}',
                'per_page': 1,
                'mailto': EMAIL
            },
            timeout=15
        )
        response.raise_for_status()
        data = response.json()

        if not data.get('results'):
            result = {'queried': True, 'found': False}
            cache[cache_key] = result
            return result

        author_id = data['results'][0].get('id', '').replace('https://openalex.org/', '')

        # Get full author details
        time.sleep(RATE_LIMIT_DELAY)
        response = requests.get(
            f"{BASE_URL}/authors/{author_id}",
            params={'mailto': EMAIL},
            timeout=15
        )
        response.raise_for_status()
        full_author = response.json()

        # Extract company affiliations only
        companies = []
        for aff in full_author.get('affiliations', []):
            inst = aff.get('institution', {})
            if inst.get('type') == 'company':
                companies.append({
                    'name': inst.get('display_name'),
                    'ror': inst.get('ror'),
                    'country': inst.get('country_code'),
                    'years': aff.get('years', [])
                })

        result = {
            'queried': True,
            'found': True,
            'openalex_id': author_id,
            'display_name': full_author.get('display_name'),
            'works_count': full_author.get('works_count'),
            'companies': companies
        }

        cache[cache_key] = result
        return result

    except Exception as e:
        result = {'queried': True, 'found': False, 'error': str(e)}
        cache[cache_key] = result
        return result


def match_to_sponsors(companies, sponsors_df):
    """Match company affiliations to clinical trial sponsors."""
    matches = []
    sponsor_names = sponsors_df['sponsor_name'].unique()

    for company in companies:
        company_name = (company.get('name') or '').lower()
        years = company.get('years', [])

        for keyword, canonical_sponsor in SPONSOR_KEYWORDS.items():
            if keyword in company_name:
                # Find matching sponsors in our data
                matching = [s for s in sponsor_names if canonical_sponsor.lower() in s.lower()]
                if matching:
                    matches.append({
                        'company': company.get('name'),
                        'company_years': years,
                        'sponsor': matching[0],
                        'keyword_matched': keyword
                    })
                    break

    return matches


def main():
    # Parse arguments
    limit = None
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == '--limit' and i < len(sys.argv) - 1:
            limit = int(sys.argv[i + 1])
        elif arg == '--resume':
            pass  # Resume is default behavior with caching

    print("=" * 70)
    print("OpenAlex Author-to-Company Matching")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    with open('data/neurips.json/neurips.json', 'r') as f:
        neurips_data = json.load(f)

    df_trials = pd.read_csv('final_sample_clinical_trials_information.csv')
    cache = load_cache()

    # Get papers from 2008-2022
    def get_year(p):
        y = p.get('year', 0)
        return int(y) if isinstance(y, int) else (int(y) if str(y).isdigit() else 0)

    papers = [p for p in neurips_data if 2008 <= get_year(p) <= 2022]

    # Build author -> papers mapping
    author_papers = defaultdict(list)
    for paper in papers:
        year = get_year(paper)
        for author in paper.get('authors', []):
            author_papers[author].append({'title': paper.get('title'), 'year': year})

    all_authors = list(author_papers.keys())

    print(f"Papers (2008-2022): {len(papers)}")
    print(f"Unique authors: {len(all_authors)}")
    print(f"Cached authors: {len(cache)}")
    print(f"Clinical trial sponsors: {df_trials['sponsor_name'].nunique()}")

    # Filter to uncached authors
    uncached = [a for a in all_authors if a.lower().strip() not in cache]
    to_process = uncached[:limit] if limit else uncached

    print(f"\nAuthors to process: {len(to_process)}")

    if not to_process:
        print("All authors already cached!")
    else:
        print(f"\nProcessing (with {RATE_LIMIT_DELAY}s delay between requests)...")
        print("-" * 70)

        for i, author in enumerate(to_process):
            result = get_author_affiliations(author, cache)

            companies = result.get('companies', [])
            status = f"{len(companies)} companies" if companies else "no companies"

            if (i + 1) % 50 == 0:
                save_cache(cache)
                print(f"  [{i+1}/{len(to_process)}] Processed, cache saved. Last: {author} ({status})")

            time.sleep(RATE_LIMIT_DELAY)

        save_cache(cache)
        print(f"\nProcessing complete. Cache saved with {len(cache)} authors.")

    # Generate matches
    print("\n" + "=" * 70)
    print("MATCHING TO CLINICAL TRIAL SPONSORS")
    print("=" * 70)

    all_matches = []

    for author in all_authors:
        cache_key = author.lower().strip()
        data = cache.get(cache_key, {})

        if isinstance(data, list):
            data = data[0] if data else {}

        if not data.get('found') or not data.get('companies'):
            continue

        matches = match_to_sponsors(data['companies'], df_trials)

        for match in matches:
            years = match['company_years']
            all_matches.append({
                'author_name': data.get('display_name', author),
                'openalex_id': data.get('openalex_id'),
                'works_count': data.get('works_count'),
                'company': match['company'],
                'company_year_start': min(years) if years else None,
                'company_year_end': max(years) if years else None,
                'clinical_trial_sponsor': match['sponsor'],
                'neurips_papers': len(author_papers.get(author, []))
            })

    # Remove duplicates
    seen = set()
    unique_matches = []
    for m in all_matches:
        key = (m['author_name'], m['clinical_trial_sponsor'])
        if key not in seen:
            seen.add(key)
            unique_matches.append(m)

    # Save results
    if unique_matches:
        df_matches = pd.DataFrame(unique_matches)
        df_matches.to_csv(RESULTS_FILE, index=False)
        print(f"\nSaved {len(unique_matches)} matches to {RESULTS_FILE}")

        # Summary
        print("\nMATCHES BY SPONSOR:")
        sponsor_counts = df_matches['clinical_trial_sponsor'].value_counts()
        for sponsor, count in sponsor_counts.head(20).items():
            print(f"  {sponsor}: {count} authors")

        print(f"\nTotal unique author-sponsor pairs: {len(unique_matches)}")
    else:
        print("\nNo matches found.")

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
