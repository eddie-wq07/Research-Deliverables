"""
Fast OpenAlex Author Matching - Async Version
Uses parallel requests for ~10x speedup

Usage: python 02_openalex_fast.py
"""

import asyncio
import aiohttp
import json
import os
import pandas as pd
from collections import defaultdict
from datetime import datetime
import time

# Configuration
EMAIL = "edward.h.jung07@gmail.com"
CACHE_FILE = "openalex_author_cache.json"
RESULTS_FILE = "author_sponsor_matches.csv"
BASE_URL = "https://api.openalex.org"

# Concurrency settings - OpenAlex allows ~10 req/sec for polite pool
MAX_CONCURRENT = 8  # Concurrent requests
BATCH_SIZE = 100    # Save cache every N authors
RATE_LIMIT = 0.1    # Seconds between batch starts

# Sponsor matching keywords
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
        json.dump(cache, f)


async def fetch_author(session, author_name, semaphore):
    """Fetch author data from OpenAlex."""
    cache_key = author_name.lower().strip()

    async with semaphore:
        try:
            # Search for author
            async with session.get(
                f"{BASE_URL}/authors",
                params={
                    'filter': f'display_name.search:{author_name}',
                    'per_page': 1,
                    'mailto': EMAIL
                },
                timeout=aiohttp.ClientTimeout(total=15)
            ) as response:
                if response.status != 200:
                    return cache_key, {'queried': True, 'found': False, 'error': f'HTTP {response.status}'}

                data = await response.json()

                if not data.get('results'):
                    return cache_key, {'queried': True, 'found': False}

                author_id = data['results'][0].get('id', '').replace('https://openalex.org/', '')

            # Get full author details
            async with session.get(
                f"{BASE_URL}/authors/{author_id}",
                params={'mailto': EMAIL},
                timeout=aiohttp.ClientTimeout(total=15)
            ) as response:
                if response.status != 200:
                    return cache_key, {'queried': True, 'found': False, 'error': f'HTTP {response.status}'}

                full_author = await response.json()

            # Extract company affiliations
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

            return cache_key, result

        except asyncio.TimeoutError:
            return cache_key, {'queried': True, 'found': False, 'error': 'timeout'}
        except Exception as e:
            return cache_key, {'queried': True, 'found': False, 'error': str(e)}


async def process_batch(authors, cache):
    """Process a batch of authors concurrently."""
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT, limit_per_host=MAX_CONCURRENT)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [fetch_author(session, author, semaphore) for author in authors]
        results = await asyncio.gather(*tasks)

        for cache_key, data in results:
            cache[cache_key] = data

    return len(results)


def match_to_sponsors(companies, sponsor_names):
    """Match company affiliations to clinical trial sponsors."""
    matches = []

    for company in companies:
        company_name = (company.get('name') or '').lower()
        years = company.get('years', [])

        for keyword, canonical_sponsor in SPONSOR_KEYWORDS.items():
            if keyword in company_name:
                matching = [s for s in sponsor_names if canonical_sponsor.lower() in s.lower()]
                if matching:
                    matches.append({
                        'company': company.get('name'),
                        'company_years': years,
                        'sponsor': matching[0]
                    })
                    break

    return matches


async def main():
    print("=" * 70)
    print("FAST OpenAlex Author Matching (Async)")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Concurrency: {MAX_CONCURRENT} parallel requests")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    with open('data/neurips.json/neurips.json', 'r') as f:
        neurips_data = json.load(f)

    df_trials = pd.read_csv('final_sample_clinical_trials_information.csv')
    sponsor_names = df_trials['sponsor_name'].unique()
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

    # Filter to uncached
    uncached = [a for a in all_authors if a.lower().strip() not in cache]

    print(f"Total authors: {len(all_authors)}")
    print(f"Already cached: {len(cache)}")
    print(f"To process: {len(uncached)}")

    if uncached:
        print(f"\nProcessing {len(uncached)} authors...")
        print("-" * 70)

        start_time = time.time()
        processed = 0

        # Process in batches
        for i in range(0, len(uncached), BATCH_SIZE):
            batch = uncached[i:i + BATCH_SIZE]
            batch_start = time.time()

            count = await process_batch(batch, cache)
            processed += count

            # Save cache
            save_cache(cache)

            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (len(uncached) - processed) / rate if rate > 0 else 0

            print(f"  [{processed}/{len(uncached)}] {rate:.1f} authors/sec | ETA: {eta/60:.1f} min")

            # Small delay between batches
            await asyncio.sleep(RATE_LIMIT)

        total_time = time.time() - start_time
        print(f"\nCompleted in {total_time/60:.1f} minutes ({processed/total_time:.1f} authors/sec)")

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

        matches = match_to_sponsors(data['companies'], sponsor_names)

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

        print("\nTOP MATCHES BY SPONSOR:")
        sponsor_counts = df_matches['clinical_trial_sponsor'].value_counts()
        for sponsor, count in sponsor_counts.head(20).items():
            print(f"  {sponsor}: {count} authors")

        print(f"\nTotal unique author-sponsor pairs: {len(unique_matches)}")
    else:
        print("\nNo matches found.")

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    asyncio.run(main())
