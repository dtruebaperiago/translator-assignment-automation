
## 🔍 Key EDA Findings
 
### Translator Activity Distribution
 
| Metric | Value |
|--------|-------|
| Total unique translators | ~983 |
| Translators with ≤ 20 tasks | **495 (50.4%)** ⚠️ |
| Average tasks per translator | 1,015 |
| Maximum tasks by one translator | **65,209** ⚠️ |
| Average hours per translator | 1,752 hrs |
| Median hours per translator | **133 hrs** ⚠️ |
 
> ⚠️ The mean/median hours gap (~13×) and the extreme task volume outlier both indicate a heavily skewed distribution that requires careful handling before training.
 
---
 
### Cold-Start Problem
 
**~50% of translators have 20 or fewer historical tasks.** This is the central challenge for any supervised learning approach — there is insufficient feedback signal to learn reliable preferences or performance profiles for half the workforce.
 
```
Translators with ≤  20 tasks  ████████████████████████░░░░░░░░░░░░░░░░  50.4%
Translators with ≤ 500 hrs   █████████████████████████████████░░░░░░░░  65.3%
Translators with ≤ 5000 hrs  ██████████████████████████████████████████  89.0%
```
 
---
 
### Language Pairs
 
- **300 unique language pairs** across 4,688 translator-pair records
- **English** dominates as source language (54% of all pair records)
- **Spanish variants** (Iberian, LA, Global) dominate on the target side
- Records exist where `SOURCE_LANG == TARGET_LANG` — likely editing/proofreading tasks, not translation
 
---
 
### Client Constraints
 
- Selling prices: **€20 – €50/hr** across 7 discrete tiers
- **33% of clients (875) have `MIN_QUALITY = 0`** — no quality floor specified
- Tiebreaker (`WILDCARD`): `Deadline` (916) · `Quality` (875) · `Price` (854)
 
---
 
### Translator Cost vs. Client Rate
 
- Translator rates: **€8 – €62/hr**
- **99 translator rates exceed the maximum client selling price (€50)** — potential margin issue or data error
 
---
 
### Availability (Schedules)
 
- Shift start times span **00:00 – 23:00** — globally distributed workforce
- **34.5% of translators work all 7 days**; no translator works fewer than 4 days
- Working patterns: 4-day (18.4%) · 5-day (27.6%) · 6-day (19.5%) · 7-day (34.5%)
 
---
