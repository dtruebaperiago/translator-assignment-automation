/*
  extract_data.js — Run with: node extract_data.js
  
  Reads DATA/Initial Dataset/CSV/Data.csv (semicolon-separated, comma decimals)
  and produces:
    - frontend/data/translators.json
    - frontend/data/clients.json
*/
const fs = require('fs');
const path = require('path');

const CSV_PATH = path.join(__dirname, '..', 'DATA', 'Initial Dataset', 'CSV', 'Data.csv');
const OUT_DIR  = path.join(__dirname, 'data');
if (!fs.existsSync(OUT_DIR)) fs.mkdirSync(OUT_DIR, { recursive: true });

console.log('Reading CSV...');
const raw = fs.readFileSync(CSV_PATH, 'utf-8');
const lines = raw.split('\n').filter(l => l.trim());
const headers = lines[0].split(';').map(h => h.trim());
console.log(`  ${lines.length - 1} rows, ${headers.length} cols`);

// Parse rows
const rows = [];
for (let i = 1; i < lines.length; i++) {
  const vals = lines[i].split(';');
  if (vals.length < headers.length) continue;
  const obj = {};
  headers.forEach((h, j) => { obj[h] = (vals[j] || '').trim(); });
  rows.push(obj);
}
console.log(`  Parsed ${rows.length} rows.`);

function commaFloat(s) { return parseFloat((s || '0').replace(',', '.')); }

// ── TRANSLATORS ──────────────────────────────────────────────
console.log('\nExtracting translators...');
const trMap = {};
for (const r of rows) {
  const name = r.TRANSLATOR;
  if (!name) continue;
  if (!trMap[name]) trMap[name] = { tasks: new Set(), quality: [], rate: [], hours: 0, types: new Set(), srcs: new Set(), tgts: new Set(), clients: new Set(), pairs: {} };
  const t = trMap[name];
  t.tasks.add(r.TASK_ID);
  if (r.QUALITY_EVALUATION) t.quality.push(commaFloat(r.QUALITY_EVALUATION));
  if (r.HOURLY_RATE) t.rate.push(commaFloat(r.HOURLY_RATE));
  t.hours += commaFloat(r.HOURS);
  if (r.TASK_TYPE) t.types.add(r.TASK_TYPE);
  if (r.SOURCE_LANG) t.srcs.add(r.SOURCE_LANG);
  if (r.TARGET_LANG) t.tgts.add(r.TARGET_LANG);
  if (r.MANUFACTURER) t.clients.add(r.MANUFACTURER);
  const pairKey = `${r.SOURCE_LANG}→${r.TARGET_LANG}`;
  t.pairs[pairKey] = (t.pairs[pairKey] || 0) + 1;
}

const translators = Object.entries(trMap).map(([name, t], i) => {
  const avg = arr => arr.length ? Math.round(arr.reduce((a,b) => a+b, 0) / arr.length * 10) / 10 : 0;
  // Find most common pair
  let topPair = { src: 'N/A', tgt: 'N/A' };
  let topCount = 0;
  for (const [pk, cnt] of Object.entries(t.pairs)) {
    if (cnt > topCount) { topCount = cnt; const [s, tg] = pk.split('→'); topPair = { src: s, tgt: tg }; }
  }
  return {
    id: `TR-${String(i + 1).padStart(4, '0')}`,
    name,
    source: topPair.src,
    target: topPair.tgt,
    rate: avg(t.rate),
    quality: avg(t.quality),
    taskTypes: [...t.types],
    taskCount: t.tasks.size,
    totalHours: Math.round(t.hours * 10) / 10,
    sourceLangs: [...t.srcs],
    targetLangs: [...t.tgts],
    clientsWorked: t.clients.size,
    status: 'Available',
    workerType: t.tasks.size > 20 ? 'Internal' : 'Third-Party',
  };
});

fs.writeFileSync(path.join(OUT_DIR, 'translators.json'), JSON.stringify(translators));
console.log(`  ${translators.length} translators → data/translators.json`);

// ── CLIENTS ──────────────────────────────────────────────────
console.log('\nExtracting clients...');
const clMap = {};
for (const r of rows) {
  const name = r.MANUFACTURER;
  if (!name) continue;
  if (!clMap[name]) clMap[name] = { industry: r.MANUFACTURER_INDUSTRY, sector: r.MANUFACTURER_SECTOR, industryGroup: r.MANUFACTURER_INDUSTRY_GROUP, subindustry: r.MANUFACTURER_SUBINDUSTRY, tasks: new Set(), quality: [], rate: [], translators: new Set(), types: new Set() };
  const c = clMap[name];
  c.tasks.add(r.TASK_ID);
  if (r.QUALITY_EVALUATION) c.quality.push(commaFloat(r.QUALITY_EVALUATION));
  if (r.HOURLY_RATE) c.rate.push(commaFloat(r.HOURLY_RATE));
  if (r.TRANSLATOR) c.translators.add(r.TRANSLATOR);
  if (r.TASK_TYPE) c.types.add(r.TASK_TYPE);
}

const clients = Object.entries(clMap).map(([name, c], i) => {
  const avg = arr => arr.length ? Math.round(arr.reduce((a,b) => a+b, 0) / arr.length * 10) / 10 : 0;
  return {
    id: `CL-${String(i + 1).padStart(4, '0')}`,
    name,
    industry: c.industry || 'General',
    sector: c.sector || '',
    industryGroup: c.industryGroup || '',
    subindustry: c.subindustry || '',
    taskCount: c.tasks.size,
    avgRate: avg(c.rate),
    avgQuality: avg(c.quality),
    translatorsUsed: c.translators.size,
    taskTypes: [...c.types],
  };
});

fs.writeFileSync(path.join(OUT_DIR, 'clients.json'), JSON.stringify(clients));
console.log(`  ${clients.length} clients → data/clients.json`);
console.log('\nDone!');
