"""
Edgewise daily digest builder.

Pulls arXiv, library-release and news feeds and keeps two lanes of items:
  - General lane: GNNs and close neighbors (graph transformers, graph
    foundation models, KG embeddings, graph generative models, GraphRAG).
  - Security & privacy lane: ANY graph learning or graph data work, as long
    as it also has a security or privacy signal (e.g. DP synthetic graphs,
    de-anonymization, KG poisoning, GraphRAG attacks, provenance-graph IDS).
Scores them 1-10 for "signal" and writes docs/data/stories.json.

With ANTHROPIC_API_KEY set: Claude filters, scores, summarizes and suggests
research angles. Without it: a free keyword-based scorer is used instead.
"""
import hashlib
import html
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import feedparser

ROOT = Path(__file__).resolve().parent.parent
FEEDS_FILE = ROOT / "scripts" / "feeds.txt"
OUT_FILE = ROOT / "docs" / "data" / "stories.json"

KEEP_DAYS = int(os.environ.get("EDGEWISE_KEEP_DAYS", "365"))
LOOKBACK_DAYS = int(os.environ.get("EDGEWISE_LOOKBACK_DAYS", "3"))
MAX_CANDIDATES = int(os.environ.get("EDGEWISE_MAX_CANDIDATES", "60"))
MODEL = os.environ.get("EDGEWISE_MODEL", "claude-haiku-5-5")

# General lane: an item mentioning any of these is kept.
TOPIC_TERMS = [
    "graph neural", "gnn", "gnns", "message passing", "message-passing", "graph transformer",
    "graph convolution", "gcn", "graph attention", "graphsage", "graph isomorphism network",
    "graph representation learning", "graph foundation model", "graph learning",
    "node classification", "link prediction", "graph classification", "node embedding",
    "pytorch geometric", "pyg", "dgl", "relational deep learning", "heterogeneous graph",
    "temporal graph", "dynamic graph", "hypergraph neural", "graph contrastive",
    "graph unlearning", "graph condensation", "graph pooling", "spectral graph",
    "equivariant graph", "geometric deep learning", "graphrag", "graph rag",
    # close neighbors of GNNs
    "knowledge graph embedding", "kg embedding", "kge", "graph embedding", "network embedding",
    "node2vec", "deepwalk", "graph kernel", "graph generative", "graph generation",
    "graph diffusion", "graph-based rag", "graph retrieval", "graph-augmented",
    "graph foundation", "graph llm", "graph language model", "graph tokeniz",
]

# Security & privacy lane: any of these graph terms counts, but ONLY when a
# STRICT_SECPRIV term also appears. This is what lets in graph work that never
# says "GNN" (graph privacy, de-anonymization, KG poisoning, provenance graphs).
BROAD_GRAPH_TERMS = [
    "graph", "knowledge graph", "social network", "social graph", "random walk",
    "link inference", "edge privacy", "node privacy", "network data", "provenance",
    "call graph", "control-flow graph", "control flow graph", "transaction network",
    "transaction graph", "relational data", "network topology", "subgraph",
]

# Strict security/privacy terms used to gate the broad lane (kept narrower than
# SECPRIV_TERMS so generic words like "robust" or "threat" don't open the gate).
STRICT_SECPRIV = [
    "adversarial attack", "adversarial example", "adversarial perturbation", "backdoor",
    "poisoning", "evasion attack", "membership inference", "attribute inference",
    "link stealing", "model stealing", "model extraction", "inversion attack",
    "reconstruction attack", "de-anonymiz", "deanonymiz", "re-identification",
    "differential privacy", "differentially private", "privacy-preserving", "privacy leak",
    "privacy risk", "privacy attack", "unlearning", "watermark", "certified robust",
    "intrusion detection", "malware", "fraud detection", "phishing", "botnet",
    "money laundering", "jailbreak", "prompt injection", "data extraction",
    "vulnerability detection", "exploit", "sybil", "spoofing",
]

# Security & privacy signals, used to rank candidates and boost heat.
SECPRIV_TERMS = [
    "adversarial", "attack", "backdoor", "poisoning", "evasion", "robust", "certified",
    "membership inference", "link stealing", "stealing", "model extraction", "inversion",
    "privacy", "differential privacy", "differentially private", "unlearning", "federated",
    "watermark", "fingerprint", "intrusion", "malware", "fraud", "anomaly detection",
    "provenance", "vulnerability", "phishing", "botnet", "money laundering", "mule",
    "jamming", "spoof", "threat", "cyber", "security", "de-anonymiz", "deanonymiz",
    "re-identification", "jailbreak", "prompt injection", "sybil",
]

HOT_TERMS = {
    # security & privacy of / with GNNs
    "backdoor": 3, "membership inference": 3, "link stealing": 3, "model extraction": 3,
    "stealing": 2, "inversion": 2, "unlearning": 2, "differential privacy": 2,
    "differentially private": 2, "poisoning": 2, "adversarial": 2, "black-box": 1,
    "certified": 2, "watermark": 2, "intrusion": 2, "malware": 2, "provenance": 2,
    "fraud": 2, "vulnerability": 1, "privacy": 1, "attack": 1, "de-anonymiz": 3,
    "deanonymiz": 3, "re-identification": 2, "synthetic graph": 2, "prompt injection": 2,
    "jailbreak": 2, "graphrag": 2, "knowledge graph": 1,
    # general research signal
    "expressiv": 2, "over-squashing": 2, "oversquashing": 2, "oversmoothing": 1,
    "over-smoothing": 1, "foundation model": 2, "scaling law": 2, "benchmark": 1,
    "dataset": 1, "theory": 1, "neurips": 1, "icml": 1, "iclr": 1, "kdd": 1,
    "usenix": 2, "ccs": 2, "ndss": 2, "s&p": 2, "code available": 1, "github": 1,
    "release": 1,
}

CATEGORIES = [
    "Attacks on graph models", "Privacy of graph models & data", "Defenses & robustness",
    "Graph learning for security & privacy", "Knowledge graphs & GraphRAG",
    "Architectures & theory", "Graph transformers & foundation models",
    "Relational & heterogeneous graphs", "Scalability & systems", "Benchmarks & datasets",
    "Tools & libraries", "Applications",
]

# Older digests used these names; they are renamed on load.
CATEGORY_RENAMES = {
    "Attacks on GNNs": "Attacks on graph models",
    "Privacy of GNNs": "Privacy of graph models & data",
    "GNNs for security & privacy": "Graph learning for security & privacy",
}

# Share of scoring slots reserved for the security & privacy lane on busy days.
SECPRIV_SHARE = float(os.environ.get("EDGEWISE_SECPRIV_SHARE", "0.65"))


def log(*a):
    print(*a, file=sys.stderr)


def clean(text, limit=900):
    text = html.unescape(re.sub(r"<[^>]+>", " ", text or ""))
    text = re.sub(r"\s+", " ", text).strip()
    # arXiv RSS prefixes abstracts with "arXiv:xxxx Announce Type: new Abstract:"
    text = re.sub(r"^arXiv:\S+\s+Announce Type:\s*\S+\s*Abstract:\s*", "", text)
    return text[:limit]


_PAT = {}


def has(text, term):
    """Whole-word-ish match: short terms need boundaries on both sides, longer ones at the start."""
    term = term.strip().lower()
    pat = _PAT.get(term)
    if pat is None:
        tail = r"(?![a-z0-9])" if len(term) <= 4 else ""
        pat = _PAT[term] = re.compile(r"(?<![a-z0-9])" + re.escape(term) + tail)
    return bool(pat.search(text))


def story_id(url, title):
    # arXiv links differ only by version suffix across feeds; normalize them
    url = re.sub(r"(arxiv\.org/abs/\d+\.\d+)v\d+", r"\1", url or "")
    return hashlib.sha1((url or title).encode("utf-8")).hexdigest()[:12]


def entry_date(e):
    for key in ("published_parsed", "updated_parsed"):
        t = e.get(key)
        if t:
            return datetime(*t[:6], tzinfo=timezone.utc)
    return None


def read_feeds():
    feeds = []
    for line in FEEDS_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            name, _, url = line.partition("|")
            feeds.append((name.strip(), url.strip()) if url else (line, line))
    return feeds


def lane_of(text, no_filter=False):
    """Return 'secpriv', 'general', or None (drop)."""
    secpriv = any(has(text, k) for k in STRICT_SECPRIV)
    if no_filter or any(has(text, k) for k in TOPIC_TERMS):
        return "secpriv" if secpriv else "general"
    if secpriv and any(has(text, k) for k in BROAD_GRAPH_TERMS):
        return "secpriv"
    return None


def heat_score(text):
    t = text.lower()
    score = 3 + sum(w for k, w in HOT_TERMS.items() if has(t, k))
    return max(1, min(10, score))


def collect(skip_ids):
    cutoff = datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)
    out, seen_titles = [], set()
    for name, url in read_feeds():
        no_filter = name.startswith("*")
        name = name.lstrip("*").strip()
        try:
            parsed = feedparser.parse(url, agent="EdgewiseDigest/1.0 (+github pages)")
        except Exception as ex:  # noqa: BLE001
            log(f"[skip] {name}: {ex}")
            continue
        if parsed.bozo and not parsed.entries:
            log(f"[skip] {name}: unreadable feed")
            continue
        n = 0
        for e in parsed.entries[:400]:
            title = clean(e.get("title"), 300)
            link = e.get("link", "")
            summary = clean(e.get("summary") or e.get("description"))
            when = entry_date(e)
            if when and when < cutoff:
                continue
            sid = story_id(link, title)
            key = re.sub(r"\W+", "", title.lower())[:80]
            if sid in skip_ids or key in seen_titles:
                continue
            text = f"{title} {summary}".lower()
            lane = lane_of(text, no_filter)
            if lane is None:
                continue
            seen_titles.add(key)
            src = name
            if "news.google.com" in url and e.get("source", {}).get("title"):
                src = e["source"]["title"]
            authors = e.get("author") or ", ".join(a.get("name", "") for a in e.get("authors", []) if a.get("name"))
            out.append({
                "id": sid, "title": title, "url": link, "source": src,
                "authors": clean(authors, 160),
                "published": (when or datetime.now(timezone.utc)).date().isoformat(),
                "raw": summary,
                "lane": lane,
                "_rank": heat_score(text) + (2 if any(has(text, k) for k in SECPRIV_TERMS) else 0),
            })
            n += 1
        log(f"[ok] {name}: {n} candidates")
    # Busy arXiv days produce more papers than we want to score. Reserve a share of
    # slots for each lane so neither crowds out the other, then fill any leftovers.
    out.sort(key=lambda c: c["_rank"], reverse=True)
    sec = [c for c in out if c["lane"] == "secpriv"]
    gen = [c for c in out if c["lane"] == "general"]
    n_sec = min(len(sec), round(MAX_CANDIDATES * SECPRIV_SHARE))
    n_gen = min(len(gen), MAX_CANDIDATES - n_sec)
    picked = sec[:n_sec] + gen[:n_gen]
    rest = sec[n_sec:] + gen[n_gen:]
    rest.sort(key=lambda c: c["_rank"], reverse=True)
    picked += rest[: MAX_CANDIDATES - len(picked)]
    log(f"lanes: {len(sec)} security/privacy, {len(gen)} general -> scoring {len(picked)}")
    return picked


# ---------- Free scorer (no API key) ----------
def guess_category(t):
    t = t.lower()
    rules = [
        ("Privacy of graph models & data", ["de-anonymiz", "deanonymiz", "re-identification", "membership inference", "link stealing", "privacy", "differential privacy",
                             "differentially private", "unlearning", "inversion", "federated"]),
        ("Defenses & robustness", ["defense", "defence", "certified", "robustness", "purif", "detect backdoor"]),
        ("Attacks on graph models", ["backdoor", "poisoning", "adversarial", "evasion", "attack", "model extraction",
                                     "stealing", "jailbreak", "prompt injection"]),
        ("Graph learning for security & privacy", ["intrusion", "malware", "fraud", "phishing", "provenance", "vulnerab",
                                         "botnet", "money laundering", "mule", "jamming", "anomaly", "cyber", "threat"]),
        ("Knowledge graphs & GraphRAG", ["knowledge graph", "graphrag", "graph rag", "kge", "graph retrieval"]),
        ("Tools & libraries", ["pytorch geometric", "pyg", "dgl", "release", "library", "graphstorm"]),
        ("Benchmarks & datasets", ["benchmark", "dataset", "leaderboard"]),
        ("Graph transformers & foundation models", ["graph transformer", "foundation model", "llm", "language model", "graphrag"]),
        ("Relational & heterogeneous graphs", ["heterogeneous", "relational", "knowledge graph"]),
        ("Scalability & systems", ["scalab", "distributed", "gpu", "billion", "sampling", "accelerat"]),
        ("Architectures & theory", ["expressiv", "over-squashing", "oversquashing", "oversmoothing", "theory",
                                    "spectral", "equivariant", "attention", "architecture"]),
    ]
    for cat, keys in rules:
        if any(has(t, k) for k in keys):
            return cat
    return "Applications"


def heuristic(cands):
    stories = []
    for c in cands:
        t = c["title"] + " " + c["raw"]
        stories.append({
            **{k: c[k] for k in ("id", "title", "url", "source", "authors", "published")},
            "category": guess_category(t),
            "heat": heat_score(t),
            "summary": c["raw"][:420] or "Open the paper for details.",
            "whyItMatters": "",
            "researchAngle": "",
        })
    return stories, ""


def auto_briefing(new):
    """Plain summary written from the day's results (used when Claude isn't scoring)."""
    if not new:
        return "No new items today. Below are the latest from previous days."
    top = max(new, key=lambda s: s.get("heat", 0))
    counts = {}
    for s in new:
        counts[s["category"]] = counts.get(s["category"], 0) + 1
    busiest, n = max(counts.items(), key=lambda kv: kv[1])
    lead = f"{len(new)} new {'item' if len(new) == 1 else 'items'} today."
    hot = f" Top signal: \u201c{top['title']}\u201d ({top['heat']}/10, {top['category']})."
    busy = f" Busiest topic: {busiest} ({n})." if n > 1 else ""
    return lead + hot + busy


# ---------- Claude scorer ----------
PROMPT = """You curate a daily graph learning digest for a researcher. Their interests, in order: (1) security and privacy OF graph models and graph data (adversarial/backdoor/poisoning attacks on GNNs, graph embeddings or knowledge graph embeddings; membership/attribute inference, link stealing, model extraction; de-anonymization of graph data; differentially private graph learning, statistics and synthetic graph release; graph unlearning; certified robustness; attacks on GraphRAG and LLM-on-graph systems), (2) graph learning applied to security and privacy (intrusion and provenance-graph detection, malware, fraud and AML, vulnerability mining), (3) important GNN work in general (architectures, theory, graph transformers and foundation models, KG embeddings, scalability, benchmarks, libraries).

Below are candidate items (JSON): mostly arXiv papers, plus library releases and news. Each has a "lane":
- "general": keep it only if its core contribution involves GNNs or a close neighbor (graph transformers, graph foundation models, KG embeddings, graph generative models, GraphRAG).
- "secpriv": keep it if it is genuinely about the security or privacy of graph learning or graph data, or uses graph learning (neural or not) for security or privacy. It does NOT need to involve GNNs.
Drop items where graphs are incidental (computation graphs, scene graphs, generic "graph" wording), classical graph algorithms with no learning or privacy angle, pure application papers with no methodological novelty unless security/privacy related, and duplicates.

For each kept item return:
- "id": copy from input
- "category": one of {cats}
- "heat": integer 1-10 for research signal. 9-10 = a new attack/defense class, a surprising negative result, or a field-shaping method/benchmark; 7-8 = strong, novel work in the researcher's core interests; 4-6 = solid incremental work or useful releases; 1-3 = minor.
- "summary": 2-3 plain sentences in your own words: what they did and the key result.
- "whyItMatters": 1 sentence on why this researcher should care.
- "researchAngle": 1-2 sentences proposing a concrete follow-up (e.g. attack it, defend it, transfer it to another graph type, audit its privacy, test its assumptions).

Also return a "briefing": 2-3 sentences on today's most important threads.

Reply with ONLY JSON: {{"briefing": "...", "stories": [ ... ]}}

Candidates:
{items}
"""


def with_claude(cands):
    import anthropic

    client = anthropic.Anthropic()
    items = [{"id": c["id"], "lane": c["lane"], "title": c["title"], "source": c["source"], "date": c["published"], "abstract": c["raw"]} for c in cands]
    msg = client.messages.create(
        model=MODEL,
        max_tokens=12000,
        messages=[{"role": "user", "content": PROMPT.format(cats=json.dumps(CATEGORIES), items=json.dumps(items, ensure_ascii=False))}],
    )
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    data = json.loads(text[text.find("{"): text.rfind("}") + 1])
    by_id = {c["id"]: c for c in cands}
    stories = []
    for s in data.get("stories", []):
        c = by_id.get(s.get("id"))
        if not c:
            continue
        stories.append({
            "id": c["id"], "title": c["title"], "url": c["url"], "source": c["source"],
            "authors": c["authors"], "published": c["published"],
            "category": s.get("category") if s.get("category") in CATEGORIES else guess_category(c["title"]),
            "heat": max(1, min(10, int(s.get("heat", 5)))),
            "summary": s.get("summary", ""),
            "whyItMatters": s.get("whyItMatters", ""),
            "researchAngle": s.get("researchAngle", ""),
        })
    return stories, data.get("briefing", "")


def main():
    old = {"briefing": "", "lastUpdated": "", "stories": []}
    if OUT_FILE.exists():
        try:
            old = json.loads(OUT_FILE.read_text())
        except json.JSONDecodeError:
            log("[warn] stories.json unreadable, starting fresh")
    existing = {s["id"]: s for s in old.get("stories", [])}
    for s_ in existing.values():
        s_["category"] = CATEGORY_RENAMES.get(s_.get("category"), s_.get("category"))
    seen = list(old.get("seen", []))  # ids already reviewed, including ones Claude dropped

    cands = collect(set(existing) | set(seen))
    log(f"{len(cands)} new candidates")

    new, briefing = [], ""
    if cands and os.environ.get("ANTHROPIC_API_KEY"):
        try:
            new, briefing = with_claude(cands)
        except Exception as ex:  # noqa: BLE001
            log(f"[warn] Claude scoring failed ({ex}); using keyword scorer")
            new, briefing = [], ""
    if cands and not new and not briefing:
        new, _ = heuristic(cands)
    if not briefing:
        briefing = auto_briefing(new)

    cutoff = (datetime.now(timezone.utc) - timedelta(days=KEEP_DAYS)).date().isoformat()
    merged = {**existing, **{s["id"]: s for s in new}}
    stories = sorted(
        (s for s in merged.values() if s.get("published", "") >= cutoff),
        key=lambda s: (s.get("published", ""), s.get("heat", 0)), reverse=True,
    )
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps({
        "lastUpdated": datetime.now(timezone.utc).date().isoformat(),
        "briefing": briefing,
        "stories": stories,
        "seen": (seen + [c["id"] for c in cands])[-5000:],
    }, indent=1, ensure_ascii=False))
    log(f"wrote {len(stories)} stories ({len(new)} new)")


if __name__ == "__main__":
    main()
