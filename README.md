# Edgewise: your daily graph learning digest

A free, self-hosted site that collects graph learning papers, library releases, and news every morning, scores each item for "signal" (1–10), and shows a summary, why it matters, and a research angle. It installs like an app.

## Two lanes

- **General lane (narrow):** GNNs and close neighbors: graph transformers, graph foundation models, knowledge graph embeddings, graph embeddings like node2vec, graph kernels, graph generative models, and GraphRAG.
- **Security & privacy lane (broad):** any graph learning or graph data work, neural or not, as long as it also has a security or privacy signal. This catches differentially private graph release, social network de-anonymization, knowledge graph poisoning, GraphRAG attacks, and provenance-graph intrusion detection, even when they never mention GNNs.

On busy days, 65% of the scoring slots go to the security & privacy lane so general papers can't crowd it out (and vice versa). Change this with `EDGEWISE_SECPRIV_SHARE`.

## How it works

- **GitHub Actions** runs `scripts/fetch.py` once a day. It reads the feeds in `scripts/feeds.txt` (mostly arXiv), sorts items into the two lanes, drops the rest, and saves `docs/data/stories.json`.
- **GitHub Pages** serves the `docs/` folder at `yourname.github.io/edgewise`. This is a project site, so it doesn't affect your existing `yourname.github.io` site or Redline.
- **Scoring:** with an Anthropic API key, Claude reads the abstracts, drops papers where "graph" is incidental (computation graphs, scene graphs, classical graph algorithms), and writes the scores, summaries, and research angles. Without a key, a free keyword scorer is used and summaries are the first part of each abstract. The broad lane is noisier in keyword mode, since it can't tell a GraphRAG attack from a backdoor in a model's "computation graph".

## Setup

Same as Redline:

1. Create a new **public** repo named `edgewise` (no README).
2. Upload `docs`, `scripts`, and `README.md` with the "uploading an existing file" link, then commit.
3. **Add file → Create new file**, name it `.github/workflows/daily.yml`, paste in the contents of that file from this folder, and commit.
4. **Settings → Actions → General → Workflow permissions →** Read and write → Save.
5. **Settings → Pages →** Deploy from a branch → `main`, `/docs` → Save.
6. (Optional) **Settings → Secrets and variables → Actions →** New secret `ANTHROPIC_API_KEY`. You can reuse a key or create a separate one to track costs per site.
7. **Actions → Daily digest → Run workflow.**

It then runs daily at 12:00 UTC, after arXiv's overnight announcement. arXiv doesn't announce on weekends, so weekend runs mostly pick up news and releases.

## Customize

- **Feeds:** `scripts/feeds.txt`. It includes cs.DB, cs.IR, cs.CL and cs.DS for the security & privacy lane; their non-security graph papers are filtered out. Add arXiv categories (e.g. `cs.CV` for point clouds, `q-bio.BM` for molecules), more Google News searches, or other release feeds. Prefix a name with `*` to keep every item from that feed without the GNN keyword filter.
- **General lane terms:** `TOPIC_TERMS` in `scripts/fetch.py`.
- **Security & privacy lane:** `BROAD_GRAPH_TERMS` (any graph wording) and `STRICT_SECPRIV` (the security/privacy terms that must also appear). Keep `STRICT_SECPRIV` specific; generic words like "robust" or "threat" let in noise.
- **What's hot:** `HOT_TERMS` (keyword scoring), `SECPRIV_TERMS` (ranking boost), and the interest list in `PROMPT` (Claude scoring). Edit the prompt to change priorities, e.g. put graph transformers first.
- **Categories:** `CATEGORIES` and the rules in `guess_category`.
- **Volume:** busy arXiv days can have 50+ matching papers. `EDGEWISE_MAX_CANDIDATES` (default 60) caps how many get scored, split between lanes by `EDGEWISE_SECPRIV_SHARE`.

## Notes

- Your API key stays private as a GitHub secret; the site only ever sees `stories.json`.
- If "Last refreshed" stops changing, check the Actions tab and re-enable the workflow.
