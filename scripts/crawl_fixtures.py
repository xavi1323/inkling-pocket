"""Run crawl4ai v0.9.6 markdown generation (no browser) on saved HTML -> expected outputs.

Usage: <crawl4ai venv python> -I scripts/crawl_fixtures.py <dir with *.html> [base_url_map.txt]
Writes <name>.expected.json next to each .html: raw, citations, references, fit (fit = PruningContentFilter defaults).
"""
import json, sys, pathlib, warnings
warnings.filterwarnings("ignore")
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import PruningContentFilter

folder = pathlib.Path(sys.argv[1])
bases = {}
if len(sys.argv) > 2:
    for line in open(sys.argv[2]):
        parts = line.split()
        if len(parts) >= 3:
            bases[parts[0]] = parts[2]
for html_path in sorted(folder.glob("*.html")):
    name = html_path.stem
    html = html_path.read_text(errors="replace")
    base = bases.get(name, "https://example.org/dir/page.html")
    gen = DefaultMarkdownGenerator(content_filter=PruningContentFilter())
    r = gen.generate_markdown(html, base_url=base)
    blocks = PruningContentFilter().filter_content(html)
    out = {"base_url": base, "raw": r.raw_markdown, "citations": r.markdown_with_citations,
           "references": r.references_markdown, "fit": r.fit_markdown, "fit_block_count": len(blocks)}
    (folder / f"{name}.expected.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(name, len(r.raw_markdown), len(r.fit_markdown), len(blocks))
