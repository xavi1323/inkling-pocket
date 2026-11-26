# Third-party notices

Inkling Pocket ports or links the projects below. Ported files keep their upstream license header.

| Project | Used for | License | Copyright |
|---|---|---|---|
| [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) @ `d998a62` | Inkling 2 model code ported to Swift (`InklingEngine/Model`, caches) | MIT | Copyright © 2025 Prince Canuma |
| [mlx-swift](https://github.com/ml-explore/mlx-swift) 0.33.0 | Array and GPU runtime (linked) | MIT | Copyright (c) 2023 ml-explore |
| [crawl4ai](https://github.com/unclecode/crawl4ai) v0.9.6 | Page cleaning and Markdown output ported to Swift (`CrawlCore`, `PocketCrawler`) | Apache-2.0 (with attribution requirement) | UncleCode |
| [OpenUI](https://github.com/thesysdev/openui) @ `7ce17a1` | Generative UI language and prompt, ported to SwiftUI (`OpenUICore`, `OpenUIViews`) | MIT | Copyright (c) 2011-2024 Thesys Inc. |
| [SwiftSoup](https://github.com/scinfu/SwiftSoup) 2.14.4 | HTML parsing (linked) | MIT | Copyright (c) 2009-2025 Jonathan Hedley; Copyright (c) 2016-2025 Nabil Chatbi (Swift port) |

Required by crawl4ai's license:

> This product includes software developed by UncleCode (https://x.com/unclecode) as part of the Crawl4AI project (https://github.com/unclecode/crawl4ai).

No GPL code is included. The HTML → Markdown writer (`CrawlCore/MarkdownWriter.swift`) is original code written for Inkling Pocket; it replaced an earlier html2text (GPL-3.0) port, and the html2text ASCII-substitution tables were removed with it. crawl4ai's own outputs are used only as test references.

The model weights (Inkling 2 Mini abliterated, MLX) are not part of this repository; their own license applies.

The full MIT and Apache-2.0 texts are in each upstream repository's `LICENSE` file at the versions above.
