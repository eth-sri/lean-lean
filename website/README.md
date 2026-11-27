# LeanLean website

<https://leanleanbench.com/>

## Run it locally

```bash
python3 website/serve.py 8000
```

Then open <http://localhost:8000>. The site is plain HTML, CSS and JavaScript in `website/site/`,
with its data already built in `website/site/data/`. Pushes to `main` that touch `website/`
deploy it to GitHub Pages (`.github/workflows/website-pages.yml`).

## License

- **Code** (the website): MIT License, see `LICENSE`.
- **Data** (`site/data/`: results, leaderboard and agent traces):
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), except the third-party
  content below.
- **Benchmark repositories:** their code shown on the site, and the agents' edits to it,
  keep their own licenses; see each upstream repository, linked from the site.
- **System prompts:** the harnesses' built-in system prompts
  (`site/data/system-prompts.json`) belong to their makers and are reproduced for research.
- **Logos:** trademarks of their owners; see `site/assets/logos/README.md`.
