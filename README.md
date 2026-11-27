# dbtmpl.github.io

Personal website of David Biertimpel, built with [Jekyll](https://jekyllrb.com/) and deployed to GitHub Pages by
the workflow in `.github/workflows/pages.yml` on every push to `main`.

## Layout

| Path | Contents |
| --- | --- |
| `index.html` | Home page: hero, about, research, projects, latest posts |
| `_data/research.yml`, `_data/projects.yml` | Papers and projects shown on the home page |
| `blog/index.html` | Blog index (lists everything in `_posts/`) |
| `_posts/` | Blog posts, named `YYYY-MM-DD-slug.md` |
| `_drafts/` | Unpublished drafts (see `example-post.md` for math and code syntax) |
| `_posts/2019-10-01-n-step-actor-critic.md` | RL student-project article, served at `/rl_section.html` |
| `assets/js/squirrel.js` | The pixel-art squirrel next to the name |
| `assets/css/style.css` | All styles; colors are CSS variables at the top (with a dark mode) |

## Math

Write LaTeX between `$$ … $$`. Inline in a sentence it renders inline; on its own lines, separated by blank
lines, it renders as a display equation. Pages need `math: true` in the front matter (posts have it by default).
KaTeX renders it in the browser.

## Run locally

With Ruby 3.x:

```sh
bundle install
bundle exec jekyll serve --livereload --drafts
```

Or with Docker, without installing Ruby:

```sh
docker run --rm -it -p 4000:4000 -v "$PWD":/site -w /site ruby:3.4 \
  sh -c "bundle install && bundle exec jekyll serve --host 0.0.0.0 --drafts"
```

Then open http://localhost:4000.
