# area256.github.io

The area256 blog, built with Jekyll on GitHub Pages: <https://area256.github.io/>.

## Add a post

Create `_posts/YYYY-MM-DD-slug.md`:

```markdown
---
title: Your title
authors: [jp, someone]
description: One sentence shown under the title and in link previews.
math: true   # optional, loads MathJax for $...$ and $$...$$
---

Post body in Markdown.
```

It is published at `/YYYY/slug/` and listed on the home page.

## Add an author

Add an entry to `_data/authors.yml`; the key is what posts list under `authors`:

```yaml
someone:
  name: Full Name
  url: https://github.com/someone
```

## Preview locally

```sh
bundle install
bundle exec jekyll serve
```
