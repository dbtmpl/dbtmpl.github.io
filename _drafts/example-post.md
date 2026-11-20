---
title: Example post
description: One-line summary shown in the blog list.
---

This is a draft. Drafts are not published. To publish a post, move it to `_posts/`
and name it `YYYY-MM-DD-slug.md`, e.g. `_posts/2026-12-01-my-first-post.md`.

Preview drafts locally with `bundle exec jekyll serve --drafts`.

<!--more-->

## Math

Inline math uses double dollars: $$\nabla_\theta J(\theta)$$. A display equation
is a `$$` block on its own lines:

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[ \sum_{t=0}^{T} \gamma^t R_{t+1} \right]
$$

## Code

```python
import torch
print(torch.__version__)
```
