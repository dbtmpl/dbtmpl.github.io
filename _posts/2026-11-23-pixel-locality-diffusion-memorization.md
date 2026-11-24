---
layout: post
title: "Scrambled Pixels and the Two Clocks of Diffusion Memorisation: An Interactive Exploration"
description: "An interactive visual tour of how diffusion models generalise before they memorise — using fixed pixel permutations to see why spatial locality opens the generalisation window from the left rather than blocking memorisation, and why fixed-budget onset curves can mislead."
kicker: Interactive explainer
date: 2026-11-23
image: /assets/diffusion_locality/preview.png
math: true
asset_base: /assets/diffusion_locality/
custom_css:
  - /assets/diffusion_locality/post.css
custom_js:
  - /assets/diffusion_locality/chart_kit.js
  - /assets/diffusion_locality/widget_hero.js
  - /assets/diffusion_locality/widget_arch.js
  - /assets/diffusion_locality/widget_ls_metric.js
  - /assets/diffusion_locality/widget_capacity_speed.js
  - /assets/diffusion_locality/widget_scaling_arch.js
  - /assets/diffusion_locality/main.js
---

<p class="lead-para">
Why do diffusion models trained on finite datasets generate novel images instead of simply photocopying their training set? A rich wave of recent theory frames the answer in terms of <strong>two training clocks</strong>: a fast feature-learning timescale \(\tau_{\text{gen}}\) and a much slower sample-memorisation timescale \(\tau_{\text{mem}} \propto n\,W^{-1}\) that grows linearly with dataset size \(n\) and inversely with network width \(W\).<sup class="cite"><a href="#ref-6">[6]</a>, <a href="#ref-13">[13]</a>, <a href="#ref-15">[15]</a>, <a href="#ref-25">[25]</a>, <a href="#ref-26">[26]</a></sup> As long as \(\tau_{\text{gen}} \ll \tau_{\text{mem}}\), early stopping lands inside a wide generalisation window.
</p>

When we also think about **spatial locality** in image models — convolutional $$3\times 3$$ kernels, local patch projections, and pixel correlations<sup class="cite"><a href="#ref-3">[3]</a>, <a href="#ref-4">[4]</a>, <a href="#ref-5">[5]</a>, <a href="#ref-12">[12]</a></sup> — it is tempting to picture locality as something that *blocks* or dramatically delays $$\tau_{\text{mem}}$$. Classic deep-learning experiments already hint that this intuition is upside-down: over-parameterised CNNs still fit pixel-scrambled images to zero training error, only with more optimisation inertia.<sup class="cite"><a href="#ref-8">[8]</a>, <a href="#ref-9">[9]</a>, <a href="#ref-10">[10]</a></sup>

To build concrete visual intuition for how those pieces fit together in diffusion models, I ran a set of controlled experiments across MNIST, FashionMNIST, and CIFAR-10 using U-Nets, permutation-equivariant MLPs, and Diffusion Transformers (DiTs), and built the interactive figures below to inspect the trajectories directly. Nothing here claims a new theoretical scaling law — instead, scrambling pixels acts as a clean pedagogical probe that makes three textbook ideas immediately visible:

1. **Locality opens the generalisation window from the left, not the right:** destroying spatial locality wrecks low-noise test denoising and purely local patch models, yet delays the U-Net's memorisation clock $$\tau_{\text{mem}}$$ by only $$\sim 2\times$$ ($$\rho \approx 1$$ octave).
2. **The fixed-budget onset trap:** because $$\tau_{\text{mem}}(n) \propto n^a$$, stopping training at any fixed step budget $$B$$ turns a pure optimisation slowdown $$\rho$$ into an *apparent* dataset-size shift $$\Delta(B) \approx -\rho/a$$.
3. **Architectural dials &amp; toy-scale boundaries:** varying U-Net width and DiT patch size illustrates Bonnaire et al.'s $$n^* \propto W$$ scaling<sup class="cite"><a href="#ref-13">[13]</a></sup> and exact $$p=1$$ token symmetry,<sup class="cite"><a href="#ref-21">[21]</a></sup> while highlighting why $$28\times 28$$ full-receptive-field runs probe early-layer optimisation bias rather than a hard geometric locality limit.

<!--more-->

<div class="viz-breakout" id="viz-hero"></div>

---

## 1. Background & Setup: Two Clocks and an Isometric Pixel Scramble

A diffusion model trained on $$n$$ images $$D = \{x_\mu\}_{\mu=1}^n \subset \mathbb{R}^d$$ under a cosine variance-preserving schedule ($$T = 1000$$ steps, signal retention $$\bar\alpha_t$$) sees noisy inputs

$$
x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, I_d).
$$

If a network minimises the empirical score-matching objective without constraint, the exact Bayes-optimal denoiser for the empirical delta-mixture over $$D$$ is a Boltzmann average — equivalent to Derrida's **Random Energy Model (REM)** in statistical physics:<sup class="cite"><a href="#ref-6">[6]</a>, <a href="#ref-7">[7]</a></sup>

$$
\hat{x}_0^{\text{emp}}(x_t, t) = \sum_{\mu=1}^n w_\mu(x_t, t)\, x_\mu, \qquad
w_\mu(x_t, t) \propto \exp\!\left(-\frac{\|x_t - \sqrt{\bar\alpha_t}\,x_\mu\|_2^2}{2(1 - \bar\alpha_t)}\right).
$$

As reverse sampling cools toward $$t \to 0$$, the empirical denoiser condenses onto a single training image $$\mu^*$$ and reproduces it verbatim. Recent work explains why real networks avoid this fate across two complementary axes:

- **Dynamical separation of timescales ("two clocks"):** Bonnaire et al.<sup class="cite"><a href="#ref-13">[13]</a></sup> and Urfin et al.<sup class="cite"><a href="#ref-25">[25]</a></sup> show that repeated noising in score matching splits the Neural Tangent Kernel spectrum into two bulks: large eigenvalues that drive fast learning of global population structure ($$\tau_{\text{gen}} \propto W^{-1}$$), and small sample-specific eigenvalues that delay memorisation to $$\tau_{\text{mem}} \propto n\,W^{-1}$$ (see also Favero et al.,<sup class="cite"><a href="#ref-15">[15]</a></sup> Buchanan et al.,<sup class="cite"><a href="#ref-27">[27]</a></sup> and Wang, Finn &amp; Liu's *innovation window* $$[\tau_{\text{rule}}, \tau_{\text{mem}}]$$<sup class="cite"><a href="#ref-26">[26]</a></sup>).
- **Spatial &amp; statistical locality:** Analytical models of generalising denoisers emphasise restricted local receptive fields or patch scores (Kamb &amp; Ganguli,<sup class="cite"><a href="#ref-3">[3]</a></sup> Niedoba et al.,<sup class="cite"><a href="#ref-17">[17]</a></sup> Hunt, Kamb &amp; Ganguli's BIRD<sup class="cite"><a href="#ref-4">[4]</a></sup>), while Lukoianov et al.<sup class="cite"><a href="#ref-5">[5]</a></sup> and Wang &amp; Pehlevan<sup class="cite"><a href="#ref-11">[11]</a></sup> show how natural-image covariance spectra and local convolutions jointly shape what gets learned first.

### Applying a fixed pixel permutation

To watch architectural locality interact with those clocks without changing global data geometry, I use the classic "shuffled pixels" intervention from Zhang et al.,<sup class="cite"><a href="#ref-8">[8]</a></sup> *Deep Image Prior*,<sup class="cite"><a href="#ref-9">[9]</a></sup> and Kadkhodaie et al.:<sup class="cite"><a href="#ref-12">[12]</a></sup> apply one fixed orthogonal permutation matrix $$P \in \{0,1\}^{d\times d}$$ to every image, $$x \mapsto Px$$. Because $$P^\top P = I_d$$, every pairwise Euclidean distance and global covariance eigenvalue stays **strictly invariant**:

$$
\|Px - Py\|_2 = \|x - y\|_2, \qquad \Sigma_{Px} = P \Sigma_x P^\top.
$$

Meanwhile, every $$3\times 3$$ convolutional window is torn into nine scattered, nearly uncorrelated coordinates across the canvas. Because $$P$$ is an $$L_2$$ isometry, the standard nearest-neighbour memorisation test from Yoon et al.,<sup class="cite"><a href="#ref-1">[1]</a></sup> Gu et al.,<sup class="cite"><a href="#ref-14">[14]</a></sup> and Bonnaire et al.<sup class="cite"><a href="#ref-13">[13]</a></sup> remains identically calibrated in both coordinate systems. Drawing 1,000 DDIM-50 samples from EMA weights, a generated sample is marked **memorised** if its nearest training distance $$d_1$$ is less than one third of its second-nearest training distance $$d_2$$:

$$
\frac{d_1}{d_2} < \frac{1}{3}.
$$

Unlike denoising PSNR or FID — where degraded image quality on scrambled inputs<sup class="cite"><a href="#ref-12">[12]</a></sup> can be mistaken for reduced memorisation — $$d_1/d_2$$ measures sample copying directly. From the memorised fraction $$m(n, \text{step})$$, it is helpful to track two quantities in octaves (base-2 logs):

- **Fixed-budget onset shift:** $$\Delta(B) = \log_2 n^*_{\text{perm}}(B) - \log_2 n^*_{\text{nat}}(B)$$, where $$n^*(B)$$ is the dataset size at which $$m$$ crosses $$0.5$$ at step budget $$B$$.
- **Memorisation-time slowdown:** $$\rho(n) = \log_2\!\big(\tau_{\text{perm}}(n) / \tau_{\text{nat}}(n)\big)$$, where $$\tau(n)$$ is the training step when $$m$$ first crosses $$0.5$$.

As a pipeline sanity check, every convolutional setup is paired with an exact permutation-equivariant control (a residual MLP, and later a patch-1 DiT with learned positional embeddings) where $$\Delta$$ and $$\rho$$ must vanish up to seed noise. The diagram below traces a real CIFAR-10 image before and after $$P$$ through forward diffusion, the purely convolutional `×1` U-Net (3.83M params, channels 32–64–128, no attention), and DDIM-50 sampling.

<div class="viz-breakout" id="viz-arch-unet"></div>

### Contrast with a strictly local patch machine

Before looking at trained U-Nets, it helps to see what would happen if a denoiser were *literally* restricted to fixed square patches throughout training. Following Kamb &amp; Ganguli,<sup class="cite"><a href="#ref-3">[3]</a></sup> I evaluated a network-free square-patch local-score (LS) estimator calibrated on natural MNIST. Under pixel permutation, that fixed square-patch machine collapses completely ($$\Delta_{\text{LS}} \le -3$$, at least an $$8\times$$ drop in $$n^*$$): each scattered patch locks onto a *different* training image, assembling un-copied mosaics with median $$d_1/d_2 \approx 0.90\text{--}0.99$$. As Hunt, Kamb &amp; Ganguli note in BIRD<sup class="cite"><a href="#ref-4">[4]</a></sup> (which itself uses data-adapted Wiener masks on MNIST/FashionMNIST that permute covariantly with the pixels), fixed-patch approximations capture trained networks early in training rather than in the late memorising regime — making $$\Delta_{\text{LS}} \le -3$$ a clean illustration of the extreme purely-local limit.

<div class="viz-breakout" id="viz-ls"></div>

<details class="deep-dive">
  <summary>Math note: Why the residual MLP and patch-1 DiT are exact symmetry controls under \(P\)</summary>
  <p>
    For a residual MLP with i.i.d. isotropic Gaussian input/output initialisations \(W_0, W_{\text{out}}\), applying \(x \mapsto Px\) is equivalent to relabelling \(\tilde{W}_0 = W_0 P^\top\) and \(\tilde{W}_{\text{out}} = P W_{\text{out}}\), leaving the initial weight distribution, forward pass, AdamW trajectory, and generated sample distribution identical up to \(P\).
  </p>
  <p>
    The exact same symmetry holds for a Diffusion Transformer with patch size \(p=1\) and i.i.d. normal learned positional embeddings \(E_{\text{pos}} \in \mathbb{R}^{d \times w}\) (cf. Tolstikhin et al.<sup class="cite"><a href="#ref-21">[21]</a></sup>): permuting pixels simply permutes the rows of \(E_{\text{pos}}\) (verified numerically in the DiT implementation to a maximum fp32 discrepancy of \(2.4 \times 10^{-7}\)).
  </p>
</details>

---

## 2. Memorisation Barely Slows Down — and Why Fixed Budgets Mislead

What happens when we train the 3.83M-parameter U-Net on natural versus permuted MNIST and FashionMNIST (3 seed replicates per arm, plus 3 independent random permutations on MNIST and the MLP symmetry control)?

1. **Memorisation still happens, delayed by roughly $$2\times$$:** On MNIST at $$n = 2048$$, natural and permuted U-Nets climb to essentially the same memorised fraction by 300k steps (`0.82` vs. `0.83`), just as *Deep Image Prior*<sup class="cite"><a href="#ref-9">[9]</a></sup> and Zhang et al.<sup class="cite"><a href="#ref-8">[8]</a></sup> would lead us to expect for an over-parameterised CNN whose bottleneck covers the full image. Crossing $$m = 0.5$$ simply takes **$$\approx 2.0\text{--}2.3\times$$ more gradient steps** on scrambled pixels: $$\rho(2048) = 1.18, 1.17, 1.06$$ across seed replicates and $$1.18, 1.07, 0.99$$ across three random permutations (while the MLP control stays centered near zero, $$\vert\rho_{\text{MLP}}\vert \le 0.28$$). On FashionMNIST at $$n = 2048$$, the permuted arm is still climbing at 300k (`m = 0.33–0.43`), consistent with a $$\ge 2\times$$ delay.
2. **At a fixed 100k budget, that delay projects onto a small horizontal shift:** Reading $$n^*$$ at 100k steps gives $$\Delta(100\text{k}) \in [-0.35, -0.41]$$ octaves on MNIST and $$[-0.36, -0.43]$$ on FashionMNIST (replicate spread $$\sigma = 0.08$$).

<div class="viz-breakout" id="viz-capacity"></div>

<div class="viz-breakout" id="viz-speed"></div>

### Algebra of the fixed-budget trap: $$\Delta(B) \approx -\rho / a$$

Why does a $$\sim 2\times$$ slowdown in training steps look like a $$-0.35$$ to $$-0.64$$ octave shift in dataset size $$n^*$$ when you slice horizontally at a fixed step budget $$B$$?

Wang &amp; Pehlevan<sup class="cite"><a href="#ref-11">[11]</a></sup> show analytically that while MLP diffusion dynamics depend only on global covariance eigenvalues $$\tau_k \propto 1/\lambda_k$$ (invariant under $$P$$), local convolutions redirect early learning into *patch-covariance* space (see also d'Ascoli et al.,<sup class="cite"><a href="#ref-10">[10]</a></sup> Arpit et al.,<sup class="cite"><a href="#ref-22">[22]</a></sup> Ingrosso &amp; Goldt,<sup class="cite"><a href="#ref-23">[23]</a></sup> and Wang et al.<sup class="cite"><a href="#ref-24">[24]</a></sup>). Meanwhile, Bonnaire et al.<sup class="cite"><a href="#ref-13">[13]</a></sup> and Favero et al.<sup class="cite"><a href="#ref-15">[15]</a></sup> establish that memorisation time grows as a power law $$\tau_{\text{nat}}(n) \propto n^a$$ (with $$a \approx 1$$ in the linear scaling regime).

Combining a multiplicative slowdown $$2^\rho$$ with $$\tau_{\text{nat}}(n) \propto n^a$$ immediately links the two observables:

$$
\tau_{\text{perm}}(n) \approx 2^\rho\,\tau_{\text{nat}}(n) \propto 2^\rho\,n^a \implies \Delta(B) = \log_2 n^*_{\text{perm}}(B) - \log_2 n^*_{\text{nat}}(B) \approx -\frac{\rho}{a}.
$$

On MNIST, the empirical log-log slope $$a = \mathrm{d}\log_2 \tau_{\text{nat}} / \mathrm{d}\log_2 n$$ steepens from $$a \approx 1.0$$ at small budgets to $$a \approx 1.7$$ around $$n = 1024\text{--}2048$$, so $$\Delta(B)$$ shrinks from $$-0.64$$ at 10k steps to $$-0.35$$ at 100k and $$-0.33$$ at 300k. Methodologically, this is a useful trap to keep in mind whenever reading empirical memorisation plots: **comparing $$n^*$$ across data transforms, optimisers, or architectures at a fixed step budget mixes optimisation speed with asymptotic capacity.**

<div class="callout callout--physics">
  <div class="callout__title">The real casualty of scrambling pixels: the generalisation clock \(\tau_{\text{gen}}\)</div>
  <p>
    Switch the right-hand panel below to <strong>Danger Zone: Per-t MSE Gap</strong> (motivated by low-noise memorisation analyses in Dodson et al.<sup class="cite"><a href="#ref-18">[18]</a></sup> and Kaiser &amp; Kollmann<sup class="cite"><a href="#ref-19">[19]</a></sup>). At low noise (\(t = 50\), \(n = 2048\)), the permuted U-Net suffers a nearly \(2\times\) larger test–train denoising gap at 100k steps (\(1.64\) vs. \(0.90\)) even as its \(d_1/d_2\) sample copying catches up — the exact denoising degradation highlighted by Kadkhodaie et al.<sup class="cite"><a href="#ref-12">[12]</a></sup>
  </p>
  <p>
    Viewed through the two-clock picture of Urfin et al.<sup class="cite"><a href="#ref-25">[25]</a></sup> and Wang, Finn &amp; Liu,<sup class="cite"><a href="#ref-26">[26]</a></sup> spatial locality does <em>not</em> protect diffusion models by pushing \(\tau_{\text{mem}}\) far to the right (scrambling only moves \(\tau_{\text{mem}}\) by \(\sim 2\times\)). Instead, convolutional locality pulls \(\tau_{\text{gen}}\) sharply to the <em>left</em>, opening the window \([\tau_{\text{gen}}, \tau_{\text{mem}}]\) where clean generalising score fields form long before sample copying sets in.
  </p>
</div>

<div class="viz-breakout" id="viz-delta-rho"></div>

---

## 3. Exploratory Probes: Width, Transformer Patch Size & CIFAR-10

To see how this picture behaves as we vary model scale, architecture, and image statistics, I also ran a set of **single-seed exploratory sweeps**. Because these runs use one seed per configuration and fixed 100k–300k budgets without per-architecture hyperparameter sweeps, they should be read as qualitative illustrations — along with instructive experimental caveats — rather than settled scaling measurements.

### U-Net width (`×½`, `×1`, `×2`) and Bonnaire scaling

Sweeping U-Net base channels across three widths on MNIST (100k steps, single seed):
- **`×½` (16–32–64, 1.15M params):** $$\log_2 n^*_{\text{nat}} = 10.30$$, $$\Delta(100\text{k}) = -0.65$$, mean $$\rho = 1.08$$ (on $$n \in \{256, 512\}$$)
- **`×1` (32–64–128, 3.83M params):** $$\log_2 n^*_{\text{nat}} = 11.42$$, $$\Delta(100\text{k}) = -0.35$$, mean $$\rho = 0.93$$ (on $$n \in \{512, 1024, 2048\}$$)
- **`×2` (64–128–256, 14.19M params):** $$\log_2 n^*_{\text{nat}} = 12.26$$, $$\Delta(100\text{k}) = +0.05$$, mean $$\rho = 0.26$$ (on $$n \in \{1024, 2048, 4096\}$$)

On natural images, fixed-budget $$n^*$$ grows by `+1.12` and `+0.84` octaves per width doubling — a clean textbook reproduction of Bonnaire et al.'s<sup class="cite"><a href="#ref-13">[13]</a></sup> scaling $$\tau_{\text{mem}} \propto n\,W^{-1}$$ (see also Gu et al.<sup class="cite"><a href="#ref-14">[14]</a></sup> and Marion &amp; Wu<sup class="cite"><a href="#ref-16">[16]</a></sup>): doubling width halves memorisation time, doubling the dataset size memorised within 100k steps.

Meanwhile, the measured permutation slowdown $$\rho$$ shrinks toward zero at `×2` (`1.08 → 0.93 → 0.26`). Intuitively, an heavily over-parameterised 14.2M-parameter U-Net on $$28\times 28$$ MNIST has abundant parallel channels to route scattered coordinates through its full-image bottleneck. At the same time, two experimental confounds sit right on the surface here: $$\rho$$ was evaluated on **non-overlapping $$n$$ ranges** across widths, and at `×2` natural memorisation occurs close enough to the earliest evaluation checkpoints that coarse checkpoint spacing can compress the apparent ratio.

### DiT patch size (`p = 1, 2, 4`): isolating where symmetry breaks

Moving from a U-Net to a Diffusion Transformer (DiT; ~4.7M params, width 192, depth 6, 6 heads, adaLN-Zero)<sup class="cite"><a href="#ref-2">[2]</a>, <a href="#ref-20">[20]</a></sup> makes it visually clear *where* spatial locality enters a transformer. Every self-attention block mixes all tokens globally from Layer 1, so spatial neighbourhood structure enters **only** at the $$p\times p$$ patchify linear projection and positional encoding. Switch $$p \in \{1, 2, 4\}$$ in the interactive schematic below to trace how permutation symmetry is preserved or broken.

<div class="viz-breakout" id="viz-arch-dit"></div>

On this single-seed MNIST probe ($$n \in \{128, 256\}$$, sharing the U-Net's `AdamW lr = 2e-4` schedule), the measured slowdown orders from the purely convolutional U-Net (`0.93`) down to `DiT p=2` (`0.55`), while `DiT p=4` (`-0.25`, within single-seed noise) sits near the exact `DiT p=1` symmetry null (`0` by construction).

Here too, keeping the caveats explicit is half the lesson:
- **Patch size bundles several changes together:** moving from $$p = 2$$ to $$p = 4$$ changes sequence length from 196 to 49 tokens, per-token embedding dimension, and how fixed 2D sin-cos positional encodings align with scrambled content.
- **Why cross-architecture $$n^*$$ cannot be read as capacity:** the DiTs' 100k-step $$n^*$$ sits ~2.8 octaves below the U-Net's on this shared learning rate — a textbook example of Section 2's fixed-budget confound when architectures have different optimisation speeds and untuned schedules, and when we have not jointly tracked a generalisation-quality clock $$\tau_{\text{gen}}$$.

### Colour photographs (CIFAR-10)

Repeating the `×1` U-Net probe on $$32\times 32$$ CIFAR-10 (single seed) produces a very similar trajectory: $$\Delta(100\text{k}) = -0.48$$ and mean $$\rho = 0.94$$ ($$1.07$$ at $$n=128$$, $$0.80$$ at $$n=256$$). Extending $$n \in \{128, 256, 512\}$$ to 300k steps narrows the $$n = 128$$ memorised-fraction gap from `0.70 vs. 1.00` to `0.93 vs. 1.00`, while $$n = 256$$ reaches `0.80 vs. 0.98` (still slowly climbing at 300k). Note that this permutation shuffled the 1,024 spatial pixel locations while keeping each pixel's `(R, G, B)` triplet intact — preserving local cross-channel colour correlation in the first convolution. Below you can inspect sixteen actual training images from the $$n = 128$$ CIFAR-10 subset alongside their 300k-step nearest-neighbour copies, followed by the combined exploratory dashboard.

<div class="viz-breakout" id="viz-cifar-train"></div>

<div class="viz-breakout" id="viz-scale-arch"></div>

---

## 4. Putting the Picture Together

Stepping back from the individual curves, these visualisations illustrate a coherent story that already sits across the deep-learning and statistical-physics literature:

| Phenomenon | Established theoretical &amp; empirical context | What we see in the interactive trajectories |
|---|---|---|
| **Over-parameterised CNNs still memorise scrambled pixels (~2× slower)** | Zhang et al.,<sup class="cite"><a href="#ref-8">[8]</a></sup> *Deep Image Prior*,<sup class="cite"><a href="#ref-9">[9]</a></sup> d'Ascoli et al.,<sup class="cite"><a href="#ref-10">[10]</a></sup> Wang &amp; Pehlevan<sup class="cite"><a href="#ref-11">[11]</a></sup> | On $$28\times 28$$ MNIST, natural and permuted U-Nets reach the same copying plateau (`~0.83`), with $$\tau_{\text{mem}}$$ delayed by $$\rho \approx 1.0\text{--}1.2$$ octaves while the MLP null stays at $$0$$. |
| **Locality widens $$[\tau_{\text{gen}}, \tau_{\text{mem}}]$$ via $$\tau_{\text{gen}}$$, not by blocking $$\tau_{\text{mem}}$$** | Kadkhodaie et al.,<sup class="cite"><a href="#ref-12">[12]</a></sup> Bonnaire et al.,<sup class="cite"><a href="#ref-13">[13]</a></sup> Urfin et al.,<sup class="cite"><a href="#ref-25">[25]</a></sup> Wang, Finn &amp; Liu<sup class="cite"><a href="#ref-26">[26]</a></sup> | Scrambling pixels barely shifts $$\tau_{\text{mem}}$$, yet collapses fixed-patch LS denoisers ($$\Delta_{\text{LS}} \le -3$$) and nearly doubles the low-noise test–train denoising MSE gap. |
| **Fixed-budget $$n^*$$ conflates speed with capacity ($$\Delta \approx -\rho/a$$)** | $$\tau_{\text{mem}} \propto n\,W^{-1}$$ in Bonnaire et al.,<sup class="cite"><a href="#ref-13">[13]</a></sup> Favero et al.,<sup class="cite"><a href="#ref-15">[15]</a></sup> Buchanan et al.<sup class="cite"><a href="#ref-27">[27]</a></sup> | A pure time multiplier $$2^\rho$$ creates a budget-dependent horizontal shift $$\Delta(B)$$ that shrinks as the $$\tau(n)$$ slope steepens, and reproduces `+1` octave per U-Net width doubling. |
| **Where locality lives across architectures &amp; scales** | Kamb &amp; Ganguli,<sup class="cite"><a href="#ref-3">[3]</a></sup> BIRD,<sup class="cite"><a href="#ref-4">[4]</a></sup> Lukoianov et al.,<sup class="cite"><a href="#ref-5">[5]</a></sup> An et al.,<sup class="cite"><a href="#ref-20">[20]</a></sup> Tolstikhin et al.<sup class="cite"><a href="#ref-21">[21]</a></sup> | In a DiT, $$p=1$$ with learned positions is strictly permutation-equivariant while $$p > 1$$ reintroduces patch locality; in a $$28\times 28$$ U-Net, full-image bottleneck receptive fields mean permutation tests early-layer optimisation bias rather than a hard geometric ceiling. |

That final geometric point sets the natural boundary of everything plotted on this page. On $$28\times 28$$ and $$32\times 32$$ images, a 3-level U-Net's bottleneck receptive field spans the entire canvas — so an over-parameterised network has no trouble routing arbitrary long-range pixel permutations once given enough gradient steps. On high-resolution images where convolutional receptive fields remain strictly smaller than the scene (or when tracking the full two-clock Pareto frontier of sample quality $$\tau_{\text{gen}}$$ against memorisation $$\tau_{\text{mem}}$$), architectural locality can bind much more tightly than early-layer optimisation inertia alone.

---

## References

<ol class="refs">
  <li id="ref-1">Yoon, T., Choi, J. Y., Kwon, S., &amp; Ryu, E. K. (2023). <em>Diffusion Probabilistic Models Generalize when They Fail to Memorize.</em> ICML 2023 Workshop on Structured Probabilistic Inference &amp; Generative Modeling (SPIGM). <a href="https://icml.cc/virtual/2023/28053">icml.cc/virtual/2023/28053</a></li>
  <li id="ref-2">Zhang, H., Zhou, J., Lu, Y., Guo, M., Wang, P., Shen, L., &amp; Qu, Q. (2023). <em>The Emergence of Reproducibility and Generalizability in Diffusion Models.</em> <a href="https://arxiv.org/abs/2310.05264">arXiv:2310.05264</a></li>
  <li id="ref-3">Kamb, M., &amp; Ganguli, S. (2024). <em>An Analytic Theory of Creativity in Convolutional Diffusion Models.</em> <a href="https://arxiv.org/abs/2412.20292">arXiv:2412.20292</a></li>
  <li id="ref-4">Hunt, B., Kamb, M., &amp; Ganguli, S. (2026). <em>An Exact Information Theory of Generalization Phase Transitions in Bayesian Diffusion Models (BIRD).</em> <a href="https://arxiv.org/abs/2607.08041">arXiv:2607.08041</a></li>
  <li id="ref-5">Lukoianov, A., Yuan, C., Solomon, J., &amp; Sitzmann, V. (2025). <em>Locality in Image Diffusion Models Emerges from Data Statistics.</em> <a href="https://arxiv.org/abs/2509.09672">arXiv:2509.09672</a></li>
  <li id="ref-6">Biroli, G., Bonnaire, T., de Bortoli, V., &amp; Mézard, M. (2024). <em>Dynamical Regimes of Diffusion Models.</em> Nature Communications, 2024. <a href="https://arxiv.org/abs/2402.18491">arXiv:2402.18491</a></li>
  <li id="ref-7">Achilli, B., Ambrogioni, L., Lucibello, C., Mézard, M., &amp; Ventura, E. (2025). <em>Memorization and Generalization in Generative Diffusion under the Manifold Hypothesis.</em> <a href="https://arxiv.org/abs/2502.09578">arXiv:2502.09578</a></li>
  <li id="ref-8">Zhang, C., Bengio, S., Hardt, M., Recht, B., &amp; Vinyals, O. (2016). <em>Understanding Deep Learning Requires Rethinking Generalization.</em> ICLR 2017. <a href="https://arxiv.org/abs/1611.03530">arXiv:1611.03530</a></li>
  <li id="ref-9">Ulyanov, D., Vedaldi, A., &amp; Lempitsky, V. (2017). <em>Deep Image Prior.</em> CVPR 2018. <a href="https://arxiv.org/abs/1711.10925">arXiv:1711.10925</a></li>
  <li id="ref-10">d'Ascoli, S., Sagun, L., Bruna, J., &amp; Biroli, G. (2019). <em>Finding the Needle in the Haystack with Convolutions: On the Benefits of Architectural Bias.</em> NeurIPS 2019. <a href="https://arxiv.org/abs/1906.06766">arXiv:1906.06766</a></li>
  <li id="ref-11">Wang, B., &amp; Pehlevan, C. (2025). <em>An Analytical Theory of Spectral Bias in the Learning Dynamics of Diffusion Models.</em> <a href="https://arxiv.org/abs/2503.03206">arXiv:2503.03206</a></li>
  <li id="ref-12">Kadkhodaie, Z., Guth, F., Simoncelli, E. P., &amp; Mallat, S. (2023). <em>Generalization in Diffusion Models Arises from Geometry-Adaptive Harmonic Representations.</em> ICLR 2024. <a href="https://arxiv.org/abs/2310.02557">arXiv:2310.02557</a></li>
  <li id="ref-13">Bonnaire, T., Urfin, R., Biroli, G., &amp; Mézard, M. (2025). <em>Why Diffusion Models Don't Memorize: The Role of Implicit Dynamical Regularization in Training.</em> NeurIPS 2025 (Oral). <a href="https://arxiv.org/abs/2505.17638">arXiv:2505.17638</a></li>
  <li id="ref-14">Gu, X., Du, C., Pang, T., Li, C., Lin, M., &amp; Wang, Y. (2023). <em>On Memorization in Diffusion Models.</em> <a href="https://arxiv.org/abs/2310.02664">arXiv:2310.02664</a></li>
  <li id="ref-15">Favero, A., Sclocchi, A., &amp; Wyart, M. (2025). <em>Bigger Isn't Always Memorizing: Early Stopping Overparameterized Diffusion Models.</em> ICML 2025 Workshop. <a href="https://arxiv.org/abs/2505.16959">arXiv:2505.16959</a></li>
  <li id="ref-16">Marion, P., &amp; Wu, A. (2026). <em>Understanding Diffusion Models Requires Rethinking (Again) Generalization.</em> <a href="https://arxiv.org/abs/2605.06077">arXiv:2605.06077</a></li>
  <li id="ref-17">Niedoba, M., Zwartsenberg, B., Murphy, K., &amp; Wood, F. (2024). <em>Towards a Mechanistic Explanation of Diffusion Model Generalization.</em> ICML 2025. <a href="https://arxiv.org/abs/2411.19339">arXiv:2411.19339</a></li>
  <li id="ref-18">Dodson, D., Gao, J., Wang, Z., Wang, S., &amp; Wan, Z. (2026). <em>Two Calm Ends and the Wild Middle: A Geometric Picture of Memorization in Diffusion Models.</em> <a href="https://arxiv.org/abs/2602.17846">arXiv:2602.17846</a></li>
  <li id="ref-19">Kaiser, T., &amp; Kollmann, R. (2026). <em>Beyond Sample Copying: Structural Memorization in Diffusion Models.</em> <a href="https://arxiv.org/abs/2603.13419">arXiv:2603.13419</a></li>
  <li id="ref-20">An, J., Wang, Y., Guo, R., Luo, J., &amp; Schwing, A. (2024). <em>On Inductive Biases that Enable Generalization of Diffusion Transformers.</em> <a href="https://arxiv.org/abs/2410.21273">arXiv:2410.21273</a></li>
  <li id="ref-21">Tolstikhin, I., et al. (2021). <em>MLP-Mixer: An All-MLP Architecture for Vision.</em> NeurIPS 2021 (§3.4 on input-permutation invariance). <a href="https://arxiv.org/abs/2105.01601">arXiv:2105.01601</a></li>
  <li id="ref-22">Arpit, D., et al. (2017). <em>A Closer Look at Memorization in Deep Networks.</em> ICML 2017. <a href="https://arxiv.org/abs/1706.05394">arXiv:1706.05394</a></li>
  <li id="ref-23">Ingrosso, A., &amp; Goldt, S. (2022). <em>Data-Driven Emergence of Convolutional Structure in Neural Networks.</em> PNAS 2022. <a href="https://arxiv.org/abs/2202.00565">arXiv:2202.00565</a></li>
  <li id="ref-24">Wang, Z., Wan, Z., Belkin, M., &amp; Wang, S. (2025). <em>Seeds of Structure: Patch PCA Reveals Universal Compositional Cues in Diffusion Models.</em> NeurIPS 2025. <a href="https://neurips.cc/virtual/2025/poster/119110">neurips.cc/virtual/2025/poster/119110</a></li>
  <li id="ref-25">Urfin, R., Bonnaire, T., Biroli, G., &amp; Mézard, M. (2026). <em>First Learn, Then Memorize: The Spectral Bias of Diffusion Models.</em> <a href="https://arxiv.org/abs/2609.35377">arXiv:2609.35377</a></li>
  <li id="ref-26">Wang, B., Finn, E. L. B., &amp; Liu, B. (2026). <em>The Two Clocks and the Innovation Window: When and How Generative Models Learn Rules.</em> <a href="https://arxiv.org/abs/2605.10019">arXiv:2605.10019</a></li>
  <li id="ref-27">Buchanan, S., Pai, D., Ma, Y., &amp; De Bortoli, V. (2025). <em>On the Edge of Memorization in Diffusion Models.</em> <a href="https://arxiv.org/abs/2508.17689">arXiv:2508.17689</a></li>
</ol>
