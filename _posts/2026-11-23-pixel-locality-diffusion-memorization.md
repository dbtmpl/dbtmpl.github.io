---
layout: post
title: "When Does Spatial Locality Matter in Diffusion Memorisation? Width, Patch Size, and the 2× Slowdown"
description: "Over-parameterised CNNs eventually memorise pixel-scrambled images — that much is already known. Here we measure the exact diffusion memorisation slowdown (ρ) and explore how it vanishes with U-Net width and dials smoothly with DiT patch size."
kicker: Research note
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
Two facts about memorisation in image models are already well established. First, over-parameterised CNNs will still memorise images whose pixels have been shuffled by a fixed random permutation — they just fit them more slowly.<sup class="cite"><a href="#ref-8">[8]</a>, <a href="#ref-9">[9]</a>, <a href="#ref-12">[12]</a></sup> Second, in diffusion models, memorisation is a dynamical condensation whose onset scales with dataset size and model width, \(\tau_{\text{mem}} \propto n\,W^{-1}\).<sup class="cite"><a href="#ref-6">[6]</a>, <a href="#ref-13">[13]</a>, <a href="#ref-15">[15]</a></sup>
</p>

Taking those results as our starting point, we wanted to probe a narrower and less charted question: **how large is the spatial-locality slowdown in diffusion memorisation time $$\rho = \log_2(\tau_{\text{perm}} / \tau_{\text{nat}})$$, and what controls it?** Specifically, does a wider U-Net outgrow its reliance on local $$3\times 3$$ convolutions, and can you dial the slowdown continuously by moving to a Diffusion Transformer (DiT) and varying patch size $$p \in \{1, 2, 4\}$$? Explore our real MNIST and CIFAR-10 outputs below before we unpack the method and what turned up.

<!--more-->

<div class="viz-breakout" id="viz-hero"></div>

---

## 1. Setup & Method: Turning Pixel Permutation Into a Clean Clock

A diffusion model trained on $$n$$ images $$D = \{x_\mu\}_{\mu=1}^n \subset \mathbb{R}^d$$ under a cosine variance-preserving process ($$T = 1000$$ steps, signal retention $$\bar\alpha_t$$) sees noisy inputs

$$
x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, I_d).
$$

For the empirical delta-mixture over $$D$$, the exact Bayes-optimal denoiser is a Boltzmann average — a **Random Energy Model (REM)** in statistical physics:<sup class="cite"><a href="#ref-6">[6]</a>, <a href="#ref-7">[7]</a></sup>

$$
\hat{x}_0^{\text{emp}}(x_t, t) = \sum_{\mu=1}^n w_\mu(x_t, t)\, x_\mu, \qquad
w_\mu(x_t, t) \propto \exp\!\left(-\frac{\|x_t - \sqrt{\bar\alpha_t}\,x_\mu\|_2^2}{2(1 - \bar\alpha_t)}\right).
$$

As reverse sampling cools toward $$t \to 0$$, the exact empirical denoiser condenses onto a single training point $$\mu^*$$ and photocopies the dataset. Real neural networks avoid collapsing when $$n$$ is large enough, and recent work points to two distinct sources of locality: **architectural patch locality** (convolutional or patch-restricted local scores)<sup class="cite"><a href="#ref-3">[3]</a>, <a href="#ref-4">[4]</a>, <a href="#ref-17">[17]</a></sup> versus **data-covariance locality** (natural-image pixel correlations that shape sensitivity fields across both U-Nets and DiTs).<sup class="cite"><a href="#ref-5">[5]</a></sup>

### The intervention and two observables

Apply one fixed orthogonal permutation matrix $$P \in \{0,1\}^{d\times d}$$ to every training and test image, $$x \mapsto Px$$ (the same "shuffled pixels" transform used by Zhang et al.,<sup class="cite"><a href="#ref-8">[8]</a></sup> *Deep Image Prior*,<sup class="cite"><a href="#ref-9">[9]</a></sup> and Kadkhodaie et al.<sup class="cite"><a href="#ref-12">[12]</a></sup>). Because $$P^\top P = I_d$$, every Euclidean distance and covariance eigenvalue stays **strictly invariant**:

$$
\|Px - Py\|_2 = \|x - y\|_2, \qquad \Sigma_{Px} = P \Sigma_x P^\top.
$$

Meanwhile, every $$3\times 3$$ convolutional patch shatters into nine scattered, nearly uncorrelated coordinates. Crucially, Euclidean invariance keeps our memorisation metric unbiased across coordinate systems. Following Yoon et al.,<sup class="cite"><a href="#ref-1">[1]</a></sup> Gu et al.,<sup class="cite"><a href="#ref-14">[14]</a></sup> and Bonnaire et al.,<sup class="cite"><a href="#ref-13">[13]</a></sup> we draw 1,000 DDIM-50 samples from EMA weights and call a sample **memorised** if its nearest training distance $$d_1$$ is less than one third of its second-nearest distance $$d_2$$:

$$
\frac{d_1}{d_2} < \frac{1}{3}.
$$

Unlike denoising PSNR or FID — where worse denoising on scrambled images<sup class="cite"><a href="#ref-12">[12]</a></sup> can masquerade as less memorisation — $$d_1/d_2$$ is identical in natural and permuted space. From the memorised fraction $$m(n, \text{step})$$, we read two quantities in octaves (powers of two):

- **Fixed-budget onset shift:** $$\Delta(B) = \log_2 n^*_{\text{perm}}(B) - \log_2 n^*_{\text{nat}}(B)$$, where $$n^*(B)$$ is the dataset size at which $$m$$ crosses $$0.5$$ at training budget $$B$$.
- **Memorisation-time slowdown:** $$\rho(n) = \log_2\!\big(\tau_{\text{perm}}(n) / \tau_{\text{nat}}(n)\big)$$, where $$\tau(n)$$ is the training step when $$m$$ first crosses $$0.5$$.

To anchor both measurements, we pair every convolutional run with an exact permutation-equivariant control (a residual MLP, and later a patch-1 DiT with learned positions) where both $$\Delta$$ and $$\rho$$ must vanish up to finite-seed noise. The schematic below traces a real CIFAR-10 image before and after $$P$$ through forward noising, our purely convolutional U-Net (3.83M params, channels 32–64–128, no attention), and the reverse DDIM-50 loop.

<div class="viz-breakout" id="viz-arch-unet"></div>

As a purely local theoretical anchor before training neural networks, we also ran a network-free square-patch local-score (LS) model calibrated on natural MNIST following Kamb &amp; Ganguli.<sup class="cite"><a href="#ref-3">[3]</a></sup> Under pixel permutation, that fixed square-patch machine collapses completely ($$\Delta_{\text{LS}} \le -3$$): each scattered patch locks onto a *different* training image, producing un-copied mosaics with median $$d_1/d_2 \approx 0.90\text{--}0.99$$. As Hunt, Kamb &amp; Ganguli note for BIRD,<sup class="cite"><a href="#ref-4">[4]</a></sup> fixed-patch machines track trained networks best early in training and diverge later, so $$\Delta_{\text{LS}} \le -3$$ marks the extreme purely-local limit rather than a late-training U-Net prediction.

<div class="viz-breakout" id="viz-ls"></div>

<details class="deep-dive">
  <summary>Math note: Why the residual MLP and patch-1 DiT are exact symmetry nulls under \(P\)</summary>
  <p>
    For a residual MLP with i.i.d. isotropic Gaussian input/output initialisations \(W_0, W_{\text{out}}\), applying \(x \mapsto Px\) is equivalent to relabelling \(\tilde{W}_0 = W_0 P^\top\) and \(\tilde{W}_{\text{out}} = P W_{\text{out}}\), leaving the initial weight distribution, forward pass, AdamW trajectory, and generated sample distribution identical up to \(P\).
  </p>
  <p>
    The exact same symmetry holds for a Diffusion Transformer with patch size \(p=1\) and i.i.d. normal learned positional embeddings \(E_{\text{pos}} \in \mathbb{R}^{d \times w}\) (analogous to Tolstikhin et al.<sup class="cite"><a href="#ref-21">[21]</a></sup>): permuting pixels simply permutes the rows of \(E_{\text{pos}}\) (verified numerically in our DiT implementation to max fp32 discrepancy \(2.4 \times 10^{-7}\)).
  </p>
</details>

---

## 2. The Baseline Clock: A Replicated ~2× Slowdown and the Fixed-Budget Trap

Before varying width or architecture, we measured $$\Delta$$ and $$\rho$$ on our base `×1` U-Net (3.83M params) across MNIST and FashionMNIST (3 replicates per arm, `rep0`/`repA`/`repB`, plus 3 independent random permutations on MNIST and the MLP null control).

Two numbers summarise the baseline cleanly:

1. **At 100k steps, the onset shift is small but nonzero:** $$\Delta(100\text{k}) \in [-0.35, -0.41]$$ on MNIST and $$[-0.36, -0.43]$$ on FashionMNIST against a replicate noise floor of $$\sigma = 0.08$$ (all six replicates inside our pre-registered $$\pm 0.5$$ equivalence band, with the MLP null passing at $$\vert\Delta_{\text{MLP}}\vert \le 0.37$$ on MNIST and $$\le 0.29$$ on FashionMNIST's finer grid).
2. **In training time, memorisation slows down by ~2×:** on MNIST at $$n = 2048$$, natural and permuted U-Nets reach the same plateau (`0.82` vs. `0.83` at 300k), yet the permuted arm takes **$$\approx 2.0\text{--}2.3\times$$ longer** to reach $$m = 0.5$$: $$\rho(2048) = 1.18, 1.17, 1.06$$ across replicates, and $$1.18, 1.07, 0.99$$ across three independent permutations (while $$\vert\rho_{\text{MLP}}\vert \le 0.28$$). On FashionMNIST at $$n = 2048$$, the permuted arm is still climbing at 300k (`m = 0.33–0.43`), giving censored one-sided lower bounds $$\rho > 0.97\text{--}1.20$$.

<div class="viz-breakout" id="viz-capacity"></div>

<div class="viz-breakout" id="viz-speed"></div>

### Why a speed gap looks like a capacity shift at any fixed budget

Earlier deep-learning work established that CNNs suffer optimisation inertia on permuted pixels<sup class="cite"><a href="#ref-9">[9]</a>, <a href="#ref-10">[10]</a>, <a href="#ref-22">[22]</a></sup> and that diffusion learning times scale inversely with covariance eigenvalues $$\tau_k \propto 1/\lambda_k$$ — invariant for an MLP, but redirected into *patch-covariance* space by local convolutions<sup class="cite"><a href="#ref-11">[11]</a>, <a href="#ref-24">[24]</a></sup> (whereas non-Gaussian statistics can eventually induce local features even in FC nets<sup class="cite"><a href="#ref-23">[23]</a></sup>). Meanwhile, Bonnaire et al.<sup class="cite"><a href="#ref-13">[13]</a></sup> and Favero et al.<sup class="cite"><a href="#ref-15">[15]</a></sup> showed that diffusion memorisation time scales as a power law $$\tau_{\text{nat}}(n) \propto n^a$$ (with $$a \approx 1$$).

Put those two facts together: if pixel permutation multiplies the memorisation clock by $$2^\rho$$, then stopping both arms at any fixed training budget $$B$$ automatically produces an *apparent* capacity shift:

$$
\tau_{\text{perm}}(n) \approx 2^\rho\,\tau_{\text{nat}}(n) \propto 2^\rho\,n^a \implies \Delta(B) = \log_2 n^*_{\text{perm}}(B) - \log_2 n^*_{\text{nat}}(B) \approx -\frac{\rho}{a}.
$$

On MNIST, the local log-log slope $$a = \mathrm{d}\log_2 \tau_{\text{nat}} / \mathrm{d}\log_2 n$$ steepens from $$a \approx 1.0$$ early on to $$a \approx 1.7$$ around $$n = 1024\text{--}2048$$. That immediately explains why $$\Delta(B)$$ shrinks from $$-0.64$$ at 10k steps toward $$-0.35$$ at 100k — and serves as a practical methodological warning: **fixed-budget $$n^*$$ comparisons across data transforms or architectures entangle memorisation speed with capacity.**

<div class="callout callout--physics">
  <div class="callout__title">Low-noise denoising error &amp; plateau scope</div>
  <p>
    Switch the right panel below to <strong>Danger Zone: Per-t MSE Gap</strong> (cf. Dodson et al.<sup class="cite"><a href="#ref-18">[18]</a></sup> and Kaiser &amp; Kollmann<sup class="cite"><a href="#ref-19">[19]</a></sup>): at low noise (\(t = 50\)), the permuted U-Net retains a much larger test–train denoising gap (\(1.64\) vs. \(0.90\) at 100k, \(n = 2048\)) even as its \(d_1/d_2\) memorisation catches up, echoing Kadkhodaie et al.<sup class="cite"><a href="#ref-12">[12]</a></sup>
  </p>
  <p>
    <strong>Plateau scope:</strong> Our runs did not reach an asymptotic plateau at \(n \ge 4096\) (at 300k steps, \(n=4096\) sits at \(m_{\text{nat}} = 0.47\) vs. \(m_{\text{perm}} = 0.26\), and \(\Delta\) sits at \(-0.33\)). Thus "onset" here always means fixed-budget onset rather than an infinite-training ceiling.
  </p>
</div>

<div class="viz-breakout" id="viz-delta-rho"></div>

---

## 3. Finding 1 — Model Width Shrinks the Slowdown

With the baseline clock $$\rho \approx 1$$ calibrated on the `×1` U-Net, we asked the first question that existing literature does not answer: **does the spatial-locality slowdown stay constant as you scale the U-Net's width, or does a wider network outgrow it?**

We swept U-Net base channels across three widths on MNIST (single seed per width, 100k steps):
- **`×½` (16–32–64, 1.15M params):** $$\log_2 n^*_{\text{nat}} = 10.30$$, $$\Delta(100\text{k}) = -0.65$$, mean **$$\rho = 1.08$$**
- **`×1` (32–64–128, 3.83M params):** $$\log_2 n^*_{\text{nat}} = 11.42$$, $$\Delta(100\text{k}) = -0.35$$, mean **$$\rho = 0.93$$**
- **`×2` (64–128–256, 14.19M params):** $$\log_2 n^*_{\text{nat}} = 12.26$$, $$\Delta(100\text{k}) = +0.05$$, mean **$$\rho = 0.26$$**

Two distinct effects appear in that sweep — one expected, one surprising:

1. **Natural onset scales by ~1 octave per width doubling (`+1.12` and `+0.84` octaves).** That directly reproduces Bonnaire et al.'s<sup class="cite"><a href="#ref-13">[13]</a></sup> scaling $$\tau_{\text{mem}} \propto n\,W^{-1}$$ (and qualitatively matches Gu et al.<sup class="cite"><a href="#ref-14">[14]</a></sup> and Marion &amp; Wu<sup class="cite"><a href="#ref-16">[16]</a></sup>): doubling width halves memorisation time, which doubles $$n^*$$ at a fixed budget.
2. **The permutation slowdown $$\rho$$ drops by 4× as width quadruples (`1.08 → 0.93 → 0.26`),** and $$\Delta(100\text{k})$$ vanishes alongside it via $$-\rho/a$$ (`-0.65 → -0.35 → +0.05`).

### What could this mean?

If the collapse of $$\rho$$ with width holds up under multi-seed replication, it points to a crisp physical picture: **convolutional locality acts primarily as an optimisation crutch for capacity-constrained networks.** When a U-Net has few channels (`1.15M`), sharing $$3\times 3$$ spatial weights across coherent neighbourhoods provides a large speed boost — removing that correlation penalises memorisation by more than $$2\times$$. Give the same 3-level U-Net four times the channel width (`14.19M` parameters, roughly $$3.7\times$$ more weights per doubling), and it has enough parallel feature channels to route scattered pixel pairs through its full-image bottleneck almost as rapidly as contiguous edges.

**Two confounds must be checked before treating this as settled:**
- **Non-overlapping $$n$$ ranges:** because wider models memorise larger datasets at 100k, mean $$\rho$$ was read on $$n \in \{256, 512\}$$ at `×½` versus $$n \in \{1024, 2048, 4096\}$$ at `×2`, and $$\rho(n)$$ itself varies with $$n$$.
- **Early-step resolution at `×2`:** at 14.2M parameters on small MNIST subsets, both arms memorise quickly enough that $$\tau_{\text{nat}}$$ sits close to our earliest evaluation checkpoints (and both `×½` capacity and `×2` speed are single-seed exploratory runs). Running 3 seeds on a **matched $$n \in \{512, 1024\}$$ grid** is therefore the highest-value follow-up from this entire project.

---

## 4. Finding 2 — Dialling Locality Through Transformer Patch Size

Our second novel axis replaces the binary CNN-vs-MLP contrast with a **continuous architectural dial**. While U-Nets and transformers already diverge in sample reproducibility,<sup class="cite"><a href="#ref-2">[2]</a></sup> in a Diffusion Transformer (DiT)<sup class="cite"><a href="#ref-20">[20]</a></sup> every self-attention layer mixes all tokens globally from Layer 1 — meaning spatial locality enters *only* at the $$p\times p$$ patchify projection and positional encoding.

We trained ~4.7M-parameter adaLN-Zero DiTs (width 192, depth 6, 6 heads) on MNIST while varying patch size $$p \in \{1, 2, 4\}$$ alongside our convolutional U-Net (`3.83M`). Use the interactive architecture diagram below to switch $$p \in \{1, 2, 4\}$$ and see exactly where permutation symmetry breaks.

<div class="viz-breakout" id="viz-arch-dit"></div>

Across architectures, the measured memorisation slowdown $$\rho$$ orders cleanly with the strength of the local patch prior:

$$
\text{Conv U-Net } (0.93) \;>\; \text{DiT } p=2 \; (0.55) \;>\; \text{DiT } p=4 \; (\approx -0.25) \;\approx\; \text{DiT } p=1 \; (0 \text{ exact}).
$$

Meanwhile, fixed-budget onset stays inside the no-shift band for both DiTs ($$\Delta = -0.39$$ for $$p=2$$; $$+0.04$$ for $$p=4$$).

### What does this ordering tell us?

- **At $$p = 1$$ with learned positional embeddings (exact null):** each pixel is its own token. Permuting pixels merely reorders the token sequence and learned position rows — giving exact permutation equivariance ($$\rho = 0$$, $$\Delta = 0$$).
- **At $$p = 2$$ with fixed 2D sin-cos positions ($$\rho = 0.55$$):** grouping pixels into $$2\times 2$$ local patches reintroduces roughly *half* of the U-Net's convolutional slowdown. Scrambling pixels breaks those 4-pixel patches and breaks the alignment between pixel content and the fixed 2D sin-cos grid, yet global self-attention in all 6 blocks compensates much faster than a stack of local $$3\times 3$$ convolutions.
- **At $$p = 4$$ with fixed 2D sin-cos positions ($$\rho \approx -0.25$$):** with only 49 coarse $$4\times 4$$ tokens on MNIST, each token's linear projection already aggregates 16 pixels and global attention dominates — the local-patch speed advantage vanishes into single-seed noise.

**Important caveats on interpreting the DiT sweep:**
- This is a single-seed exploratory sweep with $$\rho$$ measured at only two dataset sizes ($$n = 128, 256$$, where $$n = 128$$ was added after the initial grid left DiT speed undefined), and the DiTs shared the U-Net's untuned optimiser schedule (`AdamW lr = 2e-4`, 100k steps).
- Patch size is not a pure single-knob dial: moving from $$p = 2$$ to $$p = 4$$ simultaneously reduces sequence length from 196 to 49 tokens, changes per-token projection width, and alters how fixed sin-cos positions interact with permuted inputs.
- Finally, although the DiTs' 100k-step $$n^*$$ sits ~2.8 octaves below the U-Net's, Section 2's identity warns against reading that fixed-budget gap as a pure capacity difference — without tuning learning rates and training each architecture to plateau, cross-architecture $$n^*$$ mixes optimisation speed with capacity.

---

## 5. Checking Natural Colour Images (CIFAR-10) & Full Exploratory Dashboard

Does the same pattern survive on $$32\times 32$$ RGB colour photographs? On CIFAR-10 with our 3.83M U-Net (single seed), 100k steps reproduce the MNIST signature closely: $$\Delta(100\text{k}) = -0.48$$ and mean $$\rho = 0.94$$ ($$1.07$$ at $$n=128$$, $$0.80$$ at $$n=256$$). Extending $$n \in \{128, 256, 512\}$$ to 300k closes the $$n = 128$$ gap from `0.70 vs. 1.00` to `0.93 vs. 1.00` (gap `0.07`), while $$n = 256$$ still sits at `0.80 vs. 0.98` (gap `0.18`, late slope `+0.015`, formally inconclusive on whether it reaches the exact same ceiling).

One clean detail to note on CIFAR-10: our permutation shuffled the 1,024 spatial positions while **keeping each pixel's `(R, G, B)` triplet together**. That preserves cross-channel colour correlation for the first convolution (biasing $$\rho$$ slightly downward compared to permuting all 3,072 scalar coordinates independently), and no CIFAR-10 MLP null was run. Below you can inspect sixteen actual training images from our $$n = 128$$ CIFAR-10 subset before and after scrambling alongside their 300k-step nearest-neighbour copies, followed by the full interactive explorer for Width, DiT Patch Size, and CIFAR-10.

<div class="viz-breakout" id="viz-cifar-train"></div>

<div class="viz-breakout" id="viz-scale-arch"></div>

---

## 6. Synthesis & What to Run Next

Pulling the baseline calibration and the new exploratory leads into one view:

| Result | Relation to prior work | Current evidence | Why it matters |
|---|---|---|---|
| **Baseline U-Net slowdown $$\rho \approx 1.0\text{--}1.2$$** with small $$\Delta(100\text{k}) \in [-0.35, -0.43]$$ | Diffusion memorisation-time measurement of CNN permutation inertia<sup class="cite"><a href="#ref-8">[8]</a>, <a href="#ref-9">[9]</a>, <a href="#ref-11">[11]</a>, <a href="#ref-12">[12]</a></sup> | **Replicated** on MNIST (3 reps, 3 perms, MLP null); censored `> 0.97–1.20` on FMNIST at 300k. | Quantifies the ~2× memorisation-time cost of losing spatial locality |
| **Fixed-budget confound $$\Delta(B) \approx -\rho/a$$** | Algebraic consequence of $$\tau_{\text{mem}} \propto n^a$$<sup class="cite"><a href="#ref-13">[13]</a>, <a href="#ref-15">[15]</a></sup> | **Descriptive** across 10k → 100k; $$n \ge 4096$$ unplateaued at 300k. | Prevents mistaking optimisation slowdowns for capacity shifts |
| **Slowdown $$\rho$$ collapses with U-Net width** (`1.08 → 0.93 → 0.26`) | Natural onset `+1` oct/2×W replicates Bonnaire et al.;<sup class="cite"><a href="#ref-13">[13]</a></sup> **$$\rho$$-vs-width is new.** | **Exploratory** (1 seed/width, non-overlapping $$n$$ ranges). | Suggests convolutional locality is an optimisation crutch for small models |
| **Slowdown orders with DiT patch size** (`U-Net 0.93 > p2 0.55 > p4 ≈ p1 = 0`) | Connects DiT locality<sup class="cite"><a href="#ref-20">[20]</a></sup> and $$p=1$$ symmetry<sup class="cite"><a href="#ref-21">[21]</a></sup> to memorisation speed; **ordering is new.** | **Exploratory** (1 seed, $$n \in \{128, 256\}$$, shared U-Net lr). | Shows global self-attention softens the locality penalty continuously |
| **CIFAR-10 transfer** ($$\Delta = -0.48$$, $$\rho = 0.94$$; 300k gap `0.07` at `n=128`, `0.18` at `n=256`) | Extends shuffled diffusion denoising<sup class="cite"><a href="#ref-12">[12]</a></sup> to memorisation time on colour images. | **Exploratory** (1 seed, RGB grouped per pixel, no CIFAR MLP null). | Confirms the ~2× slowdown on colour photos; ceiling at $$n=256$$ still open |

### The physical boundary: small images vs. sub-image receptive fields

At $$28\times 28$$ or $$32\times 32$$, our 3-level U-Net's bottleneck receptive field already spans the entire image. That means all of these experiments probe **early-layer local weight sharing and optimisation bias**, not a hard geometric bottleneck where a CNN literally cannot integrate distant pixels. On $$64\times 64$$ or $$256\times 256$$ images — or in a shallow CNN whose receptive field stays strictly smaller than the image — pixel permutation could still trigger a genuine asymptotic capacity drop rather than pure optimisation inertia.

### The four sharpest follow-up experiments

1. **Multi-seed width sweep on a matched $$n$$ grid:** 3 seeds each for `×½`, `×1`, and `×2` U-Nets evaluated on common $$n \in \{512, 1024\}$$ (plus dense early checkpoints for `×2`), turning Finding 1 from an exploratory clue into a firm result.
2. **Multi-seed, lr-tuned DiT patch sweep:** 3 seeds and per-architecture learning-rate checks for `p=2` and `p=4` DiTs (plus a DiT noise floor) to nail down Finding 2 cleanly.
3. **Zero-GPU spectral-bias theory check:** compute the eigenvalue spectrum of natural vs. permuted MNIST $$3\times 3$$ patch covariance matrices under Wang &amp; Pehlevan's<sup class="cite"><a href="#ref-11">[11]</a></sup> local-convolution dynamics and test whether $$\log_2(\lambda_{\text{nat}} / \lambda_{\text{perm}})$$ predicts $$\rho \approx 1$$ quantitatively.
4. **A true receptive-field dial (and full 3,072-coord CIFAR permutation):** vary CNN depth without downsampling (receptive field 5, 9, 15 pixels on $$28\times 28$$) and scale training steps with $$n$$ ($$B \propto n$$) to test where optimisation inertia gives way to a real plateau shift.

<details class="deep-dive">
  <summary>Pre-registration &amp; protocol amendment log (Amendments 1–10)</summary>
  <p>
    All primary decision thresholds were pre-registered before running confirmatory replicates, with amendments logged whenever evaluation grids or controls needed adjustment:
  </p>
  <ul>
    <li><strong>Amendment 1 &amp; 2 (post-hoc after <code>exp01</code> pilot):</strong> Added \(\tau_{\text{mem}}(n)\) and \(\rho(n)\) after noticing the natural vs. permuted speed gap on the first \(n=1024\) pilot pair; raised deciding budget from 30k to 100k because \(n^*\) was still drifting. Confirmatory only for runs launched after Amendment 2 (\(n=2048\) and all replicates).</li>
    <li><strong>Amendment 3 (confirmatory MNIST replicates):</strong> Added <code>repA</code> &amp; <code>repB</code>, defined \(\sigma\), added 300k seed-0 predictions (<code>exp06</code>), and re-registered the MLP null from \(m=0.5\) to \(m=0.25\) (last downward crossing) because the MLP never reached \(0.5\) on the decreasing side.</li>
    <li><strong>Amendment 4 &amp; 5 (FashionMNIST transfer &amp; follow-ups):</strong> Pre-registered FMNIST transfer failed the original MLP null by <code>0.01</code> (\(\vert\Delta\vert = 0.51\) at 70k due to coarse-grid straddling at \(n=128\)); Amendment 5 added \(n=96, 192\) (<code>exp11</code>, passes at \(\le 0.29\)) and extended \(n=2048\) to 300k (<code>exp10</code>, censored one-sided <code>INCONCLUSIVE</code>).</li>
    <li><strong>Process deviations on FMNIST:</strong> <code>exp10</code> \(n=2048\) runs accidentally resumed from 70k instead of 100k checkpoints (recomputed 100k evals agree within <code>0.013</code>); two FMNIST 300k runs had <code>metrics.json</code> overwritten on relaunch and were reconstructed from stdout logs (raw distance files \(\ge\) 100k lost; verdicts unaffected).</li>
    <li><strong>Amendment 6 (permutation-identity control):</strong> Pre-registered before <code>exp12</code>; <code>perm1</code> passed, <code>perm2</code> hit \(\rho(2048) = 0.99\) (<code>INCONCLUSIVE</code> by the letter).</li>
    <li><strong>Amendment 7 &amp; 8 (width sweep &amp; CIFAR-10):</strong> Single-seed exploratory rules relaxed speed threshold from \(\rho \ge 1.0\) to consistent \(\rho > 0\) with mean \(\ge 0.5\). Amendment 8 extended CIFAR-10 \(n=128\text{--}512\) to 300k (<code>INCONCLUSIVE</code> on the ceiling test).</li>
    <li><strong>Amendment 9 &amp; 10 (DiT locality dial):</strong> Initial <code>n = 256..4096</code> grid left DiT speed <code>UNDEFINED</code>; Amendment 10 added <code>n = 128</code> post hoc so \(\rho\) could be read at two points.</li>
  </ul>
</details>

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
  <li id="ref-13">Bonnaire, T., Urfin, R., Biroli, G., &amp; Mézard, M. (2025). <em>Why Diffusion Models Don't Memorize: The Role of Implicit Dynamical Regularization in Training.</em> <a href="https://arxiv.org/abs/2505.17638">arXiv:2505.17638</a></li>
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
</ol>
