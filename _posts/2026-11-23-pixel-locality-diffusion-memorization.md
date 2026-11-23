---
layout: post
title: "What Happens When You Scramble the Pixels of a Diffusion Model?"
description: "Using a fixed random pixel permutation to separate spatial locality from data geometry in diffusion memorisation — what was already known, what surprised us, and why fixed-budget comparisons mix speed with capacity."
kicker: Research note
date: 2026-11-23
math: true
asset_base: /assets/diffusion_locality/
custom_css:
  - /assets/diffusion_locality/post.css
custom_js:
  - /assets/diffusion_locality/chart_kit.js
  - /assets/diffusion_locality/widget_hero.js
  - /assets/diffusion_locality/widget_ls_metric.js
  - /assets/diffusion_locality/widget_capacity_speed.js
  - /assets/diffusion_locality/widget_scaling_arch.js
  - /assets/diffusion_locality/main.js
---

<p class="lead-para">
Train a diffusion model on a tiny dataset and it becomes a photocopier: feed it Gaussian noise, and it snaps straight back to a training image. Give it a larger dataset and it generalises — inventing novel digits, shirts, or cars even though the exact empirical score points exclusively toward the training points.<sup class="cite"><a href="#ref-1">[1]</a>, <a href="#ref-2">[2]</a></sup> Two accounts disagree on why: does generalisation come from <strong>architectural locality</strong> (convolutional patches)<sup class="cite"><a href="#ref-3">[3]</a>, <a href="#ref-4">[4]</a></sup> or from <strong>data statistics</strong> (natural-image pixel correlations)<sup class="cite"><a href="#ref-5">[5]</a></sup>?
</p>

Apply **one fixed random permutation** $$\pi$$ of the pixel grid to every image and the two accounts separate cleanly: every pairwise $$L_2$$ distance and covariance eigenvalue stays identical, while every $$3\times 3$$ convolutional patch shatters into nine scattered, nearly uncorrelated coordinates. Play with our real U-Net outputs below before we unpack the physics and pre-registered experiments.

<!--more-->

<div class="viz-breakout" id="viz-hero"></div>

What follows is a curiosity-driven walk through a small, pre-registered series of experiments built one step on top of the next. We started expecting a clean binary test of capacity, got surprised by a dynamical slowdown that looked like a capacity shift at a fixed budget, and ended up learning just as much from reviewing older deep-learning literature as from our own GPU runs.

---

## 1. Diffusion Through a Statistical-Physics Lens

Before interpreting any curves, it helps to write down the exact physical picture of memorisation in a diffusion model.

Take a training set $$D = \{x_\mu\}_{\mu=1}^n \subset \mathbb{R}^d$$ of $$n$$ clean images. Under the variance-preserving cosine forward process used in our experiments (with $$T = 1000$$ noise levels and signal retention $$\bar\alpha_t$$ decreasing from $$1$$ toward $$0$$), a noisy image at step $$t$$ is

$$
x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, I_d).
$$

If you replace the unknown true data distribution with the empirical delta-mixture over the $$n$$ training points, the exact Bayes-optimal denoiser $$\hat{x}_0^{\text{emp}}(x_t, t) = \mathbb{E}[x_0 \mid x_t]$$ takes a form any statistical physicist recognises immediately:

$$
\hat{x}_0^{\text{emp}}(x_t, t) = \sum_{\mu=1}^n w_\mu(x_t, t)\, x_\mu, \qquad
w_\mu(x_t, t) = \frac{\exp\!\big(-E_\mu(x_t, t) / T_t\big)}{\sum_{\nu=1}^n \exp\!\big(-E_\nu(x_t, t) / T_t\big)},
$$

with quadratic interaction energy $$E_\mu(x_t, t) = \tfrac{1}{2}\|x_t - \sqrt{\bar\alpha_t}\,x_\mu\|_2^2$$ and effective temperature $$T_t = 1 - \bar\alpha_t$$. The corresponding empirical score is $$s_{\text{emp}}(x_t, t) = \big(\sqrt{\bar\alpha_t}\,\hat{x}_0^{\text{emp}}(x_t, t) - x_t\big) / (1 - \bar\alpha_t)$$.

In statistical mechanics, this partition sum is a **Random Energy Model (REM)**.<sup class="cite"><a href="#ref-6">[6]</a>, <a href="#ref-7">[7]</a></sup> Early in reverse sampling (high $$t$$, high temperature $$T_t$$), thermal fluctuations smear the Boltzmann weights across broad clusters of images — the *speciation* regime. As sampling cools down toward $$t \to 0$$ ($$T_t \to 0$$), the weights collapse onto whichever single training point $$\mu^*$$ minimizes the energy $$E_\mu$$. Follow $$s_{\text{emp}}$$ faithfully, and the system **condenses**: every generated sample ends in a training-set copy.

So why don't trained neural networks condense when $$n$$ is reasonably large? Two geometric explanations compete in the recent diffusion literature:

1. **Patch-restricted local scores (the architecture account).** Kamb & Ganguli<sup class="cite"><a href="#ref-3">[3]</a></sup> and Hunt, Kamb & Ganguli's **BIRD** framework<sup class="cite"><a href="#ref-4">[4]</a></sup> replace the global energy $$E_\mu$$ with a separate local energy for each pixel $$i$$ restricted to an $$L \times L$$ patch $$P_i$$:
   $$E_\mu^{(i)}(x_t, t) = \tfrac{1}{2}\big\| (x_t - \sqrt{\bar\alpha_t}\,x_\mu)_{P_i} \big\|_2^2.$$
   Each pixel picks its own Boltzmann mixture over patches from different training images, synthesising novel global images until the dataset shrinks enough for patch mutual information $$I(x_{t, P_i}; \mu)$$ to cross the Fano / REM condensation limit $$\ln n$$.
2. **Data-covariance locality (the data-statistics account).** Lukoianov et al.<sup class="cite"><a href="#ref-5">[5]</a></sup> argue that local sensitivity is *not* mainly an inductive bias of convolutions: U-Nets and Diffusion Transformers (DiTs) develop remarkably similar local sensitivity fields because natural images have decaying spatial covariance $$\Sigma_{ij} = \text{Cov}(x_i, x_j)$$. In fact, linear Wiener filters computed from $$\Sigma$$ already isolate local neighbourhoods — and when BIRD compares against trained networks on MNIST and FashionMNIST, it actually picks its per-pixel patch masks from those data-derived Wiener filters rather than square boxes!<sup class="cite"><a href="#ref-4">[4]</a></sup>

<div class="callout callout--honesty">
  <div class="callout__title">Honesty check: What was already known before we ran anything</div>
  <p>
    Our adversarial literature pass turned up a humbling lesson: the cleanest qualitative intuition behind pixel-scrambling was already established years ago outside the recent diffusion-memorisation wave.
  </p>
  <ul>
    <li><strong>Shuffled pixels are memorised eventually:</strong> Zhang et al. (2016)<sup class="cite"><a href="#ref-8">[8]</a></sup> defined the exact same fixed permutation ("shuffled pixels") and showed CNNs still fit them to zero training error (Fig. 1a), while Arpit et al. (2017)<sup class="cite"><a href="#ref-22">[22]</a></sup> established that deep nets fit noise only after prioritising simpler patterns. Ulyanov et al. (2017, <em>Deep Image Prior</em> §2.1 &amp; Fig. 4)<sup class="cite"><a href="#ref-9">[9]</a></sup> showed a CNN fits a pixel-permuted image "very reluctantly" with "significant inertia", yet "in the limit the parametrization can fit noise as well".</li>
    <li><strong>Optimisation path &amp; spectral bias:</strong> d'Ascoli et al. (2019)<sup class="cite"><a href="#ref-10">[10]</a></sup> showed the convolutional prior acts primarily through the optimisation path in function space, and Ingrosso &amp; Goldt (2022)<sup class="cite"><a href="#ref-23">[23]</a></sup> showed even fully connected nets can acquire local receptive fields from non-Gaussian image statistics. Wang &amp; Pehlevan (2025)<sup class="cite"><a href="#ref-11">[11]</a></sup> proved analytically that diffusion learning times scale inversely with data covariance eigenvalues (\(\tau_k \propto 1/\lambda_k\)) — permutation-invariant for an MLP, but redirected into <em>patch-covariance space</em> by local convolutions, echoing patch-PCA cues in diffusion seeds.<sup class="cite"><a href="#ref-24">[24]</a></sup></li>
    <li><strong>Shuffled diffusion denoisers &amp; linear memorisation time:</strong> Kadkhodaie et al. (2023, App. C.3)<sup class="cite"><a href="#ref-12">[12]</a></sup> already trained a CNN denoiser on \(10^5\) pixel-shuffled CelebA faces and observed markedly worse denoising even though the Bayes optimum is unchanged. And Bonnaire et al. (2025)<sup class="cite"><a href="#ref-13">[13]</a></sup> — using the exact same U-Net family and \(d_1/d_2 < 1/3\) metric we adopt — already established \(\tau_{\text{mem}} \propto n\,W^{-1}\), alongside Gu et al.<sup class="cite"><a href="#ref-14">[14]</a></sup>, Favero et al.<sup class="cite"><a href="#ref-15">[15]</a></sup>, and Marion &amp; Wu<sup class="cite"><a href="#ref-16">[16]</a></sup>.</li>
  </ul>
  <p>
    <strong>So what is genuinely ours?</strong> Not the qualitative phenomenon. Our incremental contribution is (i) a pre-registered, replicated measurement of the <em>diffusion memorisation onset and slowdown</em> under pixel permutation with an explicit permutation-equivariant MLP null; (ii) tracking how that slowdown creates an apparent fixed-budget capacity shift via \(\Delta \approx -\rho/a\); and (iii) two single-seed exploratory clues — how the slowdown shrinks with U-Net width and tracks DiT patch locality.
  </p>
</div>

---

## 2. Designing a Clean Intervention (and Making Predictions First)

Apply one fixed orthogonal permutation matrix $$P \in \{0,1\}^{d \times d}$$ to every training and test image, $$x \mapsto Px$$. Because $$P^\top P = I_d$$, every Euclidean quantity in the problem is **strictly invariant**:

$$
\|Px - Py\|_2 = \|x - y\|_2, \qquad \Sigma_{Px} = P \Sigma_x P^\top.
$$

Why does that matter experimentally? First, our memorisation metric stays completely unbiased across coordinate systems. Following Yoon et al.<sup class="cite"><a href="#ref-1">[1]</a></sup>, Gu et al.<sup class="cite"><a href="#ref-14">[14]</a></sup>, and Bonnaire et al.<sup class="cite"><a href="#ref-13">[13]</a></sup>, we draw 1,000 DDIM-50 samples from the EMA weights (at fixed sampling noise seed) and call a sample **memorised** if its nearest training-set neighbour distance $$d_1$$ is less than one third of its second-nearest distance $$d_2$$ (with $$d_1/d_2 < 1/2$$ logged as a secondary check):

$$
\frac{d_1}{d_2} < \frac{1}{3}.
$$

Had we measured denoising PSNR or FID instead, worse denoising on scrambled images (Kadkhodaie's observation<sup class="cite"><a href="#ref-12">[12]</a></sup>) could easily masquerade as "less memorisation". Because $$L_2$$ distance is untouched by $$P$$, a sample with $$d_1/d_2 < 1/3$$ in permuted space has the exact same $$d_1/d_2$$ when unscrambled back to natural pixels.

Second, any model whose predictions depend *only* on pairwise distances, the global REM energy, or covariance-derived Wiener masks (which permute covariantly as $$PW P^\top$$) must give an **exact zero shift**:

$$
\Delta = \log_2 n^*_{\text{perm}} - \log_2 n^*_{\text{nat}} = 0,
$$

where $$n^*$$ is the training-set size at which the memorised fraction $$m(n)$$ crosses $$0.5$$ (measured in octaves, i.e. doublings of $$n$$).

What about a purely architectural square-patch local-score model? Before touching a GPU neural network, we built a network-free square-patch LS model (`src/ls_predict.py`) following Kamb & Ganguli.<sup class="cite"><a href="#ref-3">[3]</a></sup> We calibrated its noise-dependent square patch width $$L(t)$$ by minimising held-out denoising MSE on natural MNIST ($$n_{\text{ref}} = 10^4$$), then froze $$L(t)$$ and ran DDIM sampling in both natural and permuted pixel grids. Notice what happened — and notice why even two people on our own team initially disagreed on the sign!

<div class="viz-breakout" id="viz-ls"></div>

Our pre-study adversarial review had guessed $$\Delta > 0$$ (reasoning that a scattered $$5\times 5$$ patch carries more independent pixels and hence more mutual information, which should memorise *more* images). Running the actual network-free LS machine gave the opposite sign, $$\Delta_{\text{LS}} \le -3$$: each scrambled patch greedily locks onto *some* training image, but neighbouring patches lock onto *different* training images, producing un-copied mosaics with median $$d_1/d_2 \approx 0.90\text{--}0.99$$! Meanwhile, a subtle point surfaced by `novelty_review.md`: no published paper actually claims a *late-training* U-Net behaves like a *fixed square-patch* LS machine (BIRD already notes its agreement drops later in training<sup class="cite"><a href="#ref-4">[4]</a></sup>), so $$\Delta_{\text{LS}} \le -3$$ is an informative baseline extreme rather than a straw-man knockout.

<details class="deep-dive">
  <summary>Math note: Why the MLP and patch-1 DiT are exact distribution nulls under \(P\)</summary>
  <p>
    Consider a residual MLP whose input layer maps \(x \in \mathbb{R}^d\) via \(W_0 x + b_0\) with i.i.d. isotropic Gaussian weight initialisation \(W_0 \sim \mathcal{N}(0, \sigma_0^2 I)\), followed by permutation-invariant hidden layers and an output layer \(W_{\text{out}} h + b_{\text{out}}\). Under \(x \mapsto P x\), relabelling the input and output weight columns/rows as \(\tilde{W}_0 = W_0 P^\top\) and \(\tilde{W}_{\text{out}} = P W_{\text{out}}\) preserves the exact initial weight distribution (since \(P\) is orthogonal and i.i.d. Gaussians are orthogonal-invariant) and leaves the forward pass, loss, AdamW updates, and generated sample distribution identical up to relabelling \(P\).
  </p>
  <p>
    The exact same symmetry holds for a Diffusion Transformer with patch size \(p=1\) and i.i.d. normal learned positional embeddings \(E_{\text{pos}} \in \mathbb{R}^{d \times w}\): permuting pixels simply permutes the rows of \(E_{\text{pos}}\) (checked numerically in our DiT code: maximum output discrepancy \(2.4 \times 10^{-7}\) in fp32). Any nonzero shift measured on the MLP must therefore come purely from finite-seed noise or pipeline bugs.
  </p>
</details>

---

## 3. Capacity Barely Moves — Until You Look at the Clock

We trained a purely convolutional 3.8M-parameter U-Net (channels 32–64–128, 2 residual blocks per level, no attention) across $$n \in \{512, 1024, 2048, 4096\}$$ on both MNIST and FashionMNIST, with three replicates per arm (`rep0`, `repA`, `repB`, varying initialisation seed and training subset), alongside the residual MLP null control. Every decision threshold was written into `facts.md` before running the confirmatory replicates (with post-hoc adjustments documented transparently in the audit log below).

### First, did the MLP pipeline null pass?

Mostly yes — with one candid hiccup that shows why pre-registration rules matter. On MNIST ($$\text{repA}, \text{repB}$$) and FashionMNIST, the MLP's $$m(n)$$ curve actually *drops* at tiny $$n$$ due to undertraining (so $$m = 0.5$$ is undefined), forcing us to re-register the null at $$m = 0.25$$ on the last downward crossing before running replicates. On MNIST, $$\vert \Delta_{\text{MLP}} \vert \le 0.37$$ and $$\vert \rho_{\text{MLP}} \vert \le 0.28$$ at every step — a clean pass. On FashionMNIST's original grid, however, one cell (`repB` at 70k) hit $$\vert \Delta_{\text{MLP}} \vert = 0.51$$ — **failing the 0.50 rule by 0.01** because $$m(128)$$ straddled $$0.25$$ and swung the linear interpolation a full octave! Adding two pre-registered grid points ($$n = 96, 192$$) brought FashionMNIST back to $$\vert \Delta_{\text{MLP}} \vert \le 0.29$$. We keep the original failure on record and treat FashionMNIST verdicts as conditional on that finer-grid null.

### Now look at the U-Net at 100k steps

At a 100k-step budget, the natural and pixel-permuted U-Nets sit remarkably close together:

| Dataset | $$\Delta(100\text{k})$$ rep0 / A / B | Noise floor $$\sigma$$ | Pre-registered verdict ($$\vert\Delta\vert \le \max(0.5, \sigma)$$) |
|---|---|---|---|
| MNIST | $$-0.35$$ / $$-0.39$$ / $$-0.41$$ | $$0.08$$ | **NO SHIFT** |
| FashionMNIST | $$-0.42$$ / $$-0.43$$ / $$-0.36$$ | $$0.08$$ | **NO SHIFT** |

Scrub the **Training Budget** slider in the capacity explorer below — and pay attention to what happens at 10k vs. 100k steps!

<div class="viz-breakout" id="viz-capacity"></div>

Look closely at two details in that figure before celebrating "NO SHIFT":
1. **The shift is not zero.** All six replicates sit tightly clustered between $$-0.35$$ and $$-0.43$$ octaves against a replicate noise floor of $$\sigma = 0.08$$ — roughly a $$5\sigma$$ nonzero shift (about $$1.3\times$$ smaller $$n^*$$ at 100k). That falls cleanly inside our pre-registered $$0.5$$-octave equivalence band (and an order of magnitude away from the $$\le -3$$ square-patch LS model), yet it clearly breaks the naive fixed-budget reading of $$\Delta = 0$$.
2. **Watch the budget slider:** at 10k steps, $$\Delta \approx -0.64$$; by 100k steps, it shrinks to $$-0.35$$. That immediately begs a dynamical question.

### Looking at the clock: how fast does each arm condense?

When we plot $$m(\text{step})$$ against $$\log(\text{step})$$ at fixed $$n$$, the mystery snaps into focus. Define $$\tau(n)$$ as the step when $$m$$ first reaches $$0.5$$, and the slowdown in octaves of training time as

$$
\rho(n) = \log_2\!\left(\frac{\tau_{\text{perm}}(n)}{\tau_{\text{nat}}(n)}\right).
$$

On MNIST at $$n = 2048$$, the permuted U-Net reaches the **same plateau** (on our seed-0 300k run, natural hits $$0.82$$ and permuted hits $$0.83$$), yet it takes **$$\approx 2.0\text{--}2.3\times$$ longer** to get there ($$\rho(2048) = 1.18, 1.17, 1.06$$). Even better: try dragging the **$$\rho$$ time-shift slider** in the next widget to slide the permuted trajectory left by $$2^{-\rho}$$ and watch the two curves collapse onto a single trajectory.

<div class="viz-breakout" id="viz-speed"></div>

What about FashionMNIST and alternative permutation seeds? Here we report the pre-registered verdicts candidly by the letter:
- **FashionMNIST speed is formally `INCONCLUSIVE`:** at $$n = 2048$$, even after 300k steps the permuted arm only reached $$m = 0.33\text{--}0.43$$, so $$\tau_{\text{perm}}$$ is censored. The one-sided lower bounds $$\log_2(300\text{k}/\tau_{\text{nat}})$$ are $$> 1.20$$ (`rep0`), $$> 1.11$$ (`repA`), and $$> 0.97$$ (`repB`) — missing the $$1.0$$ rule by $$0.03$$ in `repB` simply because the 300k clock ran out, even though $$\rho > 0$$ everywhere.
- **Permutation-identity control (`perm_seed` 1 and 2 on MNIST `rep0`):** $$\Delta(100\text{k})$$ barely budges ($$-0.35, -0.34, -0.27$$), confirming capacity doesn't care *which* random permutation you pick. The slowdown is present for all three permutations too, though `perm_seed 2` lands at $$\rho(2048) = 0.99$$ — **$$0.01$$ under the pre-registered $$1.0$$ rule**, making that control `INCONCLUSIVE` by the letter and reminding us that a fair summary of the MNIST slowdown is $$\rho(2048) \approx 1.0\text{--}1.2$$, not strictly $$\ge 1$$ in every draw.

---

## 4. One Line of Algebra Explains Both — and Exposes a Confound

Why did $$\Delta(B)$$ sit at $$-0.7$$ early in training and shrink to $$-0.35$$ by 100k? Suppose the natural network's memorisation time grows as a power law in dataset size, $$\tau_{\text{nat}}(n) \propto n^a$$ (Bonnaire et al.<sup class="cite"><a href="#ref-13">[13]</a></sup> and Favero et al.<sup class="cite"><a href="#ref-15">[15]</a></sup> already established $$a \approx 1$$ for this exact regime), and suppose scrambling pixels multiplies that time scale by $$2^\rho$$:

$$
\tau_{\text{perm}}(n) \approx 2^\rho\, \tau_{\text{nat}}(n) \propto 2^\rho\, n^a.
$$

If you stop both models at any fixed training budget $$B$$ (setting $$\tau_{\text{nat}}(n^*_{\text{nat}}) = B$$ and $$\tau_{\text{perm}}(n^*_{\text{perm}}) = B$$), one line of algebra gives the apparent capacity shift:

$$
n^*_{\text{perm}}(B)^a \approx 2^{-\rho}\, n^*_{\text{nat}}(B)^a \implies \Delta(B) = \log_2 n^*_{\text{perm}}(B) - \log_2 n^*_{\text{nat}}(B) \approx -\frac{\rho}{a}.
$$

On MNIST, the local log-log slope $$a = \mathrm{d}\log_2 \tau_{\text{nat}} / \mathrm{d}\log_2 n$$ steepens from $$a \approx 1.0$$ around $$n = 256\text{--}512$$ to $$a \approx 1.7$$ around $$n = 1024\text{--}2048$$. Plugging $$\rho \approx 1$$ in gives $$\Delta \approx -0.7$$ early on, tapering toward $$-0.35\text{ to }-0.5$$ later — matching our observed trajectory cleanly.

**Why does Wang & Pehlevan's spectral bias explain $$\rho > 0$$?**<sup class="cite"><a href="#ref-11">[11]</a></sup> In linearised diffusion training dynamics, each covariance eigenmode of the data is learned on a timescale inversely proportional to its eigenvalue variance, $$\tau_k \propto 1/\lambda_k$$. An MLP sees the full $$d\times d$$ covariance matrix $$\Sigma$$, whose eigenvalues are invariant under $$\Sigma \mapsto P\Sigma P^\top$$ — hence $$\rho_{\text{MLP}} = 0$$. A local convolutional layer, by contrast, effectively learns in *patch-covariance space*: natural patches concentrate massive variance into a handful of smooth low-frequency modes (learned very quickly), whereas scrambled $$3\times 3$$ patches whiten that spectrum toward weaker, flatter eigenvalues, slowing optimisation down without altering what the overparameterised network can eventually fit!

<div class="callout callout--physics">
  <div class="callout__title">The "Danger Zone" &amp; an important caveat about plateaus</div>
  <p>
    Switch the right panel below to <strong>Danger Zone: Per-t MSE Gap</strong> to see the held-out minus training \(\varepsilon\)-MSE across noise levels (studied in detail by Dodson et al.<sup class="cite"><a href="#ref-18">[18]</a></sup> and Kaiser &amp; Kollmann<sup class="cite"><a href="#ref-19">[19]</a></sup>). At low noise (\(t = 50\)), the permuted U-Net keeps a much larger test denoising error (\(1.64\) vs \(0.90\) at 100k, \(n = 2048\)) even while its \(d_1/d_2\) memorisation fraction catches up toward the natural arm — consistent with Kadkhodaie et al.<sup class="cite"><a href="#ref-12">[12]</a></sup> and explaining why you still see faint salt-and-pepper grain on non-memorised permuted samples in the Hero sandbox!
  </p>
  <p>
    <strong>Honest caveat:</strong> We never reached an asymptotic plateau at \(n \ge 4096\) (at 300k steps, \(n=4096\) sits at \(m_{\text{nat}} = 0.47\) vs \(m_{\text{perm}} = 0.26\)). Along our single 300k seed-0 run, \(\Delta\) only nudged from \(-0.29\) at 100k to \(-0.33\) at 300k (our pre-registered prediction \(\vert\Delta(300\text{k})\vert < 0.35\) passed marginally: \(0.33 < 0.35\)). So calling the residual \(\Delta\) purely a vanishing time-scale artefact remains an <em>inference</em>, not a plateau proof — and it strictly means "capacity" in our experiments refers to 100k-step fixed-budget onset rather than an infinite-training ceiling.
  </p>
</div>

<div class="viz-breakout" id="viz-delta-rho"></div>

---

## 5. Exploratory Scale-Out: Width, Colour Images & Diffusion Transformers

Once the MNIST/FashionMNIST picture became clear, we ran three **single-seed exploratory** scale-outs to probe how architectural locality and capacity interact. Treat every trend in this section as a preliminary clue rather than a replicated verdict.

1. **Varying U-Net width (`1.15M` → `3.83M` → `14.19M` params, MNIST).**
   - **What is already known / replicated:** Natural onset grows at $$+1.12$$ and $$+0.84$$ octaves per width doubling ($$\log_2 n^*_{\text{nat}} = 10.30 \to 11.42 \to 12.26$$). That cleanly replicates Bonnaire et al.'s<sup class="cite"><a href="#ref-13">[13]</a></sup> published scaling $$\tau_{\text{mem}} \propto n W^{-1}$$, which already predicts $$n^* \propto W$$ ($$+1$$ octave per doubling) at fixed budget.
   - **What looks new (with strong caveats):** The permutation slowdown *shrinks sharply with width*: mean $$\rho$$ falls from $$1.08$$ ($$\times\tfrac{1}{2}$$) to $$0.93$$ ($$\times 1$$) to $$0.26$$ ($$\times 2$$), while $$\Delta(100\text{k})$$ dutifully tracks it via $$-\rho/a$$ ($$-0.65 \to -0.35 \to +0.05$$). One tempting physical interpretation is that the convolutional prior acts mainly as an optimisation crutch for capacity-constrained networks, whereas a 14.2M-parameter U-Net has enough channels to fit scrambled images nearly as fast as natural ones. **Crucial caveat:** this rests on one seed per width, read across *different* $$n$$ ranges ($$256\text{--}512$$ at $$\times\tfrac{1}{2}$$ vs. $$1024\text{--}4096$$ at $$\times 2$$), and at $$\times\tfrac{1}{2}$$ capacity is formally `INCONCLUSIVE` while at $$\times 2$$ speed is formally `INCONCLUSIVE`. Notice also that at $$\times 2$$ (14.2M params on $$n \le 4096$$), fast memorisation pushes $$\tau_{\text{nat}}$$ closer to our earliest evaluation grids.

2. **Dialling locality in Diffusion Transformers (DiTs, ~4.7M params, MNIST).**
   Inspired by An et al.<sup class="cite"><a href="#ref-20">[20]</a></sup>, we trained adaLN-Zero DiTs (width 192, depth 6) with fixed 2D sin-cos positions while varying patch size $$p \in \{2, 4\}$$ alongside our exact-equivariant $$p = 1$$ learned-position null. Notice the ordering of the slowdown:
   $$\text{U-Net } (0.93) \;>\; \text{DiT } p=2 \; (0.55) \;>\; \text{DiT } p=4 \; (\approx -0.25) \;\approx\; \text{DiT } p=1 \; (0 \text{ exact}).$$
   Capacity stays inside the NO SHIFT band for both DiTs ($$\Delta = -0.39$$ for $$p=2$$; $$+0.04$$ for $$p=4$$). **Important caveats:** One seed per DiT; $$\rho$$ rests on just two points ($$n = 128, 256$$, where $$n = 128$$ was added post hoc in Amendment 10 after the initial grid came back `UNDEFINED`); the DiTs used the U-Net's untuned optimiser schedule (AdamW `lr = 2e-4`, 100k steps); and although the DiTs' fixed-budget $$n^*$$ sits ~2.8 octaves lower than the U-Net's, Section 4's exact warning forbids treating fixed-budget $$n^*$$ across architectures as a pure capacity comparison!

3. **CIFAR-10 at 100k → 300k (32×32 colour images, `3.8M` U-Net).**
   At 100k steps, CIFAR-10 repeats the pattern: $$\Delta(100\text{k}) = -0.48$$ (`NO SHIFT`, though sitting just $$0.02$$ inside the $$0.5$$ border) and mean $$\rho = 0.94$$ (`SPEED EFFECT`). Extending $$n \in \{128, 256, 512\}$$ to 300k largely closes the suspicious `0.70` vs `1.00` gap at $$n=128$$ (reaching `0.93` vs `1.00`, gap `0.07`), yet $$n = 256$$ levels out at `0.80` vs `0.98` (gap `0.18`, late slope `+0.015`, formally `INCONCLUSIVE`). **Two important caveats:** no CIFAR-10 MLP null was run, and our CIFAR permutation shuffled the 1,024 pixel positions while **keeping each pixel's `(R, G, B)` triplet together** — preserving strong cross-channel colour correlations for the first convolution! A cleaner follow-up should permute all 3,072 scalar coordinates independently.

<div class="viz-breakout" id="viz-scale-arch"></div>

---

## 6. Honest Stock-Taking: What Holds, What Is Thin, and What Comes Next

Pulling back from individual plots, it helps to separate cleanly what our experiments **replicated from prior work**, what they **newly quantified**, and what remains **unsettled**.

| Finding | Prior work &amp; relationship | Strength of evidence here | Status |
|---|---|---|---|
| **Eventual capacity is permutation-invariant** ($$\vert\Delta\vert \le 0.43$$ at 100k on MNIST &amp; FMNIST) | Known qualitatively for CNNs since Zhang et al. (2016)<sup class="cite"><a href="#ref-8">[8]</a></sup> and DIP (2017)<sup class="cite"><a href="#ref-9">[9]</a></sup>; every capacity-by-parameters account is permutation-invariant. | **Moderate–strong** at 100k (3 reps, $$\sigma=0.08$$, MLP null passed on finer grid), **weak as an asymptotic plateau claim** ($$n \ge 4096$$ unplateaued). | Incremental quantification; careful null &amp; replicates |
| **Pixel permutation slows diffusion memorisation ~2×** ($$\rho \approx 1.0\text{--}1.2$$ on MNIST) | Qualitative inertia known since DIP Fig. 4<sup class="cite"><a href="#ref-9">[9]</a></sup>; denoising penalty in Kadkhodaie et al.<sup class="cite"><a href="#ref-12">[12]</a></sup>; patch-covariance dynamics in Wang &amp; Pehlevan.<sup class="cite"><a href="#ref-11">[11]</a></sup> | **Moderate** on MNIST (3 reps; `perm2` hit `0.99`), **censored/inconclusive** on FashionMNIST at 300k. | New diffusion-memorisation measurement |
| **Budget confound $$\Delta(B) \approx -\rho/a$$** | Follows algebraically once $$\tau_{\text{mem}} \propto n^a$$ is known (Bonnaire et al.<sup class="cite"><a href="#ref-13">[13]</a></sup>; Favero et al.<sup class="cite"><a href="#ref-15">[15]</a></sup>). | **Descriptive / inferential** — fits budget drift from 10k to 100k cleanly, though $$\Delta$$ stalls near $$-0.3$$ out to 300k. | Useful methodological caution |
| **Fixed-budget $$n^*$$ grows +1 octave per U-Net width doubling** | **Directly predicted** by Bonnaire et al. (2025)<sup class="cite"><a href="#ref-13">[13]</a></sup> $$\tau_{\text{mem}} \propto n W^{-1}$$. | Single seed (`+1.12` / `+0.84` octaves). | Replication |
| **Slowdown $$\rho$$ shrinks with U-Net width** ($$1.08 \to 0.93 \to 0.26$$) | No prior measurement found. | **Thin / exploratory** (1 seed per width, non-overlapping $$n$$ ranges, $$\times\tfrac{1}{2}$$ &amp; $$\times 2$$ inconclusive by rule). | **Most interesting open lead** |
| **Slowdown orders with DiT patch locality** (`U-Net > p2 > p4 ≈ p1`) | An et al.<sup class="cite"><a href="#ref-20">[20]</a></sup> linked DiT locality to generalisation; $$p=1$$ learned is an exact symmetry null.<sup class="cite"><a href="#ref-21">[21]</a></sup> | **Thin / exploratory** (1 seed, 2 values of $$n$$ via post-hoc `n=128`, untuned U-Net lr). | Promising clue |

### The main physics caveat: tiny $$28 \times 28$$ images

At $$28 \times 28$$ or $$32 \times 32$$, our 3-level U-Net's full receptive field already covers the entire image! That means our experiment probes **early-layer local weight sharing and optimisation bias**, not a hard architectural bottleneck where a network literally cannot combine information across a large canvas. On $$256 \times 256$$ or $$512 \times 512$$ images where a CNN's effective receptive field stays genuinely smaller than the image, pixel permutation could still trigger a true asymptotic capacity collapse rather than mere optimisation inertia.

### What we would run next

If we push this further from a curiosity-driven note toward a rigorous paper, four cheap-to-medium follow-ups stand out directly from the gaps above:
1. **Seeds on a matched $$n$$ grid for the width sweep** (`H1`, `H2` in our project plan) — testing whether the shrinking $$\rho$$ with width really holds at matched $$n = 512, 1024$$ across 3 seeds.
2. **A quantitative Wang &amp; Pehlevan theory check (zero GPU compute):** computing the eigenvalues of natural vs. permuted MNIST $$3\times 3$$ patch covariance matrices and comparing $$\log_2(\lambda_{\text{nat}} / \lambda_{\text{perm}})$$ directly against our measured $$\rho \approx 1$$!
3. **Scaling training steps with $$n$$ ($$B \propto n$$)** so large-$$n$$ runs actually reach plateau rather than stopping at a fixed 100k clock.
4. **A receptive-field dial inside the CNN** (varying depth or removing downsampling so the receptive field is strictly smaller than $$28 \times 28$$, or scaling to $$64 \times 64$$ images with all $$C\cdot H\cdot W$$ coordinates permuted).

<details class="deep-dive">
  <summary>Pre-registration record &amp; honest deviation audit log (Amendments 1–10)</summary>
  <p>
    All primary decision thresholds were logged in <code>facts.md</code> on 2026-11-17 before running replicates, and amended transparently as gaps surfaced. Full raw logs live in the research repository; here is the honest summary:
  </p>
  <ul>
    <li><strong>Amendment 1 &amp; 2 (post-hoc after <code>exp01</code> pilot):</strong> Added \(\tau_{\text{mem}}(n)\) and \(\rho(n)\) after noticing the natural vs. permuted speed gap on the first \(n=1024\) pilot pair; raised deciding budget from 30k to 100k because \(n^*\) was still drifting. Confirmatory only for runs launched after Amendment 2 (\(n=2048\) and all replicates).</li>
    <li><strong>Amendment 3 (confirmatory MNIST replicates):</strong> Added <code>repA</code> &amp; <code>repB</code>, defined \(\sigma\), added 300k seed-0 predictions (<code>exp06</code>), and re-registered the MLP null from \(m=0.5\) to \(m=0.25\) (last downward crossing) because the MLP never reached \(0.5\) on the decreasing side.</li>
    <li><strong>Amendment 4 &amp; 5 (FashionMNIST transfer &amp; follow-ups):</strong> Pre-registered FMNIST transfer failed the original MLP null by <code>0.01</code> (\(\vert\Delta\vert = 0.51\) at 70k due to coarse-grid straddling); Amendment 5 added \(n=96, 192\) (<code>exp11</code>, passes at \(\le 0.29\)) and extended \(n=2048\) to 300k (<code>exp10</code>, censored one-sided <code>INCONCLUSIVE</code>).</li>
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
