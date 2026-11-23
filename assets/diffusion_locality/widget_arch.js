/* Visual Architecture Diagrams (U-Net & DiT Diffusion Data Flow) and CIFAR-10 Training Showcase */
(function () {
  "use strict";

  const K = window.DiffMemKit;
  if (!K) return;

  const { el, svgEl, clear, makeSeg, tooltip, tipRow, mountResponsive, observeAutoHide } = K;

  function clamp(v, lo, hi) {
    return v < lo ? lo : v > hi ? hi : v;
  }

  function decodeB64(b64) {
    const bin = atob(b64);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }

  function invertPerm(perm) {
    const inv = new Int32Array(perm.length);
    for (let i = 0; i < perm.length; i++) inv[perm[i]] = i;
    return inv;
  }

  // Exact cosine alpha_bar schedule matching diffmem.py (T = 1000, s = 0.008)
  function cosineAlphaBar(t) {
    const T = 1000;
    const s = 0.008;
    const f0 = Math.pow(Math.cos((s / (1 + s)) * (Math.PI / 2)), 2);
    const u = clamp(t, 0, T) / T;
    const ft = Math.pow(Math.cos(((u + s) / (1 + s)) * (Math.PI / 2)), 2);
    return clamp(ft / f0, 1e-5, 1 - 1e-5);
  }

  // Deterministic Gaussian noise generator (Mulberry32 + Box-Muller)
  function makeGaussianField(len, seed) {
    let a = seed >>> 0;
    const rand = () => {
      a |= 0;
      a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    const out = new Float32Array(len);
    for (let i = 0; i < len; i += 2) {
      const u1 = Math.max(1e-7, rand());
      const u2 = rand();
      const mag = Math.sqrt(-2 * Math.log(u1));
      out[i] = mag * Math.cos(2 * Math.PI * u2);
      if (i + 1 < len) out[i + 1] = mag * Math.sin(2 * Math.PI * u2);
    }
    return out;
  }

  function computeLocalCorr(bytes, C, H, W, invPerm) {
    const nPix = H * W;
    const lum = new Float32Array(nPix);
    let mu = 0;
    for (let i = 0; i < nPix; i++) {
      const v = C === 1 ? bytes[i] : 0.299 * bytes[i * 3] + 0.587 * bytes[i * 3 + 1] + 0.114 * bytes[i * 3 + 2];
      lum[i] = v;
      mu += v;
    }
    mu /= nPix;
    let num = 0, vx = 0, vy = 0;
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const p = y * W + x;
        const a = lum[invPerm ? invPerm[p] : p] - mu;
        if (x + 1 < W) {
          const b = lum[invPerm ? invPerm[p + 1] : p + 1] - mu;
          num += a * b; vx += a * a; vy += b * b;
        }
        if (y + 1 < H) {
          const b = lum[invPerm ? invPerm[p + W] : p + W] - mu;
          num += a * b; vx += a * a; vy += b * b;
        }
      }
    }
    const den = Math.sqrt(vx * vy);
    return den > 1e-9 ? num / den : 0;
  }

  function sizeCanvas(canvas, cssSize) {
    const rect = canvas.getBoundingClientRect();
    const css = rect.width > 16 ? rect.width : cssSize;
    const dpr = Math.min(window.devicePixelRatio || 1, 3);
    const px = Math.max(1, Math.round(css * dpr));
    if (canvas.width !== px || canvas.height !== px) {
      canvas.width = px;
      canvas.height = px;
    }
    return px;
  }

  const sharedOff = document.createElement("canvas");

  function drawRaster(canvas, bytes, C, H, W, opts) {
    const o = opts || {};
    const px = sizeCanvas(canvas, o.css || 96);
    if (sharedOff.width !== W || sharedOff.height !== H) {
      sharedOff.width = W;
      sharedOff.height = H;
    }
    const octx = sharedOff.getContext("2d");
    const img = octx.createImageData(W, H);
    const out = img.data;
    const nPix = H * W;
    const perm = o.perm || null;
    const noiseField = o.noiseField || null;
    const tStep = o.noiseT || 0;
    const pureNoise = !!o.pureNoise;
    const ab = tStep > 0 ? cosineAlphaBar(tStep) : 1.0;
    const sa = Math.sqrt(ab);
    const sb = Math.sqrt(1 - ab);

    for (let i = 0; i < nPix; i++) {
      const dst = (perm ? perm[i] : i) * 4;
      if (pureNoise && noiseField) {
        if (C === 1) {
          const v = clamp(Math.round((noiseField[i] * 0.24 + 0.5) * 255), 0, 255);
          out[dst] = v; out[dst + 1] = v; out[dst + 2] = v; out[dst + 3] = 255;
        } else {
          for (let c = 0; c < 3; c++) {
            out[dst + c] = clamp(Math.round((noiseField[i * 3 + c] * 0.24 + 0.5) * 255), 0, 255);
          }
          out[dst + 3] = 255;
        }
      } else if (C === 1) {
        let x0 = (bytes[i] / 255) * 2 - 1;
        if (tStep > 0 && noiseField) x0 = sa * x0 + sb * noiseField[i];
        const v = clamp(Math.round(((x0 + 1) * 0.5) * 255), 0, 255);
        out[dst] = v; out[dst + 1] = v; out[dst + 2] = v; out[dst + 3] = 255;
      } else {
        for (let c = 0; c < 3; c++) {
          let x0 = (bytes[i * 3 + c] / 255) * 2 - 1;
          if (tStep > 0 && noiseField) x0 = sa * x0 + sb * noiseField[i * 3 + c];
          out[dst + c] = clamp(Math.round(((x0 + 1) * 0.5) * 255), 0, 255);
        }
        out[dst + 3] = 255;
      }
    }
    octx.putImageData(img, 0, 0);

    const ctx = canvas.getContext("2d");
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, px, px);
    ctx.drawImage(sharedOff, 0, 0, px, px);

    const cell = px / W;

    // Optional DiT patch grid overlay
    if (o.gridP && o.gridP > 1) {
      ctx.save();
      ctx.strokeStyle = "rgba(255, 255, 255, 0.45)";
      ctx.lineWidth = Math.max(1, px * 0.008);
      for (let g = o.gridP; g < W; g += o.gridP) {
        const pos = Math.round(g * cell) + 0.5;
        ctx.beginPath(); ctx.moveTo(pos, 0); ctx.lineTo(pos, px); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(0, pos); ctx.lineTo(px, pos); ctx.stroke();
      }
      ctx.restore();
    }

    // Optional highlighted square patch (natural coords)
    if (o.patchBox) {
      const { x0, y0, size, color } = o.patchBox;
      ctx.save();
      ctx.strokeStyle = "rgba(0,0,0,0.72)";
      ctx.lineWidth = Math.max(2.6, cell * 0.42);
      ctx.strokeRect(x0 * cell, y0 * cell, size * cell, size * cell);
      ctx.strokeStyle = color;
      ctx.lineWidth = Math.max(1.5, cell * 0.24);
      ctx.strokeRect(x0 * cell, y0 * cell, size * cell, size * cell);
      ctx.restore();
    }

    // Optional scattered pixels (permuted coords)
    if (o.scatteredCells) {
      const { cells, color } = o.scatteredCells;
      ctx.save();
      for (const pt of cells) {
        ctx.fillStyle = color;
        ctx.globalAlpha = 0.35;
        ctx.fillRect(pt.x * cell, pt.y * cell, cell, cell);
        ctx.globalAlpha = 1;
        ctx.strokeStyle = "rgba(0,0,0,0.7)";
        ctx.lineWidth = Math.max(1.8, cell * 0.28);
        ctx.strokeRect(pt.x * cell, pt.y * cell, cell, cell);
        ctx.strokeStyle = color;
        ctx.lineWidth = Math.max(1.1, cell * 0.16);
        ctx.strokeRect(pt.x * cell, pt.y * cell, cell, cell);
      }
      ctx.restore();
    }
  }

  function bindBlockTip(node, title, rowsFn) {
    const show = (e) => {
      const rows = rowsFn();
      let html = `<div class="viz-tooltip__title">${title}</div>`;
      for (const r of rows) {
        html += tipRow(r[0], r[1], r[2]);
      }
      tooltip.show(e.clientX, e.clientY, html, node);
    };
    node.style.cursor = "pointer";
    node.addEventListener("mouseenter", show);
    node.addEventListener("mousemove", show);
    node.addEventListener("mouseleave", () => tooltip.hide());
  }

  /* =========================================================================
   * FIGURE A: U-Net + Diffusion Data Flow (#viz-arch-unet)
   * ========================================================================= */
  function mountArchUnet(container, data) {
    const ctrain = data && data.cifar_train;
    const hero = data && data.hero;
    if (!ctrain || !hero) return;

    const perm1024 = hero.perm1024;
    const picks = (ctrain.arch_picks || [0, 1, 2, 3, 4]).slice(0, 5).map((idx) => {
      const item = ctrain.showcase[idx];
      return { ...item, bytes: decodeB64(item.b64) };
    });

    const genPair = (ctrain.nn_pairs && ctrain.nn_pairs[0]) || null;
    const genNatBytes = genPair ? decodeB64(genPair.nat_b64) : picks[0].bytes;
    const genPermBytes = genPair ? decodeB64(genPair.perm_b64) : picks[0].bytes;

    const state = {
      imgIdx: 0,
      noiseT: 320,
      armFocus: "both", // "both" | "nat" | "perm"
      noiseSeed: 42,
    };
    let noiseField = makeGaussianField(32 * 32 * 3, state.noiseSeed);

    const header = el("div", { class: "viz-header" }, [
      el("div", null, [
        el("h3", {
          class: "viz-title",
          text: "Architecture & Diffusion Data Flow: Natural vs. Scrambled CIFAR-10 in the U-Net",
        }),
        el("p", {
          class: "viz-subtitle",
          text: "Trace a real CIFAR-10 training image before and after pixel permutation P through forward noising, the purely convolutional U-Net denoiser (3.83M params), and the reverse DDIM-50 sampling loop. Scrub timestep t or hover any block.",
        }),
      ]),
      el("span", { class: "viz-badge viz-badge--nat", text: "Conv U-Net · 32→64→128 · No Attention" }),
    ]);

    const imgSeg = makeSeg(
      picks.map((p, i) => ({ value: String(i), label: `${p.label[0].toUpperCase() + p.label.slice(1)} #${p.train_idx}` })),
      "0",
      (v) => {
        state.imgIdx = Number(v);
        redraw();
      }
    );

    const armSeg = makeSeg(
      [
        { value: "both", label: "Both arms" },
        { value: "nat", label: "Natural only" },
        { value: "perm", label: "Permuted only" },
      ],
      state.armFocus,
      (v) => {
        state.armFocus = v;
        redraw();
      }
    );

    const tSlider = el("input", {
      type: "range",
      class: "viz-slider",
      min: 0,
      max: 999,
      step: 5,
      value: state.noiseT,
      "aria-label": "Diffusion timestep t",
    });
    const tReadout = el("span", { class: "viz-readout" });
    const resampleBtn = el("button", {
      type: "button",
      class: "viz-btn",
      text: "New ε",
      onClick: () => {
        state.noiseSeed = (state.noiseSeed + 97) % 10000;
        noiseField = makeGaussianField(32 * 32 * 3, state.noiseSeed);
        redraw();
      },
    });

    const controls = el("div", { class: "viz-controls" }, [
      el("div", { class: "viz-control-group" }, [
        el("span", { class: "viz-control-label", text: "CIFAR-10 image" }),
        imgSeg,
      ]),
      el("div", { class: "viz-slider-wrap" }, [
        el("span", { class: "viz-control-label", text: "Noise step t" }),
        tSlider,
        tReadout,
        resampleBtn,
      ]),
      el("div", { class: "viz-control-group" }, [
        el("span", { class: "viz-control-label", text: "Highlight" }),
        armSeg,
      ]),
    ]);

    // Stage 1 canvases: clean natural x0 & clean scrambled Px0
    const cvsNatClean = el("canvas", { class: "arch-canvas" });
    const cvsPermClean = el("canvas", { class: "arch-canvas" });
    const natBranchBox = el("div", { class: "arch-img-card arch-img-card--nat" }, [
      cvsNatClean,
      el("div", { class: "arch-img-cap" }, [
        el("strong", { text: "Natural x₀" }),
        el("span", { text: "3×3 local patch" }),
      ]),
    ]);
    const permBranchBox = el("div", { class: "arch-img-card arch-img-card--perm" }, [
      cvsPermClean,
      el("div", { class: "arch-img-cap" }, [
        el("strong", { text: "Scrambled Px₀" }),
        el("span", { text: "9 scattered coords" }),
      ]),
    ]);
    const stage1 = el("div", { class: "arch-stage" }, [
      el("div", { class: "arch-stage__head" }, [
        el("span", { class: "arch-stage__num", text: "1" }),
        el("span", { text: "Input & Permutation P" }),
      ]),
      natBranchBox,
      el("div", { class: "arch-op-pill", text: "Fixed P (L₂ isometry · RGB intact)" }),
      permBranchBox,
    ]);

    // Stage 2 canvases: noisy xt (Natural) & noisy Pxt (Permuted)
    const cvsNatNoisy = el("canvas", { class: "arch-canvas" });
    const cvsPermNoisy = el("canvas", { class: "arch-canvas" });
    const snrLabel = el("div", { class: "arch-formula-sub" });
    const snrBarFill = el("div", { class: "arch-snr-fill" });
    const natNoisyBox = el("div", { class: "arch-img-card arch-img-card--nat" }, [
      cvsNatNoisy,
      el("div", { class: "arch-img-cap" }, [
        el("strong", { text: "Noisy xₜ" }),
        el("span", { text: "Natural arm" }),
      ]),
    ]);
    const permNoisyBox = el("div", { class: "arch-img-card arch-img-card--perm" }, [
      cvsPermNoisy,
      el("div", { class: "arch-img-cap" }, [
        el("strong", { text: "Noisy Pxₜ" }),
        el("span", { text: "Permuted arm" }),
      ]),
    ]);
    const stage2 = el("div", { class: "arch-stage" }, [
      el("div", { class: "arch-stage__head" }, [
        el("span", { class: "arch-stage__num arch-stage__num--diff", text: "2" }),
        el("span", { text: "Forward Diffusion" }),
      ]),
      el("div", { class: "arch-formula-pill", html: "x<sub>t</sub> = √ᾱ<sub>t</sub> x<sub>0</sub> + √(1−ᾱ<sub>t</sub>) ε" }),
      natNoisyBox,
      el("div", { class: "arch-snr-wrap" }, [
        el("div", { class: "arch-snr-track" }, [snrBarFill]),
        snrLabel,
      ]),
      permNoisyBox,
    ]);

    // Stage 3: Interactive U-Net SVG schematic
    const unetSvgWrap = el("div", { class: "arch-svg-wrap" });
    const stage3 = el("div", { class: "arch-stage arch-stage--core" }, [
      el("div", { class: "arch-stage__head" }, [
        el("span", { class: "arch-stage__num", text: "3" }),
        el("span", { text: "U-Net Denoiser ε̂_θ(xₜ, t) · 3.83M params" }),
      ]),
      unetSvgWrap,
    ]);

    // Stage 4: Predicted noise, MSE loss & Reverse DDIM-50 sampling loop
    const cvsEpsPred = el("canvas", { class: "arch-canvas arch-canvas--sm" });
    const cvsGenNat = el("canvas", { class: "arch-canvas arch-canvas--mini" });
    const cvsGenPerm = el("canvas", { class: "arch-canvas arch-canvas--mini" });
    const stage4 = el("div", { class: "arch-stage" }, [
      el("div", { class: "arch-stage__head" }, [
        el("span", { class: "arch-stage__num arch-stage__num--diff", text: "4" }),
        el("span", { text: "Loss & Reverse DDIM" }),
      ]),
      el("div", { class: "arch-img-card" }, [
        cvsEpsPred,
        el("div", { class: "arch-img-cap" }, [
          el("strong", { text: "Predicted noise ε̂_θ" }),
          el("span", { text: "32×32×3 output" }),
        ]),
      ]),
      el("div", { class: "arch-loss-box" }, [
        el("div", { class: "arch-loss-title", text: "Training Objective" }),
        el("div", { class: "arch-formula-pill", html: "ℒ = ‖ε − ε̂<sub>θ</sub>(x<sub>t</sub>, t)‖<sub>2</sub><sup>2</sup>" }),
      ]),
      el("div", { class: "arch-ddim-box" }, [
        el("div", { class: "arch-loss-title", text: "Reverse Sampling (DDIM-50)" }),
        el("div", { class: "arch-ddim-sub", text: "x_T ~ N(0,I) ↺ 50 steps (t: 999→0) → x̂₀, then P⁻¹x̂₀" }),
        el("div", { class: "arch-mini-pair" }, [
          el("div", { class: "arch-mini-item" }, [cvsGenNat, el("span", { text: "Nat 300k" })]),
          el("div", { class: "arch-mini-item" }, [cvsGenPerm, el("span", { text: "Perm→P⁻¹" })]),
        ]),
      ]),
    ]);

    const flowGrid = el("div", { class: "arch-flow-grid" }, [stage1, stage2, stage3, stage4]);

    const figcaption = el("div", { class: "viz-figcaption" }, [
      el("b", { text: "Where diffusion meets convolutional locality: " }),
      "Both arms share the exact same forward noising schedule ",
      el("span", { style: "font-family:var(--mono)", text: "q(xₜ|x₀)" }),
      ", sinusoidal timestep injection ",
      el("span", { style: "font-family:var(--mono)", text: "t → MLP(256)" }),
      ", MSE noise-prediction loss, and 50-step DDIM reverse sampler. The only difference happens at the ",
      el("strong", { text: "3×3 convolutions" }),
      ": natural inputs feed coherent 9-pixel neighbourhoods into the first layers, whereas permuted inputs feed 9 scattered coordinates — slowing optimisation by ~2× even though the 8×8 bottleneck spans the whole 32×32 canvas.",
    ]);

    clear(container);
    container.appendChild(header);
    container.appendChild(controls);
    container.appendChild(flowGrid);
    container.appendChild(figcaption);

    // Wire tooltips on stages 1, 2, 4
    bindBlockTip(natBranchBox, "Natural Input x₀ (32×32×3)", () => {
      const p = picks[state.imgIdx];
      return [
        ["CIFAR-10 class", `${p.label} (train #${p.train_idx})`],
        ["Pixel grid", "32 × 32 × 3 RGB in [−1, 1]"],
        ["3×3 neighbourhood", "Strongly correlated adjacent pixels"],
      ];
    });
    bindBlockTip(permBranchBox, "Pixel-Permuted Input Px₀", () => [
      ["Permutation", "Fixed perm_seed = 1234 on 1,024 positions"],
      ["RGB channels", "Grouped per pixel (isometry ‖Px−Py‖₂ = ‖x−y‖₂)"],
      ["3×3 neighbourhood", "9 scattered coordinates across the image"],
    ]);
    bindBlockTip(stage2, "Forward Diffusion Process q(xₜ | x₀)", () => {
      const ab = cosineAlphaBar(state.noiseT);
      return [
        ["Schedule", "Cosine VP (T = 1000, s = 0.008)"],
        ["Signal √ᾱₜ", Math.sqrt(ab).toFixed(3)],
        ["Noise √(1−ᾱₜ)", Math.sqrt(1 - ab).toFixed(3)],
        ["Symmetry", "Isotropic ε ~ N(0, I) is invariant under P"],
      ];
    });

    function drawUnetSvg() {
      const T = K.theme();
      clear(unetSvgWrap);
      const W = 430;
      const H = 294;
      const svg = svgEl("svg", {
        viewBox: `0 0 ${W} ${H}`,
        width: "100%",
        style: "display:block;height:auto;max-height:310px;user-select:none;",
      });

      const defs = svgEl("defs", null, [
        svgEl("marker", {
          id: "arch-arrow-muted", viewBox: "0 0 8 8", refX: 7, refY: 4,
          markerWidth: 6, markerHeight: 6, orient: "auto",
        }, [svgEl("path", { d: "M0,1 L7,4 L0,7 Z", fill: T.muted })]),
        svgEl("marker", {
          id: "arch-arrow-gold", viewBox: "0 0 8 8", refX: 7, refY: 4,
          markerWidth: 5.5, markerHeight: 5.5, orient: "auto",
        }, [svgEl("path", { d: "M0,1 L7,4 L0,7 Z", fill: T.gold })]),
        svgEl("marker", {
          id: "arch-arrow-accent", viewBox: "0 0 8 8", refX: 7, refY: 4,
          markerWidth: 6, markerHeight: 6, orient: "auto",
        }, [svgEl("path", { d: "M0,1 L7,4 L0,7 Z", fill: state.armFocus === "perm" ? T.perm : T.nat })]),
      ]);
      svg.appendChild(defs);

      const accentStroke = state.armFocus === "perm" ? T.perm : T.nat;
      const accentSoft = state.armFocus === "perm" ? T.permSoft : T.natSoft;

      // Top Timestep Embedding Conditioning Bar
      const tBar = svgEl("g");
      tBar.appendChild(svgEl("rect", {
        x: 18, y: 10, width: 394, height: 32, rx: 7,
        fill: T.surface, stroke: T.gold, "stroke-width": 1.5,
      }));
      tBar.appendChild(svgEl("text", {
        x: 215, y: 24, "text-anchor": "middle", fill: T.gold,
        "font-family": T.sans, "font-size": 10.2, "font-weight": 700,
        text: `Timestep Conditioning: t = ${state.noiseT} → sin/cos(256) → MLP`,
      }));
      tBar.appendChild(svgEl("text", {
        x: 215, y: 36, "text-anchor": "middle", fill: T.muted,
        "font-family": T.mono, "font-size": 8.2,
        text: "injected into every ResBlock (encoder · mid · decoder)",
      }));
      bindBlockTip(tBar, "Diffusion Timestep Embedding (tdim = 256)", () => [
        ["Current step t", String(state.noiseT)],
        ["Encoding", "Sinusoidal(t) → Linear(256) → SiLU → Linear(256)"],
        ["Conditioning", "Per-ResBlock Linear(256 → C_out) added to feature maps"],
      ]);
      svg.appendChild(tBar);

      // Dashed gold injection arrows down from tBar to blocks
      const injXs = [68, 134, 215, 296, 362];
      const injYs = [82, 146, 214, 146, 82];
      injXs.forEach((ix, k) => {
        svg.appendChild(svgEl("line", {
          x1: ix, y1: 42, x2: ix, y2: injYs[k] - 4,
          stroke: T.gold, "stroke-width": 1.15, "stroke-dasharray": "3,3",
          opacity: 0.72, "marker-end": "url(#arch-arrow-gold)",
        }));
      });

      // U-Net blocks geometry
      const blocks = [
        {
          id: "enc0", x: 16, y: 82, w: 104, h: 48,
          title: "Conv 3×3 + Down 1", sub: "32×32 · 32 ch", detail: "2× ResBlock",
          fill: accentSoft, stroke: accentStroke,
          tip: [
            ["Input conv", "Conv2d(3 → 32, k=3, pad=1) + 2× ResBlock"],
            ["Resolution", "32 × 32 (MNIST: 28 × 28)"],
            ["Locality impact", "Sees 3×3 pixel patches — directly disrupted by P"],
          ],
        },
        {
          id: "enc1", x: 84, y: 146, w: 100, h: 46,
          title: "Down 2 (↓ stride 2)", sub: "16×16 · 64 ch", detail: "2× ResBlock",
          fill: T.surface, stroke: accentStroke,
          tip: [
            ["Downsample", "Conv2d(32 → 32, k=3, s=2) + 2× ResBlock(64ch)"],
            ["Resolution", "16 × 16 (MNIST: 14 × 14)"],
            ["Skip output", "Concatenated into Up 2 decoder blocks"],
          ],
        },
        {
          id: "mid", x: 158, y: 214, w: 114, h: 52,
          title: "Bottleneck (Mid)", sub: "8×8 · 128 ch", detail: "2× ResBlock · Full RF",
          fill: accentSoft, stroke: T.accent,
          tip: [
            ["Bottleneck", "Down(s=2) → 2× ResBlock(128ch) + 2× Mid ResBlock"],
            ["Resolution", "8 × 8 (MNIST: 7 × 7)"],
            ["Receptive field", "Covers entire 32×32 image — enables eventual fit of P"],
          ],
        },
        {
          id: "dec1", x: 246, y: 146, w: 100, h: 46,
          title: "Up 2 (↑ ×2 + concat)", sub: "16×16 · 64 ch", detail: "3× ResBlock",
          fill: T.surface, stroke: accentStroke,
          tip: [
            ["Upsample", "Nearest ×2 + Conv3×3 + 3× ResBlock(concat skip)"],
            ["Resolution", "16 × 16 (MNIST: 14 × 14)"],
          ],
        },
        {
          id: "dec0", x: 310, y: 82, w: 104, h: 48,
          title: "Up 1 + Out Conv 3×3", sub: "32×32 · 32→3 ch", detail: "3× ResBlock → ε̂_θ",
          fill: accentSoft, stroke: accentStroke,
          tip: [
            ["Output head", "GroupNorm → SiLU → Conv2d(32 → 3, k=3)"],
            ["Output tensor", "Predicted noise ε̂_θ(xₜ, t) ∈ ℝ^(3×32×32)"],
          ],
        },
      ];

      // Skip connections (dashed horizontal arcs between enc0->dec0 and enc1->dec1)
      const skipPairs = [
        { x1: 120, y1: 106, x2: 307, y2: 106, lbl: "skip concat (32×32)" },
        { x1: 184, y1: 169, x2: 243, y2: 169, lbl: "skip" },
      ];
      for (const sp of skipPairs) {
        svg.appendChild(svgEl("line", {
          x1: sp.x1, y1: sp.y1, x2: sp.x2, y2: sp.y2,
          stroke: T.muted, "stroke-width": 1.3, "stroke-dasharray": "4,3",
          "marker-end": "url(#arch-arrow-muted)",
        }));
        svg.appendChild(svgEl("text", {
          x: (sp.x1 + sp.x2) / 2, y: sp.y1 - 5,
          "text-anchor": "middle", fill: T.muted,
          "font-family": T.mono, "font-size": 8.8,
          text: sp.lbl,
        }));
      }

      // Main U-path connectors: enc0 -> enc1 -> mid -> dec1 -> dec0
      const pathSegs = [
        "M68,130 L68,169 L81,169",
        "M134,192 L134,240 L155,240",
        "M272,240 L296,240 L296,195",
        "M346,169 L362,169 L362,133",
      ];
      for (const d of pathSegs) {
        svg.appendChild(svgEl("path", {
          d, fill: "none", stroke: accentStroke, "stroke-width": 1.8,
          "marker-end": "url(#arch-arrow-accent)",
        }));
      }

      // Draw interactive U-Net blocks
      for (const b of blocks) {
        const g = svgEl("g");
        const rect = svgEl("rect", {
          x: b.x, y: b.y, width: b.w, height: b.h, rx: 7,
          fill: b.fill, stroke: b.stroke, "stroke-width": 1.5,
        });
        g.appendChild(rect);
        g.appendChild(svgEl("text", {
          x: b.x + b.w / 2, y: b.y + 15, "text-anchor": "middle",
          fill: T.text, "font-family": T.sans, "font-size": 10, "font-weight": 700,
          text: b.title,
        }));
        g.appendChild(svgEl("text", {
          x: b.x + b.w / 2, y: b.y + 29, "text-anchor": "middle",
          fill: b.stroke, "font-family": T.mono, "font-size": 9.3, "font-weight": 600,
          text: b.sub,
        }));
        g.appendChild(svgEl("text", {
          x: b.x + b.w / 2, y: b.y + 41, "text-anchor": "middle",
          fill: T.muted, "font-family": T.sans, "font-size": 8.6,
          text: b.detail,
        }));
        g.addEventListener("mouseenter", () => rect.setAttribute("stroke-width", "2.4"));
        g.addEventListener("mouseleave", () => rect.setAttribute("stroke-width", "1.5"));
        bindBlockTip(g, b.title, () => b.tip);
        svg.appendChild(g);
      }

      // Bottom annotation note inside SVG
      svg.appendChild(svgEl("text", {
        x: 215, y: 285, "text-anchor": "middle", fill: T.muted,
        "font-family": T.sans, "font-size": 8.8,
        text: "Same 3.83M U-Net trained on Natural vs. Permuted · Hover any block",
      }));

      unetSvgWrap.appendChild(svg);
    }

    function redraw() {
      const T = K.theme();
      const pick = picks[state.imgIdx];
      const ab = cosineAlphaBar(state.noiseT);
      const sa = Math.sqrt(ab);
      const sb = Math.sqrt(1 - ab);
      tReadout.textContent = `t = ${state.noiseT}`;
      snrLabel.textContent = `√ᾱ = ${sa.toFixed(2)} · √(1−ᾱ) = ${sb.toFixed(2)}`;
      snrBarFill.style.width = `${Math.round(ab * 100)}%`;

      // Highlight a 3x3 patch near center (14, 14) on natural & its 9 scattered destinations on permuted
      const px0 = 14, py0 = 14;
      const scattered = [];
      for (let dy = 0; dy < 3; dy++) {
        for (let dx = 0; dx < 3; dx++) {
          const dst = perm1024[(py0 + dy) * 32 + (px0 + dx)];
          scattered.push({ x: dst % 32, y: Math.floor(dst / 32) });
        }
      }

      const dimNat = state.armFocus === "perm";
      const dimPerm = state.armFocus === "nat";
      natBranchBox.style.opacity = dimNat ? "0.35" : "1";
      natNoisyBox.style.opacity = dimNat ? "0.35" : "1";
      permBranchBox.style.opacity = dimPerm ? "0.35" : "1";
      permNoisyBox.style.opacity = dimPerm ? "0.35" : "1";

      drawRaster(cvsNatClean, pick.bytes, 3, 32, 32, {
        css: 88,
        patchBox: { x0: px0, y0: py0, size: 3, color: T.nat },
      });
      drawRaster(cvsPermClean, pick.bytes, 3, 32, 32, {
        css: 88,
        perm: perm1024,
        scatteredCells: { cells: scattered, color: T.perm },
      });
      drawRaster(cvsNatNoisy, pick.bytes, 3, 32, 32, {
        css: 88,
        noiseField,
        noiseT: state.noiseT,
      });
      drawRaster(cvsPermNoisy, pick.bytes, 3, 32, 32, {
        css: 88,
        perm: perm1024,
        noiseField,
        noiseT: state.noiseT,
      });
      drawRaster(cvsEpsPred, pick.bytes, 3, 32, 32, {
        css: 72,
        noiseField,
        pureNoise: true,
      });
      drawRaster(cvsGenNat, genNatBytes, 3, 32, 32, { css: 48 });
      drawRaster(cvsGenPerm, genPermBytes, 3, 32, 32, { css: 48 });

      drawUnetSvg();
    }

    tSlider.addEventListener("input", () => {
      state.noiseT = Number(tSlider.value);
      redraw();
    });

    redraw();
    mountResponsive(container, redraw);
    observeAutoHide(container);
  }

  /* =========================================================================
   * FIGURE B: DiT Architecture & Patch Locality Dial (#viz-arch-dit)
   * ========================================================================= */
  function mountArchDit(container, data) {
    const ctrain = data && data.cifar_train;
    const hero = data && data.hero;
    if (!ctrain || !hero) return;

    const c10Pick = { ...ctrain.showcase[0], bytes: decodeB64(ctrain.showcase[0].b64) }; // vivid blue car #19
    const mnRaw = (ctrain.mnist_arch_picks && ctrain.mnist_arch_picks[0]) || hero.conditions.mnist_n2048.train_samples[0];
    const mnPick = { ...mnRaw, bytes: decodeB64(mnRaw.b64) };

    const state = {
      patch: 2,          // 1 | 2 | 4
      dataset: "cifar",  // "cifar" | "mnist"
      noiseT: 260,
    };

    const noiseC10 = makeGaussianField(32 * 32 * 3, 1337);
    const noiseMN = makeGaussianField(28 * 28, 1337);

    const header = el("div", { class: "viz-header" }, [
      el("div", null, [
        el("h3", {
          class: "viz-title",
          text: "Diffusion Transformer (DiT) Data Flow: Patchifying Natural vs. Scrambled Images",
        }),
        el("p", {
          class: "viz-subtitle",
          text: "Dial patch size p ∈ {1, 2, 4} to see how spatial locality enters only at the patchify boundary (and positional embedding), while all 6 adaLN-Zero transformer blocks mix tokens globally via self-attention.",
        }),
      ]),
      el("span", { class: "viz-badge", text: "DiT · w=192 · d=6 · 6 heads · ~4.7M" }),
    ]);

    const patchSeg = makeSeg(
      [
        { value: "1", label: "p = 1 (Exact Null)" },
        { value: "2", label: "p = 2 (Local 2×2)" },
        { value: "4", label: "p = 4 (Coarse 4×4)" },
      ],
      String(state.patch),
      (v) => {
        state.patch = Number(v);
        redraw();
      }
    );

    const dsSeg = makeSeg(
      [
        { value: "cifar", label: "CIFAR-10 (32×32 RGB)" },
        { value: "mnist", label: "MNIST (28×28, DiT exp)" },
      ],
      state.dataset,
      (v) => {
        state.dataset = v;
        redraw();
      }
    );

    const tSlider = el("input", {
      type: "range",
      class: "viz-slider",
      min: 0,
      max: 999,
      step: 5,
      value: state.noiseT,
      "aria-label": "DiT diffusion timestep t",
    });
    const tReadout = el("span", { class: "viz-readout" });

    const controls = el("div", { class: "viz-controls" }, [
      el("div", { class: "viz-control-group" }, [
        el("span", { class: "viz-control-label", text: "Patch size" }),
        patchSeg,
      ]),
      el("div", { class: "viz-control-group" }, [
        el("span", { class: "viz-control-label", text: "Input preview" }),
        dsSeg,
      ]),
      el("div", { class: "viz-slider-wrap" }, [
        el("span", { class: "viz-control-label", text: "Noise t" }),
        tSlider,
        tReadout,
      ]),
    ]);

    // Stage 1: Noisy input with p x p patch grid
    const cvsNat = el("canvas", { class: "arch-canvas" });
    const cvsPerm = el("canvas", { class: "arch-canvas" });
    const stage1 = el("div", { class: "arch-stage" }, [
      el("div", { class: "arch-stage__head" }, [
        el("span", { class: "arch-stage__num", text: "1" }),
        el("span", { text: "Noisy Input + p×p Grid" }),
      ]),
      el("div", { class: "arch-img-card arch-img-card--nat" }, [
        cvsNat,
        el("div", { class: "arch-img-cap" }, [
          el("strong", { text: "Natural xₜ" }),
          el("span", { text: "contiguous p×p patch" }),
        ]),
      ]),
      el("div", { class: "arch-img-card arch-img-card--perm" }, [
        cvsPerm,
        el("div", { class: "arch-img-cap" }, [
          el("strong", { text: "Permuted Pxₜ" }),
          el("span", { text: "same grid after P" }),
        ]),
      ]),
    ]);

    // Stage 2: Patchify + Positional Embedding card
    const tokenSpecVal = el("div", { class: "arch-formula-pill" });
    const posEmbBadge = el("div", { class: "arch-op-pill" });
    const patchNote = el("div", { class: "arch-ddim-sub" });
    const tokenStripWrap = el("div", { class: "arch-token-strip" });
    const stage2 = el("div", { class: "arch-stage" }, [
      el("div", { class: "arch-stage__head" }, [
        el("span", { class: "arch-stage__num", text: "2" }),
        el("span", { text: "Patchify + Pos Emb" }),
      ]),
      tokenSpecVal,
      tokenStripWrap,
      posEmbBadge,
      patchNote,
    ]);

    // Stage 3: 6x adaLN-Zero DiT Blocks SVG
    const ditSvgWrap = el("div", { class: "arch-svg-wrap" });
    const stage3 = el("div", { class: "arch-stage arch-stage--core" }, [
      el("div", { class: "arch-stage__head" }, [
        el("span", { class: "arch-stage__num arch-stage__num--diff", text: "3" }),
        el("span", { text: "6× adaLN-Zero DiT Blocks (Global Attention)" }),
      ]),
      ditSvgWrap,
    ]);

    // Stage 4: Unpatchify + Noise prediction + DDIM
    const cvsEps = el("canvas", { class: "arch-canvas arch-canvas--sm" });
    const rhoBadge = el("div", { class: "arch-op-pill" });
    const stage4 = el("div", { class: "arch-stage" }, [
      el("div", { class: "arch-stage__head" }, [
        el("span", { class: "arch-stage__num arch-stage__num--diff", text: "4" }),
        el("span", { text: "Unpatchify & DDIM" }),
      ]),
      el("div", { class: "arch-formula-pill", html: "adaLN(h,t) → Linear → Unpatch" }),
      el("div", { class: "arch-img-card" }, [
        cvsEps,
        el("div", { class: "arch-img-cap" }, [
          el("strong", { text: "Predicted noise ε̂_θ" }),
          el("span", { text: "MSE ‖ε − ε̂_θ‖² + DDIM-50" }),
        ]),
      ]),
      el("div", { class: "arch-loss-box" }, [
        el("div", { class: "arch-loss-title", text: "Measured Slowdown (MNIST)" }),
        rhoBadge,
      ]),
    ]);

    const flowGrid = el("div", { class: "arch-flow-grid" }, [stage1, stage2, stage3, stage4]);

    const figcaption = el("div", { class: "viz-figcaption" }, [
      el("b", { text: "Why DiT isolates locality cleanly: " }),
      "Unlike a U-Net whose 3×3 kernels enforce locality at every layer, a DiT mixes all tokens globally in every self-attention layer. At ",
      el("strong", { text: "p = 1 with learned positions" }),
      ", permuting pixels merely reorders the token sequence and position table — giving exact permutation equivariance (max fp32 diff ",
      el("span", { style: "font-family:var(--mono)", text: "2.4×10⁻⁷" }),
      ", ",
      el("span", { style: "font-family:var(--mono)", text: "ρ = 0" }),
      "). Grouping pixels into ",
      el("strong", { text: "p = 2" }),
      " patches with fixed 2D sin-cos positions reintroduces a local patch prior (",
      el("span", { style: "font-family:var(--mono)", text: "ρ = 0.55" }),
      "), still weaker than the purely convolutional U-Net (",
      el("span", { style: "font-family:var(--mono)", text: "ρ = 0.93" }),
      ").",
    ]);

    clear(container);
    container.appendChild(header);
    container.appendChild(controls);
    container.appendChild(flowGrid);
    container.appendChild(figcaption);

    function drawDitSvg(nTok) {
      const T = K.theme();
      clear(ditSvgWrap);
      const W = 420;
      const H = 290;
      const svg = svgEl("svg", {
        viewBox: `0 0 ${W} ${H}`,
        width: "100%",
        style: "display:block;height:auto;max-height:305px;user-select:none;",
      });

      const defs = svgEl("defs", null, [
        svgEl("marker", {
          id: "dit-arrow-gold", viewBox: "0 0 8 8", refX: 7, refY: 4,
          markerWidth: 5.5, markerHeight: 5.5, orient: "auto",
        }, [svgEl("path", { d: "M0,1 L7,4 L0,7 Z", fill: T.gold })]),
        svgEl("marker", {
          id: "dit-arrow-nat", viewBox: "0 0 8 8", refX: 7, refY: 4,
          markerWidth: 6, markerHeight: 6, orient: "auto",
        }, [svgEl("path", { d: "M0,1 L7,4 L0,7 Z", fill: T.nat })]),
      ]);
      svg.appendChild(defs);

      // Outer 6x block frame
      svg.appendChild(svgEl("rect", {
        x: 12, y: 54, width: 396, height: 214, rx: 10,
        fill: T.cardAlt, stroke: T.border, "stroke-width": 1.3, "stroke-dasharray": "5,3",
      }));
      svg.appendChild(svgEl("text", {
        x: 24, y: 71, fill: T.muted, "font-family": T.mono, "font-size": 9.5, "font-weight": 700,
        text: `Repeat ×6 layers · Token tensor h ∈ ℝ^(${nTok} × 192)`,
      }));

      // Top Timestep adaLN-Zero Bar
      const tBar = svgEl("g");
      tBar.appendChild(svgEl("rect", {
        x: 16, y: 8, width: 388, height: 34, rx: 7,
        fill: T.surface, stroke: T.gold, "stroke-width": 1.5,
      }));
      tBar.appendChild(svgEl("text", {
        x: 210, y: 22, "text-anchor": "middle", fill: T.gold,
        "font-family": T.sans, "font-size": 10.2, "font-weight": 700,
        text: `Diffusion Time t = ${state.noiseT} → MLP(256) → adaLN-Zero`,
      }));
      tBar.appendChild(svgEl("text", {
        x: 210, y: 35, "text-anchor": "middle", fill: T.muted,
        "font-family": T.mono, "font-size": 8.1,
        text: "emits (s₁, b₁, g₁) for Attn & (s₂, b₂, g₂) for MLP (zero-init)",
      }));
      bindBlockTip(tBar, "adaLN-Zero Diffusion Conditioning", () => [
        ["Formula", "x = LN(h)·(1 + s) + b,  h ← h + g · sublayer(x)"],
        ["Zero init", "Gate weights initialized to 0 → identity block at step 0"],
        ["Diffusion role", "Steers every attention & MLP block for noise level t"],
      ]);
      svg.appendChild(tBar);

      // Gold arrows into Attention & MLP
      svg.appendChild(svgEl("line", {
        x1: 120, y1: 42, x2: 120, y2: 86,
        stroke: T.gold, "stroke-width": 1.2, "stroke-dasharray": "3,3",
        "marker-end": "url(#dit-arrow-gold)",
      }));
      svg.appendChild(svgEl("line", {
        x1: 304, y1: 42, x2: 304, y2: 86,
        stroke: T.gold, "stroke-width": 1.2, "stroke-dasharray": "3,3",
        "marker-end": "url(#dit-arrow-gold)",
      }));

      // Left Sub-block: Multi-Head Self-Attention with All-to-All Token Web
      const attnG = svgEl("g");
      const attnRect = svgEl("rect", {
        x: 26, y: 88, width: 178, height: 158, rx: 8,
        fill: T.surface, stroke: T.nat, "stroke-width": 1.5,
      });
      attnG.appendChild(attnRect);
      attnG.appendChild(svgEl("text", {
        x: 115, y: 106, "text-anchor": "middle", fill: T.text,
        "font-family": T.sans, "font-size": 10.5, "font-weight": 700,
        text: "Global Self-Attention",
      }));
      attnG.appendChild(svgEl("text", {
        x: 115, y: 120, "text-anchor": "middle", fill: T.nat,
        "font-family": T.mono, "font-size": 8.8, "font-weight": 600,
        text: "6 heads · global all-to-all",
      }));

      // Mini all-to-all bipartite/complete web of 6 nodes showing global mixing (inside attnG so it renders ON TOP of attnRect!)
      const nodeXs = [48, 74, 100, 128, 154, 180];
      const yTop = 148;
      const yBot = 194;
      for (let i = 0; i < nodeXs.length; i++) {
        for (let j = 0; j < nodeXs.length; j++) {
          attnG.appendChild(svgEl("line", {
            x1: nodeXs[i], y1: yTop, x2: nodeXs[j], y2: yBot,
            stroke: i === 2 ? T.perm : T.nat,
            "stroke-width": i === 2 ? 1.15 : 0.6,
            opacity: i === 2 ? 0.72 : 0.28,
          }));
        }
      }
      for (let i = 0; i < nodeXs.length; i++) {
        attnG.appendChild(svgEl("circle", {
          cx: nodeXs[i], cy: yTop, r: 4.2,
          fill: i === 2 ? T.perm : T.nat, stroke: T.surface, "stroke-width": 1.2,
        }));
        attnG.appendChild(svgEl("circle", {
          cx: nodeXs[i], cy: yBot, r: 4.2,
          fill: T.nat, stroke: T.surface, "stroke-width": 1.2,
        }));
      }
      attnG.appendChild(svgEl("text", {
        x: 115, y: 218, "text-anchor": "middle", fill: T.muted,
        "font-family": T.sans, "font-size": 9,
        text: "No 3×3 spatial kernel — distant",
      }));
      attnG.appendChild(svgEl("text", {
        x: 115, y: 230, "text-anchor": "middle", fill: T.muted,
        "font-family": T.sans, "font-size": 9,
        text: "scrambled tokens interact in Layer 1",
      }));
      attnG.addEventListener("mouseenter", () => attnRect.setAttribute("stroke-width", "2.4"));
      attnG.addEventListener("mouseleave", () => attnRect.setAttribute("stroke-width", "1.5"));
      bindBlockTip(attnG, "Multi-Head Self-Attention (6 heads, d=192)", () => [
        ["Receptive field", "Global (all N tokens attend to all N tokens in Layer 1)"],
        ["Permutation symmetry", "Attention itself is strictly permutation-equivariant"],
        ["Where locality lives", state.patch === 1 ? "Nowhere (p=1 learned pos = exact null)" : `Only inside ${state.patch}×${state.patch} patch + 2D sin-cos pos`],
      ]);
      svg.appendChild(attnG);

      // Connector arrow Attn -> MLP
      svg.appendChild(svgEl("line", {
        x1: 204, y1: 166, x2: 226, y2: 166,
        stroke: T.nat, "stroke-width": 1.8, "marker-end": "url(#dit-arrow-nat)",
      }));

      // Right Sub-block: Per-token MLP
      const mlpG = svgEl("g");
      const mlpRect = svgEl("rect", {
        x: 230, y: 88, width: 164, height: 158, rx: 8,
        fill: T.surface, stroke: T.accent, "stroke-width": 1.5,
      });
      mlpG.appendChild(mlpRect);
      mlpG.appendChild(svgEl("text", {
        x: 312, y: 108, "text-anchor": "middle", fill: T.text,
        "font-family": T.sans, "font-size": 10.5, "font-weight": 700,
        text: "Pointwise Token MLP",
      }));
      mlpG.appendChild(svgEl("text", {
        x: 312, y: 123, "text-anchor": "middle", fill: T.accent,
        "font-family": T.mono, "font-size": 9.2, "font-weight": 600,
        text: "192 → 768 (GELU) → 192",
      }));
      // Visual expansion bars inside MLP
      const bars = [
        { y: 142, w: 124, lbl: "LN + adaLN (192)" },
        { y: 168, w: 142, lbl: "GELU hidden (768)" },
        { y: 194, w: 128, lbl: "× gate g₂ + residual" },
      ];
      for (const br of bars) {
        mlpG.appendChild(svgEl("rect", {
          x: 312 - br.w / 2, y: br.y, width: br.w, height: 18, rx: 4,
          fill: T.permSoft, stroke: T.accent, "stroke-width": 1,
        }));
        mlpG.appendChild(svgEl("text", {
          x: 312, y: br.y + 12, "text-anchor": "middle", fill: T.text,
          "font-family": T.mono, "font-size": 8.5, text: br.lbl,
        }));
      }
      mlpG.appendChild(svgEl("text", {
        x: 312, y: 230, "text-anchor": "middle", fill: T.muted,
        "font-family": T.sans, "font-size": 9,
        text: "Applied identically per token",
      }));
      mlpG.addEventListener("mouseenter", () => mlpRect.setAttribute("stroke-width", "2.4"));
      mlpG.addEventListener("mouseleave", () => mlpRect.setAttribute("stroke-width", "1.5"));
      bindBlockTip(mlpG, "Token-wise Feed-Forward MLP", () => [
        ["Dimensions", "192 → 768 (4× width, GELU) → 192"],
        ["Gating", "h ← h + g₂ · MLP(LN(h)·(1+s₂) + b₂)"],
        ["Symmetry", "Applied independently to every token"],
      ]);
      svg.appendChild(mlpG);

      svg.appendChild(svgEl("text", {
        x: 210, y: 282, "text-anchor": "middle", fill: T.muted,
        "font-family": T.sans, "font-size": 9.3,
        text: "Hover any block for equations · Toggle patch size p above to inspect locality",
      }));

      ditSvgWrap.appendChild(svg);
    }

    function redraw() {
      const T = K.theme();
      const isCifar = state.dataset === "cifar";
      const C = isCifar ? 3 : 1;
      const H = isCifar ? 32 : 28;
      const W = H;
      const p = state.patch;
      const perm = isCifar ? hero.perm1024 : hero.perm784;
      const bytes = isCifar ? c10Pick.bytes : mnPick.bytes;
      const nField = isCifar ? noiseC10 : noiseMN;
      const gh = H / p;
      const nTok = gh * gh;
      const patchDim = C * p * p;

      tReadout.textContent = `t = ${state.noiseT}`;

      // Highlight one patch near center
      const px0 = Math.floor(gh / 2) * p;
      const py0 = Math.floor(gh / 2) * p;
      const scattered = [];
      for (let dy = 0; dy < p; dy++) {
        for (let dx = 0; dx < p; dx++) {
          const dst = perm[(py0 + dy) * W + (px0 + dx)];
          scattered.push({ x: dst % W, y: Math.floor(dst / W) });
        }
      }

      drawRaster(cvsNat, bytes, C, H, W, {
        css: 92,
        noiseField: nField,
        noiseT: state.noiseT,
        gridP: p,
        patchBox: { x0: px0, y0: py0, size: p, color: T.nat },
      });
      drawRaster(cvsPerm, bytes, C, H, W, {
        css: 92,
        perm,
        noiseField: nField,
        noiseT: state.noiseT,
        gridP: p,
        scatteredCells: { cells: scattered, color: T.perm },
      });
      drawRaster(cvsEps, bytes, C, H, W, {
        css: 72,
        noiseField: nField,
        pureNoise: true,
      });

      tokenSpecVal.textContent = `${nTok} tokens × (${C}·${p}×${p} = ${patchDim}) → d=192`;

      // Visual token chip strip
      clear(tokenStripWrap);
      for (let i = 0; i < 6; i++) {
        tokenStripWrap.appendChild(el("span", {
          class: `arch-tok-chip ${i === 2 ? "arch-tok-chip--hi" : ""}`,
          text: i === 5 ? `z_${nTok}` : `z_${i + 1}`,
        }));
      }

      if (p === 1) {
        posEmbBadge.textContent = "Learned Pos Emb · Exact Null";
        posEmbBadge.style.color = T.nat;
        patchNote.textContent = "1 pixel = 1 token. Permuting pixels only relabels tokens & learned position rows → exact symmetry.";
        rhoBadge.textContent = "ρ = 0.00 (exact) · Δ = 0.00";
        rhoBadge.style.color = T.nat;
      } else if (p === 2) {
        posEmbBadge.textContent = "Fixed 2D Sin-Cos Pos Emb";
        posEmbBadge.style.color = T.perm;
        patchNote.textContent = "Each 2×2 token mixes 4 coherent pixels (Natural) vs. 4 scattered pixels (Permuted).";
        rhoBadge.textContent = "ρ = 0.55 · Δ(100k) = −0.39";
        rhoBadge.style.color = T.perm;
      } else {
        posEmbBadge.textContent = "Fixed 2D Sin-Cos Pos Emb";
        posEmbBadge.style.color = T.muted;
        patchNote.textContent = "Coarse 4×4 tokens (49 on MNIST / 64 on CIFAR) weaken the local-patch advantage.";
        rhoBadge.textContent = "ρ ≈ −0.25 · Δ(100k) = +0.04";
        rhoBadge.style.color = T.muted;
      }

      drawDitSvg(nTok);
    }

    tSlider.addEventListener("input", () => {
      state.noiseT = Number(tSlider.value);
      redraw();
    });

    redraw();
    mountResponsive(container, redraw);
    observeAutoHide(container);
  }

  /* =========================================================================
   * FIGURE C: Real CIFAR-10 Training Images Showcase (#viz-cifar-train)
   * ========================================================================= */
  function mountCifarTrain(container, data) {
    const ctrain = data && data.cifar_train;
    const hero = data && data.hero;
    if (!ctrain || !hero) return;

    const perm1024 = hero.perm1024;
    const invPerm = invertPerm(perm1024);
    const items = ctrain.showcase.map((it) => {
      const bytes = decodeB64(it.b64);
      return {
        ...it,
        bytes,
        corrNat: computeLocalCorr(bytes, 3, 32, 32, null),
        corrPerm: computeLocalCorr(bytes, 3, 32, 32, invPerm),
      };
    });

    const pairs = (ctrain.nn_pairs || []).map((p) => ({
      ...p,
      natBytes: decodeB64(p.nat_b64),
      natNNBytes: decodeB64(p.nat_nn_b64),
      permBytes: decodeB64(p.perm_b64),
      permNNBytes: decodeB64(p.perm_nn_b64),
    }));

    const state = {
      view: "both", // "nat" | "both" | "perm"
    };

    const header = el("div", { class: "viz-header" }, [
      el("div", null, [
        el("h3", {
          class: "viz-title",
          text: "The Real CIFAR-10 Training Images (n = 128) — Natural vs. Permuted & Memorised Copies",
        }),
        el("p", {
          class: "viz-subtitle",
          text: "Sixteen actual training images from our n = 128 subset (subset_seed = 0) spanning all 10 CIFAR-10 classes, shown before and after the fixed pixel permutation (perm_seed = 1234), followed by 300k-step U-Net outputs paired with their exact nearest training image.",
        }),
      ]),
      el("span", { class: "viz-badge", text: "CIFAR-10 · 32×32 RGB · subset_seed = 0" }),
    ]);

    const viewSeg = makeSeg(
      [
        { value: "nat", label: "Natural (x₀)" },
        { value: "both", label: "Side-by-side (x₀ | Px₀)" },
        { value: "perm", label: "Permuted (Px₀)" },
      ],
      state.view,
      (v) => {
        state.view = v;
        redrawGrid();
      }
    );

    const clsChips = el("div", { class: "cifar-cls-chips" });
    for (const [cls, cnt] of Object.entries(ctrain.class_counts || {})) {
      clsChips.appendChild(el("span", { class: "cifar-cls-chip", text: `${cls}: ${cnt}` }));
    }

    const controls = el("div", { class: "viz-controls", style: "justify-content:space-between;" }, [
      el("div", { class: "viz-control-group" }, [
        el("span", { class: "viz-control-label", text: "Training view" }),
        viewSeg,
      ]),
      clsChips,
    ]);

    const gridWrap = el("div", { class: "cifar-train-grid" });

    // Bottom section: Generated vs Nearest Training Image pairs (n=128, 300k)
    const pairsGrid = el("div", { class: "cifar-nn-grid" });
    const pairsSection = el("div", { class: "cifar-nn-section" }, [
      el("div", { class: "viz-panel-title", style: "margin-bottom:8px;" }, [
        el("span", { text: "Generated Output vs. Exact Nearest Training Image (n = 128, 300k steps)" }),
        el("span", { class: "viz-readout", text: "Natural m = 1.000 · Permuted m = 0.933" }),
      ]),
      pairsGrid,
    ]);

    const figcaption = el("div", { class: "viz-figcaption" }, [
      el("b", { text: "Reading the colour images: " }),
      "Hover any training tile above to compare its local pixel correlation before vs. after permutation (",
      el("span", { style: "font-family:var(--mono)", text: "+0.8x → ≈ 0.00" }),
      "). Below, each generated sample from ",
      el("strong", { text: "Natural 300k" }),
      " and ",
      el("strong", { text: "Permuted 300k (unscrambled via P⁻¹)" }),
      " is placed next to its nearest L₂ neighbour in the 128-image training set — visually confirming that both arms condense onto real training photographs, with slightly higher residual grain on the permuted arm.",
    ]);

    clear(container);
    container.appendChild(header);
    container.appendChild(controls);
    container.appendChild(gridWrap);
    container.appendChild(pairsSection);
    container.appendChild(figcaption);

    function redrawGrid() {
      const T = K.theme();
      clear(gridWrap);
      gridWrap.className = `cifar-train-grid cifar-train-grid--${state.view}`;
      for (const it of items) {
        const card = el("div", { class: "cifar-train-card" });
        const cvsRow = el("div", { class: "cifar-train-canvases" });

        if (state.view === "nat" || state.view === "both") {
          const cNat = el("canvas", { class: "cifar-train-cvs" });
          cvsRow.appendChild(cNat);
          requestAnimationFrame(() => drawRaster(cNat, it.bytes, 3, 32, 32, { css: state.view === "both" ? 56 : 74 }));
        }
        if (state.view === "perm" || state.view === "both") {
          const cPerm = el("canvas", { class: "cifar-train-cvs" });
          cvsRow.appendChild(cPerm);
          requestAnimationFrame(() => drawRaster(cPerm, it.bytes, 3, 32, 32, { css: state.view === "both" ? 56 : 74, perm: perm1024 }));
        }

        card.appendChild(cvsRow);
        card.appendChild(el("div", { class: "cifar-train-meta" }, [
          el("strong", { text: it.label }),
          el("span", { text: `#${it.train_idx}` }),
        ]));

        bindBlockTip(card, `CIFAR-10 Train #${it.train_idx} · ${it.label}`, () => [
          ["Original CIFAR-10 ID", `#${it.cifar_id}`],
          ["Natural neighbour corr", `+${it.corrNat.toFixed(2)}`, T.nat],
          ["Permuted neighbour corr", `${it.corrPerm >= 0 ? "+" : ""}${it.corrPerm.toFixed(2)}`, T.perm],
          ["RGB grouping", "3 channels move together per pixel"],
        ]);
        gridWrap.appendChild(card);
      }
    }

    function drawPairs() {
      const T = K.theme();
      clear(pairsGrid);
      for (const p of pairs) {
        const card = el("div", { class: "cifar-nn-card" });

        const makeRow = (armLabel, armCol, genBytes, nnBytes, d1, ratio, nnIdx, nnLbl) => {
          const cGen = el("canvas", { class: "cifar-nn-cvs" });
          const cNN = el("canvas", { class: "cifar-nn-cvs" });
          requestAnimationFrame(() => {
            drawRaster(cGen, genBytes, 3, 32, 32, { css: 50 });
            drawRaster(cNN, nnBytes, 3, 32, 32, { css: 50 });
          });
          const row = el("div", { class: "cifar-nn-row" }, [
            el("div", { class: "cifar-nn-col" }, [
              cGen,
              el("span", { style: `color:${armCol};font-weight:600`, text: armLabel }),
            ]),
            el("div", { class: "cifar-nn-arrow", text: "≈" }),
            el("div", { class: "cifar-nn-col" }, [
              cNN,
              el("span", { text: `train #${nnIdx}` }),
            ]),
            el("div", { class: "cifar-nn-metrics" }, [
              el("div", { style: "font-weight:600;color:var(--text);", text: nnLbl }),
              el("div", { style: `color:${ratio < 1 / 3 ? T.gold : T.muted};font-family:var(--mono);font-size:10.5px;`, text: `r=${ratio.toFixed(3)}` }),
            ]),
          ]);
          bindBlockTip(row, `${armLabel} Sample #${p.seed_idx} vs. Train #${nnIdx} (${nnLbl})`, () => [
            ["Nearest L₂ dist d₁", d1.toFixed(2), armCol],
            ["Ratio d₁ / d₂", ratio.toFixed(3)],
            ["Status", ratio < 1 / 3 ? "Memorised (< 1/3)" : "Novel (≥ 1/3)", ratio < 1 / 3 ? T.gold : T.muted],
          ]);
          return row;
        };

        card.appendChild(makeRow("Nat 300k", T.nat, p.natBytes, p.natNNBytes, p.nat_d1, p.nat_ratio, p.nat_nn_idx, p.nat_nn_label));
        card.appendChild(makeRow("Perm 300k", T.perm, p.permBytes, p.permNNBytes, p.perm_d1, p.perm_ratio, p.perm_nn_idx, p.perm_nn_label));
        pairsGrid.appendChild(card);
      }
    }

    redrawGrid();
    drawPairs();
    mountResponsive(container, () => {
      redrawGrid();
      drawPairs();
    });
    observeAutoHide(container);
  }

  window.DiffMemWidgets = window.DiffMemWidgets || {};
  window.DiffMemWidgets.mountArchUnet = mountArchUnet;
  window.DiffMemWidgets.mountArchDit = mountArchDit;
  window.DiffMemWidgets.mountCifarTrain = mountCifarTrain;
})();
