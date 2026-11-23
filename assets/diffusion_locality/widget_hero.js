/* Interactive Hero Sandbox: Pixel Scrambling, 5x5 Receptive-Field Loupe, and Real d1/d2 Condensation */
(function () {
  "use strict";

  const K = window.DiffMemKit;
  if (!K) return;

  const COND_ORDER = ["mnist_n2048", "mnist_n1024", "cifar10_n128", "cifar10_n256"];
  const DATASET_CONDS = {
    mnist: [
      { value: "mnist_n2048", label: "n = 2048 (300k)" },
      { value: "mnist_n1024", label: "n = 1024 (100k)" },
    ],
    cifar10: [
      { value: "cifar10_n128", label: "n = 128 (300k)" },
      { value: "cifar10_n256", label: "n = 256 (300k)" },
    ],
  };
  const THRESHOLD = 1 / 3;

  function clamp(v, lo, hi) {
    return v < lo ? lo : v > hi ? hi : v;
  }

  function easeInOutCubic(t) {
    return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
  }

  function fmtSigned(v) {
    const s = v.toFixed(2);
    return v >= 0 && s !== "-0.00" ? `+${s}` : s === "-0.00" ? "+0.00" : s;
  }

  function fmtPct(v, digits = 1) {
    return `${(v * 100).toFixed(digits)}%`;
  }

  function decodeB64(b64) {
    const bin = atob(b64);
    const len = bin.length;
    const out = new Uint8Array(len);
    for (let i = 0; i < len; i++) out[i] = bin.charCodeAt(i);
    return out;
  }

  function invertPerm(perm) {
    const inv = new Int32Array(perm.length);
    for (let i = 0; i < perm.length; i++) inv[perm[i]] = i;
    return inv;
  }

  function pixelLum(bytes, idx, C) {
    if (C === 1) return bytes[idx];
    const o = idx * 3;
    return 0.299 * bytes[o] + 0.587 * bytes[o + 1] + 0.114 * bytes[o + 2];
  }

  function prepareSample(raw, C, H, W) {
    const bytes = decodeB64(raw.b64);
    const nPix = H * W;
    const lum = new Float32Array(nPix);
    let sum = 0;
    for (let i = 0; i < nPix; i++) {
      const v = pixelLum(bytes, i, C);
      lum[i] = v;
      sum += v;
    }
    const order = new Uint16Array(nPix);
    for (let i = 0; i < nPix; i++) order[i] = i;
    if (C === 1) {
      order.sort((a, b) => lum[a] - lum[b]);
    }
    return {
      idx: raw.idx,
      origId: raw.orig_id ?? null,
      label: raw.label || null,
      d1: raw.d1 ?? null,
      d2: raw.d2 ?? null,
      ratio: raw.ratio ?? null,
      memorized: !!raw.memorized,
      nnIdx: raw.nn_idx ?? null,
      nnLabel: raw.nn_label || null,
      nnB64: raw.nn_b64 || null,
      nnBytes: raw.nn_b64 ? decodeB64(raw.nn_b64) : null,
      bytes,
      lum,
      mu: sum / nPix,
      order,
    };
  }

  function computePatchCorr(sample, H, W, cx, cy, invPerm) {
    const x0 = clamp(cx - 2, 0, W - 5);
    const y0 = clamp(cy - 2, 0, H - 5);
    const mu = sample.mu;
    const lum = sample.lum;
    let num = 0;
    let vx = 0;
    let vy = 0;
    for (let y = y0; y < y0 + 5; y++) {
      for (let x = x0; x < x0 + 5; x++) {
        const p = y * W + x;
        const a = lum[invPerm ? invPerm[p] : p] - mu;
        if (x + 1 < x0 + 5) {
          const pr = y * W + (x + 1);
          const b = lum[invPerm ? invPerm[pr] : pr] - mu;
          num += a * b;
          vx += a * a;
          vy += b * b;
        }
        if (y + 1 < y0 + 5) {
          const pd = (y + 1) * W + x;
          const b = lum[invPerm ? invPerm[pd] : pd] - mu;
          num += a * b;
          vx += a * a;
          vy += b * b;
        }
      }
    }
    const den = Math.sqrt(vx * vy);
    return den > 1e-9 ? num / den : 0;
  }

  function computeL2Both(sampleA, sampleB, perm, C) {
    const a = sampleA.bytes;
    const b = sampleB.bytes;
    const nPix = perm.length;
    let sumNat = 0;
    for (let i = 0; i < a.length; i++) {
      const df = (a[i] - b[i]) / 255;
      sumNat += df * df;
    }
    // Reorder explicitly via perm to verify exact isometry in code
    let sumPerm = 0;
    if (C === 1) {
      const pa = new Float64Array(nPix);
      const pb = new Float64Array(nPix);
      for (let i = 0; i < nPix; i++) {
        const p = perm[i];
        pa[p] = a[i];
        pb[p] = b[i];
      }
      for (let p = 0; p < nPix; p++) {
        const df = (pa[p] - pb[p]) / 255;
        sumPerm += df * df;
      }
    } else {
      const pa = new Float64Array(nPix * 3);
      const pb = new Float64Array(nPix * 3);
      for (let i = 0; i < nPix; i++) {
        const dst = perm[i] * 3;
        const src = i * 3;
        pa[dst] = a[src];
        pa[dst + 1] = a[src + 1];
        pa[dst + 2] = a[src + 2];
        pb[dst] = b[src];
        pb[dst + 1] = b[src + 1];
        pb[dst + 2] = b[src + 2];
      }
      for (let p = 0; p < pa.length; p++) {
        const df = (pa[p] - pb[p]) / 255;
        sumPerm += df * df;
      }
    }
    const dNat = Math.sqrt(sumNat);
    const dPerm = Math.sqrt(sumPerm);
    return { dNat, dPerm, delta: Math.abs(dNat - dPerm) };
  }

  function sizeHiDPICanvas(canvas, fallbackCss) {
    const rect = canvas.getBoundingClientRect();
    const css = rect.width > 12 ? rect.width : fallbackCss;
    const dpr = Math.min(window.devicePixelRatio || 1, 3);
    const px = Math.max(1, Math.round(css * dpr));
    if (canvas.width !== px || canvas.height !== px) {
      canvas.width = px;
      canvas.height = px;
    }
    return px;
  }

  function hexOrCssToRgba(color, alpha) {
    const c = color.trim();
    if (c.startsWith("#")) {
      let hex = c.slice(1);
      if (hex.length === 3) {
        hex = hex[0] + hex[0] + hex[1] + hex[1] + hex[2] + hex[2];
      }
      const n = parseInt(hex, 16);
      if (!Number.isNaN(n)) {
        return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${alpha})`;
      }
    }
    const m = c.match(/rgba?\(\s*([\d.]+)[,\s]+([\d.]+)[,\s]+([\d.]+)/);
    if (m) return `rgba(${m[1]}, ${m[2]}, ${m[3]}, ${alpha})`;
    return color;
  }

  function renderGridToCanvas(canvas, sample, C, H, W, perm, offscreen, fallbackCss) {
    const px = sizeHiDPICanvas(canvas, fallbackCss);
    if (offscreen.width !== W || offscreen.height !== H) {
      offscreen.width = W;
      offscreen.height = H;
    }
    const octx = offscreen.getContext("2d");
    const img = octx.createImageData(W, H);
    const out = img.data;
    const bytes = sample.bytes;
    const nPix = H * W;
    if (C === 1) {
      for (let i = 0; i < nPix; i++) {
        const dst = (perm ? perm[i] : i) * 4;
        const v = bytes[i];
        out[dst] = v;
        out[dst + 1] = v;
        out[dst + 2] = v;
        out[dst + 3] = 255;
      }
    } else {
      for (let i = 0; i < nPix; i++) {
        const src = i * 3;
        const dst = (perm ? perm[i] : i) * 4;
        out[dst] = bytes[src];
        out[dst + 1] = bytes[src + 1];
        out[dst + 2] = bytes[src + 2];
        out[dst + 3] = 255;
      }
    }
    octx.putImageData(img, 0, 0);
    const ctx = canvas.getContext("2d");
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, px, px);
    ctx.drawImage(offscreen, 0, 0, px, px);
    return px;
  }

  function renderMorphToCanvas(canvas, sample, C, H, W, perm, t, T, offscreen, fallbackCss) {
    if (t <= 0.0005) {
      return renderGridToCanvas(canvas, sample, C, H, W, null, offscreen, fallbackCss);
    }
    if (t >= 0.9995) {
      return renderGridToCanvas(canvas, sample, C, H, W, perm, offscreen, fallbackCss);
    }
    const px = sizeHiDPICanvas(canvas, fallbackCss);
    const ctx = canvas.getContext("2d");
    ctx.imageSmoothingEnabled = false;
    ctx.fillStyle = C === 1 ? "#07090c" : T.cardAlt;
    ctx.fillRect(0, 0, px, px);

    const cell = px / W;
    const cw = Math.max(1, Math.ceil(cell));
    const bytes = sample.bytes;
    const order = sample.order;
    const nPix = H * W;

    for (let k = 0; k < nPix; k++) {
      const i = order[k];
      const nx = i % W;
      const ny = (i - nx) / W;
      const p = perm[i];
      const gx = p % W;
      const gy = (p - gx) / W;
      const ix = nx + (gx - nx) * t;
      const iy = ny + (gy - ny) * t;
      if (C === 1) {
        const v = bytes[i];
        ctx.fillStyle = `rgb(${v},${v},${v})`;
      } else {
        const o = i * 3;
        ctx.fillStyle = `rgb(${bytes[o]},${bytes[o + 1]},${bytes[o + 2]})`;
      }
      ctx.fillRect(ix * cell, iy * cell, cw, cw);
    }
    return px;
  }

  function drawPatchBox(ctx, px, H, W, x0, y0, hx, hy, color, T) {
    const cell = px / W;
    const bx = x0 * cell;
    const by = y0 * cell;
    const bw = 5 * cell;

    ctx.save();
    ctx.fillStyle = hexOrCssToRgba(color, 0.14);
    ctx.fillRect(bx, by, bw, bw);

    // Contrast halo underneath so box reads over both dark and bright pixels
    ctx.strokeStyle = "rgba(0, 0, 0, 0.65)";
    ctx.lineWidth = Math.max(3, cell * 0.42);
    ctx.strokeRect(bx, by, bw, bw);

    ctx.strokeStyle = color;
    ctx.lineWidth = Math.max(1.6, cell * 0.24);
    ctx.strokeRect(bx, by, bw, bw);

    // Subtle reticle on exact hovered pixel
    ctx.strokeStyle = T.surface;
    ctx.lineWidth = 1;
    ctx.strokeRect(hx * cell + 0.5, hy * cell + 0.5, Math.max(1, cell - 1), Math.max(1, cell - 1));
    ctx.restore();
  }

  function drawScatteredCells(ctx, px, W, cells, color) {
    const cell = px / W;
    const fill = hexOrCssToRgba(color, 0.30);
    const halo = "rgba(0, 0, 0, 0.65)";
    ctx.save();
    for (const c of cells) {
      const rx = c.x * cell;
      const ry = c.y * cell;
      ctx.fillStyle = fill;
      ctx.fillRect(rx, ry, cell, cell);
      ctx.strokeStyle = halo;
      ctx.lineWidth = Math.max(2, cell * 0.28);
      ctx.strokeRect(rx, ry, cell, cell);
      ctx.strokeStyle = color;
      ctx.lineWidth = c.isCenter ? Math.max(1.8, cell * 0.25) : Math.max(1.2, cell * 0.16);
      ctx.strokeRect(rx, ry, cell, cell);
    }
    ctx.restore();
  }

  function niceBinCap(peak) {
    const raw = Math.max(0.10, peak * 1.16);
    const step = raw <= 0.16 ? 0.02 : raw <= 0.40 ? 0.05 : 0.10;
    return Math.ceil(raw / step) * step;
  }

  function sampleDensityAt(edges, bins, x) {
    if (x <= edges[0]) return bins[0];
    if (x >= edges[edges.length - 1]) return bins[bins.length - 1];
    const w = edges[1] - edges[0];
    const idx = clamp(Math.floor((x - edges[0]) / w), 0, bins.length - 1);
    return bins[idx];
  }

  function mountHero(container, data) {
    const hero = data && data.hero;
    if (!hero || !hero.conditions) {
      container.appendChild(K.el("div", { class: "viz-loading", text: "Hero dataset unavailable." }));
      return;
    }

    const permCache = {
      784: { fwd: hero.perm784, inv: invertPerm(hero.perm784) },
      1024: { fwd: hero.perm1024, inv: invertPerm(hero.perm1024) },
    };

    const condCache = {};
    for (const key of COND_ORDER) {
      const c = hero.conditions[key];
      if (!c) continue;
      const [C, H, W] = c.shape;
      condCache[key] = {
        raw: c,
        dataset: c.dataset || (C === 3 ? "cifar10" : "mnist"),
        C,
        H,
        W,
        permObj: permCache[H * W],
        nat: c.nat_samples.map((s) => prepareSample(s, C, H, W)),
        perm: c.perm_samples.map((s) => prepareSample(s, C, H, W)),
        train: (c.train_samples || []).map((s) => prepareSample(s, C, H, W)),
      };
    }

    const prefersReduced =
      typeof window.matchMedia === "function" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    const initKey = "mnist_n2048";
    const initCond = condCache[initKey].raw;
    const initStepIdx = Math.max(0, initCond.steps.indexOf(initCond.default_step));

    const state = {
      dataset: "mnist",
      condKey: initKey,
      stepIdx: initStepIdx,
      scramble: 1.0,
      sampleArm: "nat", // "nat" | "perm" | "train"
      sampleIdx: 0,
      customSample: null, // optional 1-NN training image override
      galleryView: "nat",
      hx: Math.floor(condCache[initKey].W / 2),
      hy: Math.floor(condCache[initKey].H / 2),
      hoverArm: "nat",
      hoveredBin: null,
      playTimer: null,
      animRaf: null,
    };

    const offscreen = document.createElement("canvas");

    // ---------- Build static DOM scaffold ONCE ----------
    const headerBadge = K.el("span", { class: "viz-badge" });
    const header = K.el("div", { class: "viz-header" }, [
      K.el("div", null, [
        K.el("h3", {
          class: "viz-title",
          text: "Interactive Sandbox: Scrambling Pixels Delays Condensation, Not Its Destination",
        }),
        K.el("p", {
          class: "viz-subtitle",
          text: "Switch between MNIST and CIFAR-10, hover to trace a 5×5 convolutional patch across coordinate systems, inspect real training vs. generated images, and scrub training steps to watch d₁/d₂ condense.",
        }),
      ]),
      headerBadge,
    ]);

    function switchCondition(val) {
      stopPlay();
      stopScrambleAnim();
      K.tooltip.hide();
      state.condKey = val;
      const cc = condCache[val];
      state.dataset = cc.dataset;
      const idx = cc.raw.steps.indexOf(cc.raw.default_step);
      state.stepIdx = idx >= 0 ? idx : 0;
      state.sampleArm = "nat";
      state.sampleIdx = 0;
      state.customSample = null;
      state.hx = Math.floor(cc.W / 2);
      state.hy = Math.floor(cc.H / 2);
      state.hoverArm = "nat";
      state.hoveredBin = null;
      syncSubCondSeg();
      syncStepSlider();
      redrawAll();
    }

    // Prominent Top-Level Dataset Tabs: MNIST | CIFAR-10
    const dsTabBtns = {};
    const datasetTabs = K.el("div", { class: "hero-dataset-tabs", role: "tablist", "aria-label": "Dataset" }, [
      ["mnist", "MNIST (28×28)"],
      ["cifar10", "CIFAR-10 (32×32 RGB)"],
    ].map(([dsKey, dsLabel]) => {
      const btn = K.el("button", {
        type: "button",
        role: "tab",
        "aria-pressed": dsKey === state.dataset ? "true" : "false",
        text: dsLabel,
        onClick: () => {
          if (state.dataset === dsKey) return;
          const opts = (DATASET_CONDS[dsKey] || []).filter((o) => condCache[o.value]);
          if (opts.length) switchCondition(opts[0].value);
        },
      });
      dsTabBtns[dsKey] = btn;
      return btn;
    }));

    const subCondWrap = K.el("div", { class: "viz-control-group" });
    function syncSubCondSeg() {
      for (const [k, b] of Object.entries(dsTabBtns)) {
        b.setAttribute("aria-pressed", k === state.dataset ? "true" : "false");
      }
      K.clear(subCondWrap);
      subCondWrap.appendChild(K.el("span", { class: "viz-control-label", text: "Subset" }));
      const opts = (DATASET_CONDS[state.dataset] || []).filter((o) => condCache[o.value]);
      subCondWrap.appendChild(K.makeSeg(opts, state.condKey, (val) => switchCondition(val)));
    }
    syncSubCondSeg();

    const scrambleSlider = K.el("input", {
      type: "range",
      class: "viz-slider",
      min: 0,
      max: 100,
      step: 1,
      value: 100,
      "aria-label": "Pixel scramble percentage",
    });
    const scrambleReadout = K.el("span", { class: "viz-readout", text: "100%" });
    const animScrambleBtn = K.el("button", {
      type: "button",
      class: "viz-btn",
      text: "Animate Scramble",
    });
    const resetBtn = K.el("button", {
      type: "button",
      class: "viz-btn",
      text: "Reset",
    });

    const stepSlider = K.el("input", {
      type: "range",
      class: "viz-slider",
      min: 0,
      max: initCond.steps.length - 1,
      step: 1,
      value: state.stepIdx,
      "aria-label": "Training step",
    });
    const stepReadout = K.el("span", { class: "viz-readout" });
    const playBtn = K.el("button", {
      type: "button",
      class: "viz-btn viz-btn--primary",
    });

    const topBar = K.el("div", {
      style: "display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:10px 14px;margin-bottom:10px;",
    }, [datasetTabs, subCondWrap]);

    const controls = K.el("div", { class: "viz-controls hero-sliders-row" }, [
      K.el("div", { class: "viz-slider-wrap", style: "flex:1 1 360px;" }, [
        K.el("span", { class: "viz-control-label", text: "Scramble" }),
        scrambleSlider,
        scrambleReadout,
        animScrambleBtn,
        resetBtn,
      ]),
      K.el("div", { class: "viz-slider-wrap", style: "flex:1 1 360px;" }, [
        K.el("span", { class: "viz-control-label", text: "Step" }),
        stepSlider,
        stepReadout,
        playBtn,
      ]),
    ]);

    // Left panel: Dual canvases + 5x5 loupe + 3 stats + 1-NN training match strip
    const leftTitleTag = K.el("span", { class: "hero-canvas-sub" });
    const natCanvas = K.el("canvas", { "aria-label": "Sample in natural pixel coordinates" });
    const permCanvas = K.el("canvas", { "aria-label": "Sample in morphed or permuted pixel coordinates" });
    const natBoxSub = K.el("div", { class: "hero-canvas-sub" });
    const permBoxLabel = K.el("div", { class: "hero-canvas-label" });
    const permBoxSub = K.el("div", { class: "hero-canvas-sub" });

    const statCorrVal = K.el("div", { class: "viz-stat__v" });
    const statL2Val = K.el("div", { class: "viz-stat__v" });
    const statL2Key = K.el("div", { class: "viz-stat__k", text: "L2 dist ||x − y||₂" });
    const statRatioKey = K.el("div", { class: "viz-stat__k", text: "Sample d₁/d₂" });
    const statRatioVal = K.el("div", { class: "viz-stat__v" });

    const nnCanvas = K.el("canvas", { "aria-label": "Nearest training image" });
    const nnMeta = K.el("div", { class: "hero-nn-strip__meta" });
    const nnInspectBtn = K.el("button", {
      type: "button",
      class: "viz-btn",
      style: "font-size:11.5px;padding:4px 9px;white-space:nowrap;",
      text: "Inspect 1-NN",
      onClick: () => {
        const { cc, sample } = activeSample();
        if (!sample || !sample.nnB64) return;
        state.customSample = prepareSample(
          { idx: sample.nnIdx, label: sample.nnLabel, b64: sample.nnB64 },
          cc.C,
          cc.H,
          cc.W
        );
        updateGallerySelection();
        redrawLeftSide();
      },
    });
    const nnStrip = K.el("div", { class: "hero-nn-strip" }, [nnCanvas, nnMeta, nnInspectBtn]);

    const leftPanel = K.el("div", { class: "viz-panel" }, [
      K.el("div", { class: "viz-panel-title" }, [
        K.el("span", { text: "Pixel Scrambling & 5×5 Receptive-Field Loupe" }),
        leftTitleTag,
      ]),
      K.el("div", { class: "hero-canvas-pair" }, [
        K.el("div", { class: "hero-canvas-box" }, [
          natCanvas,
          K.el("div", { class: "hero-canvas-label", text: "Natural coords" }),
          natBoxSub,
        ]),
        K.el("div", { class: "hero-canvas-box" }, [
          permCanvas,
          permBoxLabel,
          permBoxSub,
        ]),
      ]),
      K.el("div", { class: "viz-stat-strip" }, [
        K.el("div", { class: "viz-stat" }, [
          K.el("div", { class: "viz-stat__k", text: "Patch corr (5×5)" }),
          statCorrVal,
        ]),
        K.el("div", { class: "viz-stat" }, [statL2Key, statL2Val]),
        K.el("div", { class: "viz-stat" }, [statRatioKey, statRatioVal]),
      ]),
      nnStrip,
    ]);

    // Right panel: SVG density plot + legend + 4 stats
    const plotContainer = K.el("div");
    const legNatSwatch = K.el("span", { class: "viz-legend__swatch" });
    const legPermSwatch = K.el("span", { class: "viz-legend__swatch" });
    const legMemSwatch = K.el("span", { class: "viz-legend__swatch" });
    const rightStatNat = K.el("div", { class: "viz-stat__v" });
    const rightStatPerm = K.el("div", { class: "viz-stat__v" });
    const rightStatMed = K.el("div", { class: "viz-stat__v" });
    const rightStatBudget = K.el("div", { class: "viz-stat__v" });

    const rightPanel = K.el("div", { class: "viz-panel" }, [
      K.el("div", { class: "viz-panel-title" }, [
        K.el("span", { text: "Real d₁/d₂ Distribution Condensation (1,000 samples)" }),
      ]),
      plotContainer,
      K.el("div", { class: "viz-legend" }, [
        K.el("span", { class: "viz-legend__item" }, [legNatSwatch, "Natural U-Net"]),
        K.el("span", { class: "viz-legend__item" }, [legPermSwatch, "Permuted U-Net"]),
        K.el("span", { class: "viz-legend__item" }, [legMemSwatch, "Memorised (d₁/d₂ < 1/3)"]),
      ]),
      K.el("div", {
        class: "viz-stat-strip",
        style: "grid-template-columns:repeat(auto-fit,minmax(105px,1fr))",
      }, [
        K.el("div", { class: "viz-stat" }, [
          K.el("div", { class: "viz-stat__k", text: "Natural m(1/3)" }),
          rightStatNat,
        ]),
        K.el("div", { class: "viz-stat" }, [
          K.el("div", { class: "viz-stat__k", text: "Permuted m(1/3)" }),
          rightStatPerm,
        ]),
        K.el("div", { class: "viz-stat" }, [
          K.el("div", { class: "viz-stat__k", text: "Median d₁/d₂" }),
          rightStatMed,
        ]),
        K.el("div", { class: "viz-stat" }, [
          K.el("div", { class: "viz-stat__k", text: "Budget" }),
          rightStatBudget,
        ]),
      ]),
    ]);

    const mainGrid = K.el("div", { class: "viz-grid-hero" }, [leftPanel, rightPanel]);

    // Bottom gallery
    const galleryTitle = K.el("span", { class: "viz-control-label" });
    const galleryViewSeg = K.makeSeg(
      [
        { value: "nat", label: "Natural pixels" },
        { value: "perm", label: "Permuted pixels" },
      ],
      state.galleryView,
      (val) => {
        state.galleryView = val;
        redrawGallery();
      }
    );
    const natArmLabel = K.el("div", { class: "hero-gallery-arm", text: "Natural U-Net" });
    const permArmLabel = K.el("div", { class: "hero-gallery-arm", text: "Permuted U-Net" });
    const trainArmLabel = K.el("div", { class: "hero-gallery-arm", text: "Training Set" });
    const natGalleryRow = K.el("div", { class: "hero-gallery-row" }, [natArmLabel]);
    const permGalleryRow = K.el("div", { class: "hero-gallery-row" }, [permArmLabel]);
    const trainGalleryRow = K.el("div", { class: "hero-gallery-row" }, [trainArmLabel]);

    const tileRegistry = { nat: [], perm: [], train: [] };
    for (const arm of ["nat", "perm", "train"]) {
      const row = arm === "nat" ? natGalleryRow : arm === "perm" ? permGalleryRow : trainGalleryRow;
      for (let i = 0; i < 8; i++) {
        const cvs = K.el("canvas");
        const dot = K.el("span", { class: "hero-sample-dot" });
        const ratioEl = K.el("div", {
          class: "hero-canvas-sub",
          style: "font-size:10px;text-align:center;margin-top:3px;line-height:1.1;",
        });
        const btn = K.el("button", {
          type: "button",
          class: "hero-sample-tile",
          onClick: () => {
            K.tooltip.hide();
            state.sampleArm = arm;
            state.sampleIdx = i;
            state.customSample = null;
            updateGallerySelection();
            redrawLeftSide();
          },
        }, [cvs, dot, ratioEl]);

        const showTileTip = (e) => {
          const cc = condCache[state.condKey];
          const s = cc[arm][i];
          if (!s) return;
          const T = K.theme();
          let html;
          if (arm === "train") {
            html =
              `<div class="viz-tooltip__title">Training Image · #${s.idx} (${s.label})</div>` +
              K.tipRow("Dataset ID", `#${s.origId ?? s.idx}`) +
              K.tipRow("Role", "Exact image in training subset", T.nat);
          } else {
            const armName = arm === "nat" ? "Natural U-Net" : "Permuted U-Net";
            const armCol = arm === "nat" ? T.nat : T.perm;
            html =
              `<div class="viz-tooltip__title">${armName} · Sample #${s.idx}</div>` +
              K.tipRow("d₁ (1-NN)", s.d1.toFixed(2), armCol) +
              K.tipRow("d₂ (2-NN)", s.d2.toFixed(2)) +
              K.tipRow("Ratio d₁/d₂", s.ratio.toFixed(3)) +
              (s.nnLabel ? K.tipRow("Nearest train", `#${s.nnIdx} (${s.nnLabel})`) : "") +
              K.tipRow("Status", s.memorized ? "Memorised (< 1/3)" : "Novel (≥ 1/3)", s.memorized ? T.gold : T.muted);
          }
          K.tooltip.show(e.clientX, e.clientY, html, btn);
        };
        btn.addEventListener("mouseenter", showTileTip);
        btn.addEventListener("mousemove", showTileTip);
        btn.addEventListener("mouseleave", () => K.tooltip.hide());
        btn.addEventListener("blur", () => K.tooltip.hide());

        row.appendChild(btn);
        tileRegistry[arm].push({ btn, cvs, dot, ratioEl });
      }
    }

    const gallery = K.el("div", { class: "hero-gallery" }, [
      K.el("div", {
        style: "display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:8px;margin-bottom:6px;",
      }, [
        galleryTitle,
        K.el("div", { class: "viz-control-group" }, [
          K.el("span", { class: "viz-control-label", text: "View tiles in" }),
          galleryViewSeg,
        ]),
      ]),
      natGalleryRow,
      permGalleryRow,
      trainGalleryRow,
    ]);

    const figcaption = K.el("div", { class: "viz-figcaption" }, [
      K.el("b", { text: "Why scrambling only buys time: " }),
      "Permuting pixel indices is an exact L2 isometry (pairwise distances ",
      K.el("span", { style: "font-family:var(--mono)", text: "||x − y||₂" }),
      " are identical to ",
      K.el("span", { style: "font-family:var(--mono)", text: "Δ = 0.000" }),
      "), yet it destroys local 5×5 spatial correlation (",
      K.el("span", { style: "font-family:var(--mono)", text: "+0.8x → ≈ 0.00" }),
      "). Deprived of convolutional inductive bias, the Permuted U-Net takes more steps to condense into the ",
      K.el("span", { style: "font-family:var(--mono)", text: "d₁/d₂ < 1/3" }),
      " memorised zone—but converges to nearly the same destination.",
    ]);

    container.appendChild(header);
    container.appendChild(topBar);
    container.appendChild(controls);
    container.appendChild(mainGrid);
    container.appendChild(gallery);
    container.appendChild(figcaption);

    // ---------- Helpers & Interactive Event Wiring ----------
    function activeSample() {
      const cc = condCache[state.condKey];
      if (state.customSample) {
        return {
          cc,
          sample: state.customSample,
          partner: cc.train[0] || cc.nat[0],
          isTrain: true,
        };
      }
      const list = cc[state.sampleArm] && cc[state.sampleArm].length ? cc[state.sampleArm] : cc.nat;
      const idx = clamp(state.sampleIdx, 0, list.length - 1);
      return {
        cc,
        sample: list[idx],
        partner: list[(idx + 1) % list.length],
        isTrain: state.sampleArm === "train",
      };
    }

    function stopPlay() {
      if (state.playTimer !== null) {
        clearInterval(state.playTimer);
        state.playTimer = null;
      }
      syncPlayButton();
    }

    function stopScrambleAnim() {
      if (state.animRaf !== null) {
        cancelAnimationFrame(state.animRaf);
        state.animRaf = null;
        animScrambleBtn.textContent = "Animate Scramble";
      }
    }

    function syncStepSlider() {
      const steps = condCache[state.condKey].raw.steps;
      stepSlider.min = 0;
      stepSlider.max = steps.length - 1;
      stepSlider.value = state.stepIdx;
      syncPlayButton();
    }

    function syncPlayButton() {
      const steps = condCache[state.condKey].raw.steps;
      if (state.playTimer !== null) {
        playBtn.textContent = "Pause";
      } else {
        playBtn.textContent = `Play ${K.fmtSteps(steps[0])} → ${K.fmtSteps(steps[steps.length - 1])}`;
      }
    }

    function updateHeaderAndDensityStats() {
      const T = K.theme();
      const cond = condCache[state.condKey].raw;
      const step = cond.steps[state.stepIdx];
      const nh = cond.nat_hist[String(step)];
      const ph = cond.perm_hist[String(step)];
      const natPct = Math.round(nh.m13 * 100);
      const permPct = Math.round(ph.m13 * 100);

      stepReadout.textContent = `Step ${K.fmtSteps(step)}`;

      let badgeText;
      let badgeCls = "viz-badge";
      if (natPct < 5 && permPct < 5) {
        badgeText = `Step ${K.fmtSteps(step)}: Both arms novel (${natPct}% / ${permPct}%)`;
      } else if (Math.abs(natPct - permPct) >= 10) {
        badgeText = `Step ${K.fmtSteps(step)}: Natural ${natPct}% vs Permuted ${permPct}%`;
        badgeCls = "viz-badge viz-badge--nat";
      } else if (Math.abs(natPct - permPct) <= 4 && (natPct + permPct) >= 100) {
        badgeText = `Step ${K.fmtSteps(step)}: Both arms ~${Math.round((natPct + permPct) / 2)}% condensed`;
        badgeCls = "viz-badge viz-badge--warn";
      } else {
        badgeText = `Step ${K.fmtSteps(step)}: Natural ${natPct}% vs Permuted ${permPct}%`;
      }
      headerBadge.className = badgeCls;
      headerBadge.textContent = badgeText;

      rightStatNat.style.color = T.nat;
      rightStatNat.textContent = fmtPct(nh.m13, 1);
      rightStatPerm.style.color = T.perm;
      rightStatPerm.textContent = fmtPct(ph.m13, 1);
      rightStatMed.textContent = `${nh.med.toFixed(2)} / ${ph.med.toFixed(2)}`;
      rightStatBudget.textContent = `${K.fmtSteps(step)} / ${K.fmtSteps(cond.steps[cond.steps.length - 1])}`;

      legNatSwatch.style.background = T.nat;
      legPermSwatch.style.background = "transparent";
      legPermSwatch.style.borderTop = `3px dashed ${T.perm}`;
      legPermSwatch.style.height = "0";
      legMemSwatch.style.background = hexOrCssToRgba(T.gold, 0.35);
      legMemSwatch.style.height = "10px";
      legMemSwatch.style.border = `1px dashed ${T.gold}`;
    }

    function redrawCanvasesAndLeftStats() {
      const T = K.theme();
      const { cc, sample, partner, isTrain } = activeSample();
      const { C, H, W, permObj } = cc;
      const perm = permObj.fwd;
      const invPerm = permObj.inv;
      const t = easeInOutCubic(state.scramble);
      const fullyPermuted = state.scramble >= 0.9995;
      const modePerm = state.hoverArm === "perm" && fullyPermuted;

      const hx = clamp(state.hx, 0, W - 1);
      const hy = clamp(state.hy, 0, H - 1);
      const x0 = clamp(hx - 2, 0, W - 5);
      const y0 = clamp(hy - 2, 0, H - 5);

      const pxNat = renderGridToCanvas(natCanvas, sample, C, H, W, null, offscreen, 196);
      const pxPerm = renderMorphToCanvas(permCanvas, sample, C, H, W, perm, t, T, offscreen, 196);

      const natCtx = natCanvas.getContext("2d");
      const permCtx = permCanvas.getContext("2d");

      let rNat;
      let rPerm;

      if (modePerm) {
        // Hovering Permuted canvas at 100%: 5x5 box in Permuted space -> 25 scattered sources in Natural
        drawPatchBox(permCtx, pxPerm, H, W, x0, y0, hx, hy, T.perm, T);
        const natCells = [];
        for (let py = y0; py < y0 + 5; py++) {
          for (let pxCell = x0; pxCell < x0 + 5; pxCell++) {
            const idx = invPerm[py * W + pxCell];
            natCells.push({
              x: idx % W,
              y: Math.floor(idx / W),
              isCenter: pxCell === hx && py === hy,
            });
          }
        }
        drawScatteredCells(natCtx, pxNat, W, natCells, T.nat);

        const centerIdx = invPerm[hy * W + hx];
        const ncx = centerIdx % W;
        const ncy = Math.floor(centerIdx / W);
        rNat = computePatchCorr(sample, H, W, ncx, ncy, null);
        rPerm = computePatchCorr(sample, H, W, hx, hy, invPerm);

        natBoxSub.textContent = "25 scattered sources";
        permBoxLabel.textContent = "Permuted coords";
        permBoxSub.textContent = `5×5 conv @ (${hx}, ${hy})`;
      } else {
        // Default / hovering Natural canvas (or morphing): 5x5 box in Natural -> 25 morphing pixels in Right canvas
        drawPatchBox(natCtx, pxNat, H, W, x0, y0, hx, hy, T.nat, T);
        const morphCells = [];
        for (let ny = y0; ny < y0 + 5; ny++) {
          for (let nx = x0; nx < x0 + 5; nx++) {
            const idx = ny * W + nx;
            const p = perm[idx];
            const gx = p % W;
            const gy = Math.floor(p / W);
            morphCells.push({
              x: nx + (gx - nx) * t,
              y: ny + (gy - ny) * t,
              isCenter: nx === hx && ny === hy,
            });
          }
        }
        drawScatteredCells(permCtx, pxPerm, W, morphCells, T.perm);

        rNat = computePatchCorr(sample, H, W, hx, hy, null);
        rPerm = computePatchCorr(sample, H, W, hx, hy, invPerm);

        natBoxSub.textContent = `5×5 patch @ (${hx}, ${hy})`;
        const pct = Math.round(state.scramble * 100);
        permBoxLabel.textContent =
          pct === 0 ? "Unscrambled (0%)" : pct === 100 ? "Permuted coords" : `Morphing (${pct}%)`;
        permBoxSub.textContent =
          pct === 0 ? "Identical grid" : pct === 100 ? "25 shattered pixels" : "Pixels in flight";
      }

      leftTitleTag.textContent = isTrain
        ? `Train #${sample.idx}${sample.label ? ` (${sample.label})` : ""}`
        : `${state.sampleArm === "nat" ? "Nat" : "Perm"} #${sample.idx} · ${K.fmtSteps(cc.raw.samples_step)}`;

      // Stat 1: Patch corr (5x5)
      K.clear(statCorrVal);
      statCorrVal.appendChild(K.el("span", { style: `color:${T.nat}`, text: fmtSigned(rNat) }));
      statCorrVal.appendChild(K.el("span", { style: `color:${T.muted};font-weight:500`, text: " vs " }));
      statCorrVal.appendChild(K.el("span", { style: `color:${T.perm}`, text: fmtSigned(rPerm) }));

      // Stat 2: L2 distance isometry
      const l2 = computeL2Both(sample, partner, perm, C);
      statL2Key.textContent = `L2 ||x − y||₂ (vs #${partner.idx})`;
      K.clear(statL2Val);
      statL2Val.appendChild(K.el("span", { text: `${l2.dNat.toFixed(2)} ` }));
      statL2Val.appendChild(
        K.el("span", {
          style: `color:${T.muted};font-size:11.5px;font-weight:500`,
          text: `(Δ=${l2.delta.toFixed(3)})`,
        })
      );

      // Stat 3: Sample d1/d2 + status badge OR Training image label
      K.clear(statRatioVal);
      if (isTrain || sample.ratio === null) {
        statRatioKey.textContent = "Training image";
        statRatioVal.appendChild(K.el("span", { style: `color:${T.nat}`, text: sample.label || `#${sample.idx}` }));
        nnStrip.style.display = "none";
      } else {
        statRatioKey.textContent = "Sample d₁/d₂";
        statRatioVal.appendChild(K.el("span", { text: `${sample.ratio.toFixed(3)} ` }));
        const memBadge = K.el("span", {
          class: "viz-badge",
          style: sample.memorized
            ? `padding:1px 6px;font-size:10.5px;color:${T.gold};border-color:${hexOrCssToRgba(T.gold, 0.5)};background:${hexOrCssToRgba(T.gold, 0.12)}`
            : `padding:1px 6px;font-size:10.5px;color:${T.muted};`,
          text: sample.memorized ? "Mem." : "Novel",
        });
        statRatioVal.appendChild(memBadge);

        if (sample.nnBytes) {
          nnStrip.style.display = "flex";
          renderGridToCanvas(nnCanvas, { bytes: sample.nnBytes }, C, H, W, null, offscreen, 40);
          K.clear(nnMeta);
          nnMeta.appendChild(K.el("div", {
            style: "font-weight:600;color:var(--text);",
            text: `Nearest training image: #${sample.nnIdx} (${sample.nnLabel})`,
          }));
          nnMeta.appendChild(K.el("div", {
            style: "color:var(--muted);font-family:var(--mono);font-size:11px;",
            text: `d₁ = ${sample.d1.toFixed(2)} · d₂ = ${sample.d2.toFixed(2)} · ${sample.memorized ? "memorised copy" : "novel generation"}`,
          }));
        } else {
          nnStrip.style.display = "none";
        }
      }
    }

    function redrawDensityChart() {
      const T = K.theme();
      const cond = condCache[state.condKey].raw;
      const step = cond.steps[state.stepIdx];
      const nh = cond.nat_hist[String(step)];
      const ph = cond.perm_hist[String(step)];
      const edges = hero.hist_edges;

      const w = Math.max(280, Math.round(plotContainer.clientWidth || 470));
      const h = 265;
      const m = { t: 24, r: 16, b: 38, l: 44 };
      const pw = w - m.l - m.r;
      const phPlot = h - m.t - m.b;

      let peak = 0;
      for (let i = 0; i < nh.bins.length; i++) {
        if (nh.bins[i] > peak) peak = nh.bins[i];
        if (ph.bins[i] > peak) peak = ph.bins[i];
      }
      const yMax = niceBinCap(peak);
      const xScale = K.scaleLinear([0, 1.08], [m.l, m.l + pw]);
      const yScale = K.scaleLinear([0, yMax], [m.t + phPlot, m.t]);

      K.clear(plotContainer);
      const svg = K.makeSvg(w, h);

      // Gradients for smooth density area fills
      const uid = `hero_${Math.random().toString(36).slice(2, 8)}`;
      const defs = K.svgEl("defs", null, [
        K.svgEl("linearGradient", { id: `${uid}_nat`, x1: 0, y1: 0, x2: 0, y2: 1 }, [
          K.svgEl("stop", { offset: "0%", "stop-color": T.nat, "stop-opacity": 0.28 }),
          K.svgEl("stop", { offset: "100%", "stop-color": T.nat, "stop-opacity": 0.03 }),
        ]),
        K.svgEl("linearGradient", { id: `${uid}_perm`, x1: 0, y1: 0, x2: 0, y2: 1 }, [
          K.svgEl("stop", { offset: "0%", "stop-color": T.perm, "stop-opacity": 0.24 }),
          K.svgEl("stop", { offset: "100%", "stop-color": T.perm, "stop-opacity": 0.02 }),
        ]),
      ]);
      svg.appendChild(defs);

      // Shaded memorisation zone d1/d2 < 1/3
      const xThresh = xScale(THRESHOLD);
      svg.appendChild(
        K.svgEl("rect", {
          x: m.l,
          y: m.t,
          width: Math.max(0, xThresh - m.l),
          height: phPlot,
          fill: hexOrCssToRgba(T.gold, 0.09),
        })
      );

      const yTicks = yScale.ticks(4).filter((v) => v <= yMax + 1e-9);
      K.drawAxes(svg, {
        w,
        h,
        m,
        xScale,
        yScale,
        xTicks: [0, 0.2, 0.4, 0.6, 0.8, 1.0],
        yTicks,
        xFormat: (v) => v.toFixed(1),
        yFormat: (v) => `${Math.round(v * 100)}%`,
        xLabel: "Nearest-neighbour ratio d₁/d₂",
        yLabel: "Share per bin",
        vLines: [
          {
            x: THRESHOLD,
            color: T.gold,
            dash: "4,3",
            width: 1.4,
            dx: 6,
            dy: 12,
            label: "← Memorised (d₁/d₂ < 1/3)",
          },
        ],
      });

      const mids = [];
      for (let i = 0; i < nh.bins.length; i++) {
        mids.push(0.5 * (edges[i] + edges[i + 1]));
      }

      const buildLinePts = (bins) => {
        const pts = [[xScale(edges[0]), yScale(bins[0])]];
        for (let i = 0; i < bins.length; i++) {
          pts.push([xScale(mids[i]), yScale(bins[i])]);
        }
        pts.push([xScale(edges[edges.length - 1]), yScale(bins[bins.length - 1])]);
        return pts;
      };

      const natPts = buildLinePts(nh.bins);
      const permPts = buildLinePts(ph.bins);

      const areaPath = (pts) => {
        const baseY = yScale(0).toFixed(2);
        return (
          K.linePath(pts) +
          ` L${pts[pts.length - 1][0].toFixed(2)},${baseY} L${pts[0][0].toFixed(2)},${baseY} Z`
        );
      };

      // Area fills then crisp lines
      svg.appendChild(K.svgEl("path", { d: areaPath(natPts), fill: `url(#${uid}_nat)` }));
      svg.appendChild(K.svgEl("path", { d: areaPath(permPts), fill: `url(#${uid}_perm)` }));

      svg.appendChild(
        K.svgEl("path", {
          d: K.linePath(natPts),
          fill: "none",
          stroke: T.nat,
          "stroke-width": 2.3,
          "stroke-linejoin": "round",
          "stroke-linecap": "round",
        })
      );
      svg.appendChild(
        K.svgEl("path", {
          d: K.linePath(permPts),
          fill: "none",
          stroke: T.perm,
          "stroke-width": 2.3,
          "stroke-dasharray": "6,4",
          "stroke-linejoin": "round",
          "stroke-linecap": "round",
        })
      );

      // Median markers (vertical stems + diamonds on curve + non-overlapping baseline ticks)
      const drawMedianTick = (medVal, bins, color, dyOffset) => {
        const mx = xScale(clamp(medVal, 0, 1.08));
        const density = sampleDensityAt(edges, bins, medVal);
        const my = yScale(density);
        const base = yScale(0);
        svg.appendChild(
          K.svgEl("line", {
            x1: mx,
            x2: mx,
            y1: my,
            y2: base,
            stroke: color,
            "stroke-width": 1.2,
            "stroke-dasharray": "2,2",
            opacity: 0.75,
          })
        );
        svg.appendChild(K.drawMarker(svg, mx, my, "diamond", 3.6, color, T.surface));
        svg.appendChild(
          K.svgEl("polygon", {
            points: `${mx},${base - 5} ${mx - 3.5},${base + 1} ${mx + 3.5},${base + 1}`,
            fill: color,
          })
        );
        svg.appendChild(
          K.svgEl("text", {
            x: mx,
            y: Math.max(m.t + 22, my - 7 - dyOffset),
            "text-anchor": mx > m.l + pw - 28 ? "end" : mx < m.l + 28 ? "start" : "middle",
            fill: color,
            "font-family": T.mono,
            "font-size": 9.5,
            "font-weight": 600,
            style: `paint-order:stroke;stroke:${T.cardAlt};stroke-width:3.5px;stroke-linejoin:round;`,
            text: `med ${medVal.toFixed(2)}`,
          })
        );
      };

      const medClose = Math.abs(xScale(nh.med) - xScale(ph.med)) < 56;
      drawMedianTick(nh.med, nh.bins, T.nat, 0);
      drawMedianTick(ph.med, ph.bins, T.perm, medClose ? 11 : 0);

      // Hover crosshair layer (updated in place for 60fps responsiveness)
      const hoverLayer = K.svgEl("g", { style: "display:none;pointer-events:none" });
      const vGuide = K.svgEl("line", {
        y1: m.t,
        y2: m.t + phPlot,
        stroke: T.muted,
        "stroke-width": 1.1,
        "stroke-dasharray": "3,3",
      });
      const dotNat = K.svgEl("circle", {
        r: 4.5,
        fill: T.nat,
        stroke: T.surface,
        "stroke-width": 1.5,
      });
      const dotPerm = K.svgEl("circle", {
        r: 4.5,
        fill: T.perm,
        stroke: T.surface,
        "stroke-width": 1.5,
      });
      hoverLayer.appendChild(vGuide);
      hoverLayer.appendChild(dotNat);
      hoverLayer.appendChild(dotPerm);
      svg.appendChild(hoverLayer);

      const updateHover = (clientX, clientY, svgClientRect) => {
        const svgX = ((clientX - svgClientRect.left) / svgClientRect.width) * w;
        const xVal = clamp(xScale.invert(svgX), 0, 1.0799);
        const bw = edges[1] - edges[0];
        const bi = clamp(Math.floor((xVal - edges[0]) / bw), 0, mids.length - 1);
        state.hoveredBin = bi;

        const cx = xScale(mids[bi]);
        vGuide.setAttribute("x1", cx);
        vGuide.setAttribute("x2", cx);
        dotNat.setAttribute("cx", cx);
        dotNat.setAttribute("cy", yScale(nh.bins[bi]));
        dotPerm.setAttribute("cx", cx);
        dotPerm.setAttribute("cy", yScale(ph.bins[bi]));
        hoverLayer.style.display = "block";

        const e0 = edges[bi];
        const e1 = edges[bi + 1];
        const isMem = mids[bi] < THRESHOLD;
        const html =
          `<div class="viz-tooltip__title">d₁/d₂ ∈ [${e0.toFixed(3)}, ${e1.toFixed(3)}) · ${isMem ? "Memorised" : "Novel"}</div>` +
          K.tipRow("Natural share", `${(nh.bins[bi] * 100).toFixed(2)}% (${Math.round(nh.raw[bi] * 1000)}/1k)`, T.nat) +
          K.tipRow("Permuted share", `${(ph.bins[bi] * 100).toFixed(2)}% (${Math.round(ph.raw[bi] * 1000)}/1k)`, T.perm) +
          K.tipRow("Cumulative m(1/3)", `${fmtPct(nh.m13, 1)} / ${fmtPct(ph.m13, 1)}`);
        K.tooltip.show(clientX, clientY, html);
      };

      const trap = K.svgEl("rect", {
        x: m.l,
        y: m.t,
        width: pw,
        height: phPlot,
        fill: "transparent",
        style: "cursor:crosshair",
      });
      trap.addEventListener("mousemove", (e) => {
        updateHover(e.clientX, e.clientY, svg.getBoundingClientRect());
      });
      trap.addEventListener(
        "touchmove",
        (e) => {
          if (e.touches && e.touches[0]) {
            updateHover(e.touches[0].clientX, e.touches[0].clientY, svg.getBoundingClientRect());
          }
        },
        { passive: true }
      );
      trap.addEventListener("mouseleave", () => {
        state.hoveredBin = null;
        hoverLayer.style.display = "none";
        K.tooltip.hide();
      });
      svg.appendChild(trap);

      plotContainer.appendChild(svg);
    }

    function updateGallerySelection() {
      for (const arm of ["nat", "perm", "train"]) {
        for (let i = 0; i < 8; i++) {
          const active = !state.customSample && state.sampleArm === arm && state.sampleIdx === i;
          tileRegistry[arm][i].btn.classList.toggle("is-selected", active);
          tileRegistry[arm][i].btn.setAttribute("aria-pressed", active ? "true" : "false");
        }
      }
    }

    function redrawGallery() {
      const T = K.theme();
      const cc = condCache[state.condKey];
      const { C, H, W, permObj } = cc;
      const perm = state.galleryView === "perm" ? permObj.fwd : null;

      galleryTitle.textContent = `Generated samples at ${K.fmtSteps(cc.raw.samples_step)} & real training images (click any tile to inspect)`;
      natArmLabel.style.color = T.nat;
      permArmLabel.style.color = T.perm;
      trainArmLabel.style.color = T.muted;

      for (const arm of ["nat", "perm", "train"]) {
        const list = cc[arm] || [];
        for (let i = 0; i < 8; i++) {
          const s = list[i];
          const reg = tileRegistry[arm][i];
          if (!s) {
            reg.btn.style.display = "none";
            continue;
          }
          reg.btn.style.display = "";
          renderGridToCanvas(reg.cvs, s, C, H, W, perm, offscreen, 72);
          if (arm === "train") {
            reg.dot.style.display = "none";
            reg.ratioEl.textContent = s.label || `#${s.idx}`;
            reg.btn.setAttribute("aria-label", `Training image #${s.idx}, ${s.label || ""}`);
          } else {
            reg.dot.style.display = "";
            reg.dot.style.background = s.memorized ? T.gold : "transparent";
            reg.dot.style.borderColor = s.memorized ? T.surface : hexOrCssToRgba(T.muted, 0.55);
            reg.dot.title = s.memorized ? "Memorised (d₁/d₂ < 1/3)" : "Novel (d₁/d₂ ≥ 1/3)";
            reg.ratioEl.textContent = s.ratio.toFixed(3);
            reg.btn.setAttribute(
              "aria-label",
              `${arm === "nat" ? "Natural" : "Permuted"} sample #${s.idx}, d1/d2 ${s.ratio.toFixed(3)}`
            );
          }
        }
      }
      updateGallerySelection();
    }

    function redrawLeftSide() {
      scrambleReadout.textContent = `${Math.round(state.scramble * 100)}%`;
      scrambleSlider.value = Math.round(state.scramble * 100);
      redrawCanvasesAndLeftStats();
    }

    function redrawRightSide() {
      updateHeaderAndDensityStats();
      redrawDensityChart();
    }

    function redrawAll() {
      redrawLeftSide();
      redrawRightSide();
      redrawGallery();
    }

    // ---------- Canvas Loupe Pointer Interaction ----------
    function bindCanvasLoupe(canvas, arm) {
      const handlePointer = (clientX, clientY) => {
        const cc = condCache[state.condKey];
        const rect = canvas.getBoundingClientRect();
        if (!rect.width || !rect.height) return;
        const gx = clamp(Math.floor(((clientX - rect.left) / rect.width) * cc.W), 0, cc.W - 1);
        const gy = clamp(Math.floor(((clientY - rect.top) / rect.height) * cc.H), 0, cc.H - 1);
        state.hx = gx;
        state.hy = gy;
        state.hoverArm = arm === "perm" && state.scramble >= 0.9995 ? "perm" : "nat";
        redrawCanvasesAndLeftStats();
      };
      canvas.addEventListener("mousemove", (e) => handlePointer(e.clientX, e.clientY));
      canvas.addEventListener("click", (e) => handlePointer(e.clientX, e.clientY));
      canvas.addEventListener(
        "touchstart",
        (e) => {
          if (e.touches && e.touches[0]) {
            handlePointer(e.touches[0].clientX, e.touches[0].clientY);
          }
        },
        { passive: true }
      );
    }

    bindCanvasLoupe(natCanvas, "nat");
    bindCanvasLoupe(permCanvas, "perm");

    // ---------- Control Listeners ----------
    scrambleSlider.addEventListener("input", () => {
      stopScrambleAnim();
      state.scramble = Number(scrambleSlider.value) / 100;
      if (state.scramble < 0.9995 && state.hoverArm === "perm") {
        state.hoverArm = "nat";
      }
      redrawLeftSide();
    });

    animScrambleBtn.addEventListener("click", () => {
      if (state.animRaf !== null) {
        stopScrambleAnim();
        return;
      }
      if (prefersReduced) {
        state.scramble = state.scramble >= 0.5 ? 0 : 1;
        if (state.scramble < 0.9995) state.hoverArm = "nat";
        redrawLeftSide();
        return;
      }
      const from = state.scramble >= 0.98 ? 0 : state.scramble;
      const to = 1.0;
      const duration = 1350;
      const start = performance.now();
      state.hoverArm = "nat";
      animScrambleBtn.textContent = "Pause Anim";

      const tick = (now) => {
        const u = clamp((now - start) / duration, 0, 1);
        state.scramble = from + (to - from) * u;
        redrawLeftSide();
        if (u < 1) {
          state.animRaf = requestAnimationFrame(tick);
        } else {
          state.animRaf = null;
          animScrambleBtn.textContent = "Animate Scramble";
        }
      };
      state.animRaf = requestAnimationFrame(tick);
    });

    resetBtn.addEventListener("click", () => {
      stopPlay();
      stopScrambleAnim();
      const cc = condCache[state.condKey];
      const defIdx = cc.raw.steps.indexOf(cc.raw.default_step);
      state.stepIdx = defIdx >= 0 ? defIdx : 0;
      state.scramble = 1.0;
      state.hx = Math.floor(cc.W / 2);
      state.hy = Math.floor(cc.H / 2);
      state.hoverArm = "nat";
      syncStepSlider();
      redrawLeftSide();
      redrawRightSide();
    });

    stepSlider.addEventListener("input", () => {
      stopPlay();
      state.stepIdx = Number(stepSlider.value);
      redrawRightSide();
    });

    playBtn.addEventListener("click", () => {
      if (state.playTimer !== null) {
        stopPlay();
        return;
      }
      const steps = condCache[state.condKey].raw.steps;
      if (state.stepIdx >= steps.length - 1) {
        state.stepIdx = 0;
        stepSlider.value = 0;
        redrawRightSide();
      }
      const delay = prefersReduced ? 1100 : 800;
      state.playTimer = setInterval(() => {
        if (state.stepIdx < steps.length - 1) {
          state.stepIdx += 1;
          stepSlider.value = state.stepIdx;
          redrawRightSide();
        } else {
          stopPlay();
        }
      }, delay);
      syncPlayButton();
    });

    syncStepSlider();
    redrawAll();
    K.mountResponsive(container, redrawAll);
    if (typeof K.observeAutoHide === "function") K.observeAutoHide(container);
  }

  window.DiffMemWidgets = window.DiffMemWidgets || {};
  window.DiffMemWidgets.mountHero = mountHero;
})();
