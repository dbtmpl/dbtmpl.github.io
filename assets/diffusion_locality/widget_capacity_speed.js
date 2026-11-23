/* Interactive Widgets 3 (Capacity Explorer) & 4 (Clock Speed / Data Collapse) */
(function () {
  "use strict";

  const K = window.DiffMemKit;
  window.DiffMemWidgets = window.DiffMemWidgets || {};

  const REPS = [
    { key: "rep0", label: "rep0 (seed 0)", short: "rep0", shape: "circle" },
    { key: "repA", label: "repA (seed 1)", short: "repA", shape: "square" },
    { key: "repB", label: "repB (seed 2)", short: "repB", shape: "triangle" },
  ];

  const BUDGET_STEPS = [10000, 20000, 30000, 40000, 50000, 70000, 100000];
  const SPEED_NS = [512, 1024, 2048, 4096];

  function isNum(v) {
    return typeof v === "number" && Number.isFinite(v);
  }

  function mean(arr) {
    const valid = arr.filter(isNum);
    if (!valid.length) return null;
    return valid.reduce((a, b) => a + b, 0) / valid.length;
  }

  function geomMean(arr) {
    const valid = arr.filter((v) => isNum(v) && v > 0);
    if (!valid.length) return null;
    const sumLog = valid.reduce((a, b) => a + Math.log2(b), 0);
    return Math.pow(2, sumLog / valid.length);
  }

  function spread(arr) {
    const valid = arr.filter(isNum);
    if (valid.length < 2) return null;
    return Math.max(...valid) - Math.min(...valid);
  }

  function fmtSigned(v, dec = 2) {
    if (!isNum(v)) return "—";
    const abs = Math.abs(v).toFixed(dec);
    if (Number(abs) === 0) return `0.${"0".repeat(dec)}`;
    return (v < 0 ? "\u2212" : "+") + abs;
  }

  function fmtNum(v, dec = 2) {
    return isNum(v) ? v.toFixed(dec) : "—";
  }

  function fmtInt(v) {
    return isNum(v) ? Math.round(v).toLocaleString("en-US") : "—";
  }

  function fmtStepShort(s) {
    if (!isNum(s)) return String(s || "—");
    if (s >= 1000) {
      const k = s / 1000;
      return (k >= 10 || Math.abs(k - Math.round(k)) < 0.05 ? Math.round(k) : k.toFixed(1)) + "k";
    }
    return String(Math.round(s));
  }

  function fmtStepStat(s) {
    if (!isNum(s)) return String(s || "—");
    if (s >= 100000) return `${Math.round(s / 1000)}k`;
    if (s >= 1000) return `${(s / 1000).toFixed(1)}k`;
    return String(Math.round(s));
  }

  function fmtTau(vals) {
    const nums = vals.filter(isNum);
    if (nums.length === vals.length && nums.length > 0) {
      return fmtStepStat(mean(nums));
    }
    const strs = vals.filter((v) => typeof v === "string");
    if (strs.length) {
      const s = strs[0];
      const m = /^>\s*(\d+)/.exec(s);
      if (m) return `> ${fmtStepShort(Number(m[1]))}`;
      return s;
    }
    return "—";
  }

  function legendSvgSwatch(opts) {
    const { lineColor = null, dash = "", shape = null, fill = null, stroke = null, width = 20 } = opts;
    const svg = K.svgEl("svg", {
      width,
      height: 12,
      viewBox: `0 0 ${width} 12`,
      style: "display:inline-block;vertical-align:middle;flex-shrink:0;",
    });
    if (lineColor) {
      svg.appendChild(
        K.svgEl("line", {
          x1: 1,
          x2: width - 1,
          y1: 6,
          y2: 6,
          stroke: lineColor,
          "stroke-width": 2,
          "stroke-dasharray": dash || null,
        })
      );
    }
    if (shape) {
      svg.appendChild(
        K.drawMarker(svg, width / 2, 6, shape, 3.2, fill || lineColor || "#888", stroke || "#fff")
      );
    }
    return svg;
  }

  function makeStatCard(keyText) {
    const valNode = K.el("div", { class: "viz-stat__v", text: "—" });
    const subNode = K.el("div", {
      style: "font-size:11px;color:var(--muted);font-family:var(--mono);margin-top:1px;",
    });
    const card = K.el("div", { class: "viz-stat" }, [
      K.el("div", { class: "viz-stat__k", text: keyText }),
      valNode,
      subNode,
    ]);
    return {
      card,
      set(val, sub = "", color = "") {
        valNode.textContent = val;
        valNode.style.color = color || "";
        subNode.textContent = sub;
        subNode.style.display = sub ? "block" : "none";
      },
    };
  }

  function interpLogStep(pts, targetStep) {
    if (!pts || !pts.length || targetStep <= 0) return null;
    const lx = Math.log2(targetStep);
    const l0 = Math.log2(pts[0][0]);
    const l1 = Math.log2(pts[pts.length - 1][0]);
    if (lx < l0 - 1e-9 || lx > l1 + 1e-9) return null;
    if (lx <= l0) return pts[0][1];
    if (lx >= l1) return pts[pts.length - 1][1];
    for (let i = 0; i < pts.length - 1; i++) {
      const a = Math.log2(pts[i][0]);
      const b = Math.log2(pts[i + 1][0]);
      if (lx >= a && lx <= b) {
        const t = (lx - a) / ((b - a) || 1);
        return pts[i][1] + t * (pts[i + 1][1] - pts[i][1]);
      }
    }
    return null;
  }

  function computeCollapseRMSE(natRaw, permRaw, rhoShift) {
    if (!natRaw || !permRaw || natRaw.length < 2 || permRaw.length < 2) return null;
    const factor = Math.pow(2, rhoShift);
    const permShifted = permRaw.map(([s, m]) => [s / factor, m]);
    const lo = Math.max(Math.log2(natRaw[0][0]), Math.log2(permShifted[0][0]));
    const hi = Math.min(
      Math.log2(natRaw[natRaw.length - 1][0]),
      Math.log2(permShifted[permShifted.length - 1][0])
    );
    if (hi <= lo) return null;
    const N = 80;
    let sq = 0;
    let count = 0;
    for (let i = 0; i <= N; i++) {
      const step = Math.pow(2, lo + (i / N) * (hi - lo));
      const yn = interpLogStep(natRaw, step);
      const yp = interpLogStep(permShifted, step);
      if (isNum(yn) && isNum(yp)) {
        sq += (yn - yp) * (yn - yp);
        count++;
      }
    }
    return count > 0 ? Math.sqrt(sq / count) : null;
  }

  function drawAnnotationBox(svg, xRight, yTop, lines, align = "right") {
    const T = K.theme();
    const g = K.svgEl("g", { style: "pointer-events:none;" });
    const padX = 8;
    const padY = 6;
    const lineH = 14.5;
    const maxChars = Math.max(...lines.map((l) => l.text.length));
    const estW = Math.max(108, maxChars * 6.3 + padX * 2 + (lines.some((l) => l.dot) ? 11 : 0));
    const boxH = lines.length * lineH + padY * 2 - 2;
    const x0 = align === "right" ? xRight - estW : xRight;
    g.appendChild(
      K.svgEl("rect", {
        x: x0,
        y: yTop,
        width: estW,
        height: boxH,
        rx: 6,
        fill: T.surface,
        "fill-opacity": 0.92,
        stroke: T.border,
        "stroke-width": 1,
      })
    );
    lines.forEach((ln, idx) => {
      const cy = yTop + padY + idx * lineH + 10;
      let tx = x0 + padX;
      if (ln.dot) {
        g.appendChild(K.svgEl("circle", { cx: tx + 3.5, cy: cy - 3.5, r: 3, fill: ln.dot }));
        tx += 10;
      }
      g.appendChild(
        K.svgEl("text", {
          x: tx,
          y: cy,
          fill: ln.color || T.text,
          "font-family": ln.mono ? T.mono : T.sans,
          "font-size": 11,
          "font-weight": ln.bold ? 600 : 500,
          text: ln.text,
        })
      );
    });
    svg.appendChild(g);
  }

  /* ============================================================================
   * WIDGET 3: Capacity Explorer (mountCapacity)
   * ============================================================================ */
  function mountCapacity(container, data) {
    K.clear(container);

    const state = {
      dataset: "both", // "mnist" | "fmnist" | "both"
      model: "both", // "both" | "unet" | "mlp"
      threshold: "1_3", // "1_3" | "1_2"
      showPermSeeds: false,
      stepIdx: BUDGET_STEPS.length - 1, // 100000
    };

    // ---- Header
    const header = K.el("div", { class: "viz-header" }, [
      K.el("div", {}, [
        K.el("h3", {
          class: "viz-title",
          text: "Capacity Explorer: Memorised Fraction m(n) Across Budgets, Replicates & the MLP Null",
        }),
        K.el("p", {
          class: "viz-subtitle",
          text: "Scrub the training budget to watch n* drift upward; compare the 3-replicate U-Net against the permutation-equivariant MLP pipeline null.",
        }),
      ]),
    ]);

    // ---- Controls (created once!)
    const dsSeg = K.makeSeg(
      [
        { value: "mnist", label: "MNIST" },
        { value: "fmnist", label: "FashionMNIST" },
        { value: "both", label: "Side-by-Side" },
      ],
      state.dataset,
      (v) => {
        state.dataset = v;
        updateControlsUI();
        redraw();
      }
    );

    const modelSeg = K.makeSeg(
      [
        { value: "both", label: "U-Net + MLP Null" },
        { value: "unet", label: "U-Net only" },
        { value: "mlp", label: "MLP Null only" },
      ],
      state.model,
      (v) => {
        state.model = v;
        updateControlsUI();
        redraw();
      }
    );

    const thrSeg = K.makeSeg(
      [
        { value: "1_3", label: "d\u2081/d\u2082 < 1/3" },
        { value: "1_2", label: "d\u2081/d\u2082 < 1/2" },
      ],
      state.threshold,
      (v) => {
        state.threshold = v;
        updateControlsUI();
        redraw();
      }
    );

    const permCheckInput = K.el("input", {
      type: "checkbox",
      onChange: (e) => {
        state.showPermSeeds = Boolean(e.target.checked);
        updateControlsUI();
        redraw();
      },
    });
    const permNote = K.el("span", {
      style: "font-size:11.5px;color:var(--muted);",
      text: "",
    });
    const permCheckLabel = K.el("label", { class: "viz-check" }, [
      permCheckInput,
      K.el("span", { text: "Show 3 permutation seeds (MNIST 100k)" }),
      permNote,
    ]);

    const budgetReadout = K.el("span", { class: "viz-readout", text: "Budget: 100k steps" });
    const budgetSlider = K.el("input", {
      type: "range",
      class: "viz-slider",
      min: 0,
      max: BUDGET_STEPS.length - 1,
      step: 1,
      value: state.stepIdx,
      "aria-label": "Training budget in optimizer steps",
      onInput: (e) => {
        state.stepIdx = Number(e.target.value);
        updateControlsUI();
        redraw();
      },
    });

    const controls = K.el("div", { class: "viz-controls" }, [
      K.el("div", { class: "viz-control-group" }, [
        K.el("span", { class: "viz-control-label", text: "Dataset" }),
        dsSeg,
      ]),
      K.el("div", { class: "viz-control-group" }, [
        K.el("span", { class: "viz-control-label", text: "Model" }),
        modelSeg,
      ]),
      K.el("div", { class: "viz-control-group" }, [
        K.el("span", { class: "viz-control-label", text: "Threshold" }),
        thrSeg,
      ]),
      K.el("div", { class: "viz-slider-wrap" }, [
        K.el("span", { class: "viz-control-label", text: "Budget" }),
        budgetSlider,
        budgetReadout,
      ]),
      K.el("div", { class: "viz-control-group" }, [permCheckLabel]),
    ]);

    // ---- Dynamic plot rows
    const unetRow = K.el("div");
    const mlpRow = K.el("div", { style: "margin-top:16px;" });
    const plotsWrap = K.el("div", {}, [unetRow, mlpRow]);

    // ---- Shared legend & figcaption
    const legendWrap = K.el("div", { class: "viz-legend" });
    const figcaption = K.el("div", { class: "viz-figcaption" }, [
      K.el("b", { text: "How to read: " }),
      "Solid teal curves show Natural training; dashed orange curves show Pixel-Permuted training (same subsets, init seeds, and optimizer). ",
      "Each marker shape denotes an independent replicate (circle = rep0, square = repA, triangle = repB). ",
      "For U-Net, n* is linearly interpolated in log\u2082 n at m = 0.5; for the MLP null, n* uses the last downward crossing at m = 0.25 to handle small-n undertraining.",
    ]);

    container.appendChild(header);
    container.appendChild(controls);
    container.appendChild(plotsWrap);
    container.appendChild(legendWrap);
    container.appendChild(figcaption);

    function currentStep() {
      return BUDGET_STEPS[state.stepIdx] || 100000;
    }

    function updateControlsUI() {
      const step = currentStep();
      budgetReadout.textContent = `Budget: ${fmtStepShort(step)} steps`;
      const mnistVisible = state.dataset === "mnist" || state.dataset === "both";
      const unetVisible = state.model === "unet" || state.model === "both";
      if (state.showPermSeeds && (!mnistVisible || !unetVisible || step !== 100000 || state.threshold !== "1_3")) {
        permNote.textContent = "(requires MNIST U-Net @ 100k, d\u2081/d\u2082 < 1/3)";
      } else {
        permNote.textContent = "";
      }
    }

    function buildLegend() {
      const T = K.theme();
      K.clear(legendWrap);
      const items = [
        { label: "Natural arm", swatch: { lineColor: T.nat } },
        { label: "Permuted arm (seed 1234)", swatch: { lineColor: T.perm, dash: "5,3" } },
        { label: "rep0 (seed 0)", swatch: { shape: "circle", fill: T.muted, stroke: T.surface, width: 14 } },
        { label: "repA (seed 1)", swatch: { shape: "square", fill: T.muted, stroke: T.surface, width: 14 } },
        { label: "repB (seed 2)", swatch: { shape: "triangle", fill: T.muted, stroke: T.surface, width: 14 } },
      ];
      if (state.showPermSeeds && data.permctl) {
        items.push({ label: "Perm seed 1 (MNIST 100k)", swatch: { lineColor: T.gold, dash: "3,2.5", shape: "diamond", fill: T.gold, stroke: T.gold } });
        items.push({ label: "Perm seed 2 (MNIST 100k)", swatch: { lineColor: T.gold, dash: "1.5,2.5", shape: "diamond", fill: T.surface, stroke: T.gold } });
      }
      for (const it of items) {
        legendWrap.appendChild(
          K.el("span", { class: "viz-legend__item" }, [
            legendSvgSwatch(it.swatch),
            K.el("span", { text: it.label }),
          ])
        );
      }
    }

    function getDeltaEntries(ds, modelKey) {
      const step = currentStep();
      const deltaKey = state.threshold === "1_3" ? "delta_1_3" : "delta_1_2";
      const out = [];
      for (const rep of REPS) {
        const arr = data[ds]?.[modelKey]?.[rep.key]?.[deltaKey];
        if (!Array.isArray(arr) || !arr.length) continue;
        const row = arr.find((d) => d.step === step);
        if (row) out.push({ rep, row });
      }
      return out;
    }

    function computeUnetStats(entries) {
      const nats = entries.map((e) => e.row.log2_nstar_nat).filter(isNum);
      const perms = entries.map((e) => e.row.log2_nstar_perm).filter(isNum);
      const deltas = entries.map((e) => e.row.delta).filter(isNum);
      const sNat = spread(nats);
      const sPerm = spread(perms);
      const sigma =
        sNat === null && sPerm === null ? null : Math.max(sNat ?? 0, sPerm ?? 0);
      const meanNat = mean(nats);
      const meanPerm = mean(perms);
      const meanDelta = mean(deltas);

      let verdict = { text: "CENSORED", cls: "viz-badge--warn" };
      if (deltas.length === entries.length && entries.length >= 2) {
        const sEff = sigma ?? 0;
        if (deltas.every((d) => Math.abs(d) <= Math.max(0.5, sEff))) {
          verdict = { text: "NO SHIFT", cls: "viz-badge--nat" };
        } else if (
          deltas.every((d) => Math.abs(d) >= Math.max(1.0, 2 * sEff)) &&
          new Set(deltas.map((d) => Math.sign(d))).size === 1
        ) {
          verdict = { text: "SHIFT", cls: "viz-badge--perm" };
        } else {
          verdict = { text: "INCONCLUSIVE", cls: "viz-badge--warn" };
        }
      } else {
        const anyBelow = entries.some(
          (e) => e.row.flag_perm === "below_grid" || e.row.flag_nat === "below_grid"
        );
        verdict = { text: anyBelow ? "BELOW GRID" : "CENSORED", cls: "viz-badge--warn" };
      }

      return { meanNat, meanPerm, meanDelta, sigma, verdict };
    }

    function computeMlpStats(ds, entries) {
      const step = currentStep();
      const deltaKey = state.threshold === "1_3" ? "delta_1_3" : "delta_1_2";
      // Check pre-registered null across confirmatory reps (repA, repB) across all budgets
      const confDeltas = [];
      const confRhos = [];
      for (const rk of ["repA", "repB"]) {
        const arr = data[ds]?.mlp?.[rk]?.[deltaKey] || [];
        for (const r of arr) {
          if (isNum(r.delta)) confDeltas.push(Math.abs(r.delta));
        }
        if (state.threshold === "1_3") {
          const rArr = data[ds]?.mlp?.[rk]?.rho_1_3 || [];
          for (const rr of rArr) {
            if (isNum(rr.rho)) confRhos.push(Math.abs(rr.rho));
          }
        }
      }
      const maxAbsDelta = confDeltas.length ? Math.max(...confDeltas) : null;
      const maxAbsRho = confRhos.length ? Math.max(...confRhos) : null;
      const overallPass =
        confDeltas.length > 0 &&
        maxAbsDelta <= 0.5 &&
        (maxAbsRho === null || maxAbsRho <= 0.5);

      if (!entries.length) {
        return {
          noEvalAtStep: true,
          meanNat: null,
          meanPerm: null,
          meanDelta: null,
          maxAbsDelta,
          verdict: { text: `NO EVAL @ ${fmtStepShort(step)}`, cls: "" },
        };
      }

      const nats = entries.map((e) => e.row.log2_nstar_nat).filter(isNum);
      const perms = entries.map((e) => e.row.log2_nstar_perm).filter(isNum);
      const deltas = entries.map((e) => e.row.delta).filter(isNum);
      const meanNat = mean(nats);
      const meanPerm = mean(perms);
      const meanDelta = mean(deltas);

      let verdict;
      if (!deltas.length) {
        verdict = {
          text: overallPass ? "PASS (BELOW 0.25 HERE)" : "BELOW 0.25",
          cls: overallPass ? "viz-badge--nat" : "",
        };
      } else {
        verdict = overallPass
          ? { text: "PASS", cls: "viz-badge--nat" }
          : { text: "FAIL", cls: "viz-badge--perm" };
      }

      return { noEvalAtStep: false, meanNat, meanPerm, meanDelta, maxAbsDelta, verdict };
    }

    function renderCapacityPanel(ds, modelKey, panelWidth) {
      const T = K.theme();
      const step = currentStep();
      const dsName = ds === "mnist" ? "MNIST" : "FashionMNIST";
      const isUnet = modelKey === "unet";
      const entries = getDeltaEntries(ds, modelKey);
      const stats = isUnet ? computeUnetStats(entries) : computeMlpStats(ds, entries);

      const badge = K.el("span", {
        class: `viz-badge ${stats.verdict.cls}`,
        text: stats.verdict.text,
      });

      const titleText = isUnet
        ? `${dsName} \u00b7 U-Net (3 reps)`
        : `${dsName} \u00b7 MLP Null (${entries.length || (ds === "fmnist" ? 2 : 3)} reps)`;

      const panel = K.el("div", { class: "viz-panel" }, [
        K.el("div", { class: "viz-panel-title" }, [
          K.el("span", { text: titleText }),
          badge,
        ]),
      ]);

      // SVG dimensions
      const w = Math.max(270, Math.floor(panelWidth - 28));
      const h = isUnet ? (state.dataset === "both" ? 248 : 276) : state.dataset === "both" ? 232 : 254;
      const m = { l: 46, r: 16, t: 18, b: 38 };
      const pw = w - m.l - m.r;
      const ph = h - m.t - m.b;

      const xDomain = isUnet ? [420, 5000] : [50, 1280];
      const xTicks = isUnet ? [512, 1024, 2048, 4096] : [64, 128, 256, 512, 1024];
      const xScale = K.scaleLog(xDomain, [m.l, m.l + pw], 2);
      const yScale = K.scaleLinear([0, 1], [m.t + ph, m.t]);

      const svg = K.makeSvg(w, h);
      K.drawAxes(svg, {
        w,
        h,
        m,
        xScale,
        yScale,
        xTicks,
        yTicks: [0, 0.25, 0.5, 0.75, 1.0],
        xFormat: (v) => String(v),
        yFormat: (v) => v.toFixed(2).replace(/\.00$/, ".0"),
        xLabel: "Training set size n (log\u2082 scale)",
        yLabel: "Memorised fraction m(n)",
        hLines: [
          {
            y: isUnet ? 0.5 : 0.25,
            color: T.muted,
            dash: "3,3",
            width: 1.2,
            label: isUnet ? "onset m = 0.5" : "null level m = 0.25 (last \u2193)",
            align: "right",
          },
        ],
      });

      const hoverLine = K.svgEl("line", {
        x1: 0,
        x2: 0,
        y1: m.t,
        y2: m.t + ph,
        stroke: T.muted,
        "stroke-width": 1,
        "stroke-dasharray": "2,2",
        opacity: 0,
        style: "pointer-events:none;",
      });
      svg.appendChild(hoverLine);

      // Handle MLP missing 40k evaluation cleanly
      if (!isUnet && stats.noEvalAtStep) {
        const bw = Math.min(pw - 24, 300);
        const bh = 54;
        const bx = m.l + (pw - bw) / 2;
        const by = m.t + (ph - bh) / 2 - 6;
        svg.appendChild(
          K.svgEl("rect", {
            x: bx,
            y: by,
            width: bw,
            height: bh,
            rx: 8,
            fill: T.surface,
            stroke: T.border,
          })
        );
        svg.appendChild(
          K.svgEl("text", {
            x: bx + bw / 2,
            y: by + 22,
            "text-anchor": "middle",
            fill: T.text,
            "font-family": T.sans,
            "font-size": 12,
            "font-weight": 600,
            text: `MLP null not evaluated at ${fmtStepShort(step)} steps`,
          })
        );
        svg.appendChild(
          K.svgEl("text", {
            x: bx + bw / 2,
            y: by + 40,
            "text-anchor": "middle",
            fill: T.muted,
            "font-family": T.sans,
            "font-size": 11,
            text: "Checkpoints: 5k, 10k, 20k, 30k, 50k, 70k, 100k",
          })
        );
      } else {
        // Plot lines first, then markers, then invisible hit targets on top
        const lineLayer = K.svgEl("g");
        const markerLayer = K.svgEl("g");
        const hitLayer = K.svgEl("g");
        svg.appendChild(lineLayer);
        svg.appendChild(markerLayer);

        // Draw subtle n* crossing indicators for U-Net when both mean n* are defined
        if (isUnet && isNum(stats.meanNat) && isNum(stats.meanPerm)) {
          const xNat = xScale(Math.pow(2, stats.meanNat));
          const xPerm = xScale(Math.pow(2, stats.meanPerm));
          const y05 = yScale(0.5);
          lineLayer.appendChild(
            K.svgEl("line", {
              x1: xNat,
              x2: xNat,
              y1: y05,
              y2: yScale(0),
              stroke: T.nat,
              "stroke-width": 1.1,
              "stroke-dasharray": "2,2",
              opacity: 0.45,
            })
          );
          lineLayer.appendChild(
            K.svgEl("line", {
              x1: xPerm,
              x2: xPerm,
              y1: y05,
              y2: yScale(0),
              stroke: T.perm,
              "stroke-width": 1.1,
              "stroke-dasharray": "2,2",
              opacity: 0.45,
            })
          );
          if (Math.abs(xNat - xPerm) > 10) {
            lineLayer.appendChild(
              K.svgEl("line", {
                x1: xPerm,
                x2: xNat,
                y1: y05 - 7,
                y2: y05 - 7,
                stroke: T.text,
                "stroke-width": 1.4,
                opacity: 0.7,
              })
            );
          }
        }

        // Draw each replicate arm
        for (const { rep, row } of entries) {
          for (const arm of ["nat", "perm"]) {
            const mObj = arm === "nat" ? row.m_nat : row.m_perm;
            if (!mObj) continue;
            const ns = Object.keys(mObj)
              .map(Number)
              .filter(Number.isFinite)
              .sort((a, b) => a - b);
            if (!ns.length) continue;

            const color = arm === "nat" ? T.nat : T.perm;
            const pts = ns.map((n) => [xScale(n), yScale(mObj[String(n)])]);

            lineLayer.appendChild(
              K.svgEl("path", {
                d: K.linePath(pts),
                fill: "none",
                stroke: color,
                "stroke-width": 1.9,
                "stroke-dasharray": arm === "perm" ? "5,3" : null,
                opacity: 0.86,
              })
            );

            ns.forEach((n, idx) => {
              const [cx, cy] = pts[idx];
              const mk = K.drawMarker(svg, cx, cy, rep.shape, 3.8, color, T.surface);
              markerLayer.appendChild(mk);

              const hit = K.svgEl("circle", {
                cx,
                cy,
                r: 10,
                fill: "transparent",
                style: "cursor:pointer;",
              });
              hit.addEventListener("mouseenter", (ev) => {
                mk.setAttribute("transform", `translate(${cx},${cy}) scale(1.45) translate(${-cx},${-cy})`);
                hoverLine.setAttribute("x1", cx);
                hoverLine.setAttribute("x2", cx);
                hoverLine.setAttribute("opacity", "0.55");
                showPointTooltip(ev, dsName, isUnet, step, n, rep, arm, entries, stats, isUnet && ds === "mnist" && step === 100000 && state.threshold === "1_3" && state.showPermSeeds);
              });
              hit.addEventListener("mousemove", (ev) => {
                showPointTooltip(ev, dsName, isUnet, step, n, rep, arm, entries, stats, isUnet && ds === "mnist" && step === 100000 && state.threshold === "1_3" && state.showPermSeeds);
              });
              hit.addEventListener("mouseleave", () => {
                mk.removeAttribute("transform");
                hoverLine.setAttribute("opacity", "0");
                K.tooltip.hide();
              });
              hitLayer.appendChild(hit);
            });
          }
        }

        // Overlay 3 permutation seeds if active (MNIST U-Net @ 100k, d1/d2 < 1/3)
        const permOverlayActive =
          isUnet &&
          ds === "mnist" &&
          step === 100000 &&
          state.threshold === "1_3" &&
          state.showPermSeeds &&
          data.permctl;

        if (permOverlayActive) {
          const seedConfigs = [
            { seed: "1", dash: "3,2.5", fill: T.gold, opacity: 0.96 },
            { seed: "2", dash: "1.5,2.5", fill: T.surface, opacity: 0.92 },
          ];
          for (const sc of seedConfigs) {
            const pc = data.permctl[sc.seed];
            if (!pc || !pc.m_perm_100k) continue;
            const ns = Object.keys(pc.m_perm_100k)
              .map(Number)
              .sort((a, b) => a - b);
            const pts = ns.map((n) => [xScale(n), yScale(pc.m_perm_100k[String(n)])]);
            lineLayer.appendChild(
              K.svgEl("path", {
                d: K.linePath(pts),
                fill: "none",
                stroke: T.gold,
                "stroke-width": 2.1,
                "stroke-dasharray": sc.dash,
                opacity: sc.opacity,
              })
            );
            ns.forEach((n, idx) => {
              const [cx, cy] = pts[idx];
              const mk = K.drawMarker(svg, cx, cy, "diamond", 4.1, sc.fill, T.gold);
              markerLayer.appendChild(mk);

              const hit = K.svgEl("circle", { cx, cy, r: 10, fill: "transparent", style: "cursor:pointer;" });
              const showSeedTip = (ev) => {
                mk.setAttribute("transform", `translate(${cx},${cy}) scale(1.4) translate(${-cx},${-cy})`);
                hoverLine.setAttribute("x1", cx);
                hoverLine.setAttribute("x2", cx);
                hoverLine.setAttribute("opacity", "0.55");
                const rows = [
                  K.tipRow(`Perm seed ${sc.seed} (m)`, pc.m_perm_100k[String(n)].toFixed(3), T.gold),
                  K.tipRow("Seed 1234 (rep0)", data.permctl["1234"]?.m_perm_100k?.[String(n)]?.toFixed(3) ?? "—", T.perm),
                  K.tipRow(`Delta (seed ${sc.seed})`, fmtSigned(pc.delta_100k), T.gold),
                  K.tipRow("log\u2082 n*_perm", pc.log2_nstar_perm_100k.toFixed(2), T.muted),
                ];
                K.tooltip.show(
                  ev.clientX,
                  ev.clientY,
                  `<div class="viz-tooltip__title">MNIST \u00b7 Perm Seed ${sc.seed} \u00b7 n = ${fmtInt(n)}</div>${rows.join("")}`
                );
              };
              hit.addEventListener("mouseenter", showSeedTip);
              hit.addEventListener("mousemove", showSeedTip);
              hit.addEventListener("mouseleave", () => {
                mk.removeAttribute("transform");
                hoverLine.setAttribute("opacity", "0");
                K.tooltip.hide();
              });
              hitLayer.appendChild(hit);
            });
          }
        }

        // Non-monotone undertraining peak callout on MLP (e.g. FMNIST n=96 > n=64)
        if (!isUnet && entries.length) {
          let max64 = 0;
          let max96 = 0;
          let has96 = false;
          for (const { row } of entries) {
            if (isNum(row.m_nat?.["64"])) max64 = Math.max(max64, row.m_nat["64"]);
            if (isNum(row.m_perm?.["64"])) max64 = Math.max(max64, row.m_perm["64"]);
            if (isNum(row.m_nat?.["96"])) {
              has96 = true;
              max96 = Math.max(max96, row.m_nat["96"]);
            }
            if (isNum(row.m_perm?.["96"])) {
              has96 = true;
              max96 = Math.max(max96, row.m_perm["96"]);
            }
          }
          if (has96 && max96 > max64 + 0.015) {
            const px = xScale(96);
            const py = yScale(max96);
            svg.appendChild(
              K.svgEl("text", {
                x: px + 6,
                y: Math.max(m.t + 12, py - 10),
                fill: T.muted,
                "font-family": T.sans,
                "font-size": 10.5,
                "font-weight": 600,
                text: "undertraining peak (n=96)",
              })
            );
          }
        }

        svg.appendChild(hitLayer);

        // Annotation box in collision-free top-right corner
        if (isUnet) {
          const annLines = [];
          if (isNum(stats.meanDelta)) {
            annLines.push({
              text: `mean \u0394 = ${fmtSigned(stats.meanDelta)}  \u00b7  \u03c3 = ${fmtNum(stats.sigma)}`,
              mono: true,
              bold: true,
            });
          } else {
            annLines.push({ text: "\u0394: censored (< 512)", mono: true, bold: true, color: T.gold });
          }
          if (permOverlayActive) {
            annLines.push({
              text: `1234: ${fmtSigned(data.permctl["1234"]?.delta_100k)}  |  1: ${fmtSigned(data.permctl["1"]?.delta_100k)}  |  2: ${fmtSigned(data.permctl["2"]?.delta_100k)}`,
              mono: true,
              dot: T.gold,
              color: T.gold,
            });
          } else {
            annLines.push({
              text: "pre-registered |Δ| \u2264 max(0.5, \u03c3)",
              color: T.muted,
            });
          }
          drawAnnotationBox(svg, m.l + pw - 2, m.t + 2, annLines, "right");
        } else {
          const annLines = [
            {
              text: isNum(stats.meanDelta)
                ? `mean \u0394 = ${fmtSigned(stats.meanDelta)}`
                : "below 0.25 at this step",
              mono: isNum(stats.meanDelta),
              bold: true,
            },
            {
              text: isNum(stats.maxAbsDelta)
                ? `max |\u0394| = ${stats.maxAbsDelta.toFixed(2)} \u2264 0.50`
                : "m = 0.25 last-\u2193 crossing",
              color: T.muted,
            },
          ];
          drawAnnotationBox(svg, m.l + pw - 2, m.t + 2, annLines, "right");
        }
      }

      panel.appendChild(svg);

      // Compact readout strip inside panel
      const strip = K.el("div", {
        class: "viz-stat-strip",
        style: "grid-template-columns: repeat(auto-fit, minmax(86px, 1fr)); gap: 6px;",
      });
      const cNat = makeStatCard(isUnet ? "n*_nat" : "n*_nat (0.25)");
      const cPerm = makeStatCard(isUnet ? "n*_perm" : "n*_perm (0.25)");
      const cDelta = makeStatCard("Mean \u0394");
      const cFourth = makeStatCard(isUnet ? "Noise \u03c3" : "Max |\u0394|");
      const cVerdict = makeStatCard("Verdict");

      if (isUnet) {
        cNat.set(
          isNum(stats.meanNat) ? fmtInt(Math.pow(2, stats.meanNat)) : "< 512",
          isNum(stats.meanNat) ? `2^${stats.meanNat.toFixed(2)}` : "below grid",
          T.nat
        );
        cPerm.set(
          isNum(stats.meanPerm) ? fmtInt(Math.pow(2, stats.meanPerm)) : "< 512",
          isNum(stats.meanPerm) ? `2^${stats.meanPerm.toFixed(2)}` : "below grid",
          T.perm
        );
        cDelta.set(
          fmtSigned(stats.meanDelta),
          isNum(stats.meanDelta) ? `${Math.pow(2, stats.meanDelta).toFixed(2)}\u00d7` : "censored"
        );
        cFourth.set(
          fmtNum(stats.sigma),
          isNum(stats.sigma) ? "max rep spread" : "—"
        );
        cVerdict.set(
          stats.verdict.text,
          step === 100000 ? "final budget" : `@ ${fmtStepShort(step)}`,
          stats.verdict.cls === "viz-badge--nat"
            ? T.nat
            : stats.verdict.cls === "viz-badge--perm"
            ? T.perm
            : T.gold
        );
      } else {
        cNat.set(
          isNum(stats.meanNat) ? fmtInt(Math.pow(2, stats.meanNat)) : "—",
          isNum(stats.meanNat) ? `2^${stats.meanNat.toFixed(2)}` : stats.noEvalAtStep ? "no ckpt" : "never 0.25",
          T.nat
        );
        cPerm.set(
          isNum(stats.meanPerm) ? fmtInt(Math.pow(2, stats.meanPerm)) : "—",
          isNum(stats.meanPerm) ? `2^${stats.meanPerm.toFixed(2)}` : stats.noEvalAtStep ? "no ckpt" : "never 0.25",
          T.perm
        );
        cDelta.set(
          fmtSigned(stats.meanDelta),
          isNum(stats.meanDelta) ? "octaves" : "—"
        );
        cFourth.set(
          isNum(stats.maxAbsDelta) ? stats.maxAbsDelta.toFixed(2) : "—",
          "all budgets"
        );
        cVerdict.set(
          stats.verdict.text.startsWith("PASS") ? "PASS" : stats.verdict.text,
          "|\u0394_MLP| \u2264 0.5",
          stats.verdict.cls === "viz-badge--nat" ? T.nat : T.muted
        );
      }

      strip.appendChild(cNat.card);
      strip.appendChild(cPerm.card);
      strip.appendChild(cDelta.card);
      strip.appendChild(cFourth.card);
      strip.appendChild(cVerdict.card);
      panel.appendChild(strip);

      return panel;
    }

    function showPointTooltip(ev, dsName, isUnet, step, n, hoveredRep, hoveredArm, entries, stats, withPermSeeds) {
      const T = K.theme();
      const modelName = isUnet ? "U-Net" : "MLP Null";
      const rows = [];

      // Hovered arm/rep first
      const hEntry = entries.find((e) => e.rep.key === hoveredRep.key);
      if (hEntry) {
        const mNat = hEntry.row.m_nat?.[String(n)];
        const mPerm = hEntry.row.m_perm?.[String(n)];
        rows.push(
          K.tipRow(
            `${hoveredRep.short} ${hoveredArm === "nat" ? "Natural" : "Permuted"}`,
            isNum(hoveredArm === "nat" ? mNat : mPerm)
              ? (hoveredArm === "nat" ? mNat : mPerm).toFixed(3)
              : "—",
            hoveredArm === "nat" ? T.nat : T.perm
          )
        );
        rows.push(
          K.tipRow(
            `${hoveredRep.short} ${hoveredArm === "nat" ? "Permuted" : "Natural"}`,
            isNum(hoveredArm === "nat" ? mPerm : mNat)
              ? (hoveredArm === "nat" ? mPerm : mNat).toFixed(3)
              : "—",
            hoveredArm === "nat" ? T.perm : T.nat
          )
        );
      }

      // Other replicates summary at same n
      for (const e of entries) {
        if (e.rep.key === hoveredRep.key) continue;
        const mn = e.row.m_nat?.[String(n)];
        const mp = e.row.m_perm?.[String(n)];
        if (isNum(mn) || isNum(mp)) {
          rows.push(
            K.tipRow(
              `${e.rep.short} (nat / perm)`,
              `${isNum(mn) ? mn.toFixed(3) : "—"} / ${isNum(mp) ? mp.toFixed(3) : "—"}`
            )
          );
        }
      }

      if (withPermSeeds && data.permctl) {
        const s1 = data.permctl["1"]?.m_perm_100k?.[String(n)];
        const s2 = data.permctl["2"]?.m_perm_100k?.[String(n)];
        if (isNum(s1) || isNum(s2)) {
          rows.push(
            K.tipRow(
              "Perm seeds 1 / 2",
              `${isNum(s1) ? s1.toFixed(3) : "—"} / ${isNum(s2) ? s2.toFixed(3) : "—"}`,
              T.gold
            )
          );
        }
      }

      if (isNum(stats.meanDelta)) {
        rows.push(K.tipRow("Mean \u0394 (across reps)", fmtSigned(stats.meanDelta)));
      }

      K.tooltip.show(
        ev.clientX,
        ev.clientY,
        `<div class="viz-tooltip__title">${dsName} \u00b7 ${modelName} \u00b7 n = ${fmtInt(n)} @ ${fmtStepShort(step)}</div>${rows.join("")}`
      );
    }

    function redraw() {
      K.clear(unetRow);
      K.clear(mlpRow);
      buildLegend();

      const totalW = Math.max(300, plotsWrap.clientWidth || container.clientWidth || 860);
      const isTwoCol = state.dataset === "both" && totalW >= 620;
      const panelW = isTwoCol ? (totalW - 18) / 2 : totalW;
      const datasets = state.dataset === "both" ? ["mnist", "fmnist"] : [state.dataset];

      const showUnet = state.model === "both" || state.model === "unet";
      const showMlp = state.model === "both" || state.model === "mlp";

      unetRow.style.display = showUnet ? "block" : "none";
      mlpRow.style.display = showMlp ? "block" : "none";
      mlpRow.style.marginTop = showUnet && showMlp ? "16px" : "0";

      if (showUnet) {
        const grid = K.el("div", { class: state.dataset === "both" ? "viz-grid-2" : "" });
        for (const ds of datasets) {
          grid.appendChild(renderCapacityPanel(ds, "unet", panelW));
        }
        unetRow.appendChild(grid);
      }

      if (showMlp) {
        const grid = K.el("div", { class: state.dataset === "both" ? "viz-grid-2" : "" });
        for (const ds of datasets) {
          grid.appendChild(renderCapacityPanel(ds, "mlp", panelW));
        }
        mlpRow.appendChild(grid);
      }
    }

    updateControlsUI();
    redraw();
    K.mountResponsive(plotsWrap, redraw);
  }

  /* ============================================================================
   * WIDGET 4: Clock Speed & Data Collapse (mountSpeed)
   * ============================================================================ */
  function mountSpeed(container, data) {
    K.clear(container);

    const state = {
      dataset: "mnist", // "mnist" | "fmnist"
      n: 2048, // 512 | 1024 | 2048 | 4096
      threshold: "1_3", // "1_3" | "1_2"
      rhoShift: 0.0,
    };

    // ---- Header
    const header = K.el("div", { class: "viz-header" }, [
      K.el("div", {}, [
        K.el("h3", {
          class: "viz-title",
          text: "Clock Speed: Permuted Trajectories Follow the Same Curve, Shifted in Log-Time",
        }),
        K.el("p", {
          class: "viz-subtitle",
          text: "Drag the time-shift slider \u03c1 to translate the permuted curve left by 2^\u03c1 \u2014 watching whether the two arms collapse onto one universal climb.",
        }),
      ]),
    ]);

    // ---- Controls (created once!)
    const dsSeg = K.makeSeg(
      [
        { value: "mnist", label: "MNIST" },
        { value: "fmnist", label: "FashionMNIST" },
      ],
      state.dataset,
      (v) => {
        state.dataset = v;
        syncSliderBoundsAndButtons();
        redraw();
      }
    );

    const nSeg = K.makeSeg(
      SPEED_NS.map((nv) => ({ value: String(nv), label: `n = ${nv}` })),
      String(state.n),
      (v) => {
        state.n = Number(v);
        syncSliderBoundsAndButtons();
        redraw();
      }
    );

    const thrSeg = K.makeSeg(
      [
        { value: "1_3", label: "d\u2081/d\u2082 < 1/3" },
        { value: "1_2", label: "d\u2081/d\u2082 < 1/2" },
      ],
      state.threshold,
      (v) => {
        state.threshold = v;
        syncSliderBoundsAndButtons();
        redraw();
      }
    );

    const shiftReadout = K.el("span", {
      class: "viz-readout",
      text: "\u03c1 = 0.00 (1.00\u00d7)",
    });

    const rhoSlider = K.el("input", {
      type: "range",
      class: "viz-slider",
      min: "0.00",
      max: "1.50",
      step: "0.01",
      value: "0.00",
      "aria-label": "Shift permuted curve left by rho octaves",
      onInput: (e) => {
        state.rhoShift = Number(e.target.value);
        syncSliderReadout();
        redraw();
      },
    });

    const snapBtn = K.el("button", {
      type: "button",
      class: "viz-btn viz-btn--primary",
      text: "Snap to measured \u03c1",
      onClick: () => {
        const info = getMeasuredRhoInfo();
        if (info.snapValue === null) return;
        state.rhoShift = Number(info.snapValue.toFixed(2));
        rhoSlider.value = state.rhoShift.toFixed(2);
        syncSliderReadout();
        redraw();
      },
    });

    const resetBtn = K.el("button", {
      type: "button",
      class: "viz-btn",
      text: "Reset (0.00)",
      onClick: () => {
        state.rhoShift = 0.0;
        rhoSlider.value = "0.00";
        syncSliderReadout();
        redraw();
      },
    });

    const controls = K.el("div", { class: "viz-controls" }, [
      K.el("div", { class: "viz-control-group" }, [
        K.el("span", { class: "viz-control-label", text: "Dataset" }),
        dsSeg,
      ]),
      K.el("div", { class: "viz-control-group" }, [
        K.el("span", { class: "viz-control-label", text: "Size" }),
        nSeg,
      ]),
      K.el("div", { class: "viz-control-group" }, [
        K.el("span", { class: "viz-control-label", text: "Threshold" }),
        thrSeg,
      ]),
      K.el(
        "div",
        {
          class: "viz-slider-wrap",
          style: "flex: 1 1 360px; min-width: 260px;",
        },
        [
          K.el("span", {
            class: "viz-control-label",
            text: "Shift permuted curve left by \u03c1 octaves",
          }),
          rhoSlider,
          shiftReadout,
        ]
      ),
      K.el("div", { class: "viz-control-group" }, [snapBtn, resetBtn]),
    ]);

    // ---- Plot Panel & Stat Strip
    const panelTitleLeft = K.el("span", { text: "Memorisation trajectories in log-step space" });
    const panelBadge = K.el("span", { class: "viz-badge", text: "Unshifted (1.00\u00d7)" });
    const svgWrap = K.el("div");
    const statStrip = K.el("div", { class: "viz-stat-strip" });
    const statTauNat = makeStatCard("\u03c4_nat (mean)");
    const statTauPerm = makeStatCard("\u03c4_perm (mean)");
    const statMeasRho = makeStatCard("Measured \u03c1");
    const statShift = makeStatCard("Current shift 2^\u03c1 \u00d7");
    const statRmse = makeStatCard("Collapse Residual RMSE");
    statStrip.appendChild(statTauNat.card);
    statStrip.appendChild(statTauPerm.card);
    statStrip.appendChild(statMeasRho.card);
    statStrip.appendChild(statShift.card);
    statStrip.appendChild(statRmse.card);

    const plotPanel = K.el("div", { class: "viz-panel" }, [
      K.el("div", { class: "viz-panel-title" }, [panelTitleLeft, panelBadge]),
      svgWrap,
      statStrip,
    ]);

    const legendWrap = K.el("div", { class: "viz-legend" });
    const figcaption = K.el("div", { class: "viz-figcaption" }, [
      K.el("b", { text: "How to read: " }),
      "Each permuted trajectory point (step, m) is plotted at (step / 2^\u03c1, m). ",
      "When \u03c1 matches the measured log-time slowdown, the permuted climb overlaps the natural climb with minimal Collapse Residual RMSE. ",
      "On MNIST at n = 2,048, the 300k exp06 run shows the permuted arm catching up to the natural plateau (m \u2248 0.83 vs 0.82 at 300k steps).",
    ]);

    container.appendChild(header);
    container.appendChild(controls);
    container.appendChild(plotPanel);
    container.appendChild(legendWrap);
    container.appendChild(figcaption);

    function getTrajBundle() {
      const trajsKey = state.threshold === "1_3" ? "trajs_1_3" : "trajs_1_2";
      const nStr = String(state.n);
      const repTrajs = REPS.map((rep) => ({
        rep,
        traj: data[state.dataset]?.unet?.[rep.key]?.[trajsKey]?.[nStr] || null,
      })).filter((x) => x.traj && x.traj.nat && x.traj.perm);

      let exp06 = null;
      if (state.dataset === "mnist" && (state.n === 2048 || state.n === 4096)) {
        const eKey = state.threshold === "1_3" ? "budget_300k_trajs_1_3" : "budget_300k_trajs_1_2";
        exp06 = data.mnist?.[eKey]?.[nStr] || null;
      }
      return { repTrajs, exp06 };
    }

    function getMeasuredRhoInfo() {
      const rhoKey = state.threshold === "1_3" ? "rho_1_3" : "rho_1_2";
      const rows = REPS.map((rep) => {
        const arr = data[state.dataset]?.unet?.[rep.key]?.[rhoKey] || [];
        return arr.find((r) => r.n === state.n) || null;
      }).filter(Boolean);

      const exactRhos = rows.map((r) => r.rho).filter(isNum);
      const lbRhos = rows.map((r) => r.rho_lb).filter(isNum);
      const tauNats = rows.map((r) => r.tau_nat);
      const tauPerms = rows.map((r) => r.tau_perm);

      let measuredRho = null;
      let isLB = false;
      if (exactRhos.length === rows.length && exactRhos.length > 0) {
        measuredRho = mean(exactRhos);
      } else if (lbRhos.length > 0) {
        measuredRho = mean(lbRhos);
        isLB = true;
      }

      // Check exp06 300k tau info on MNIST 2048/4096 when 100k is censored
      let exp06RhoRow = null;
      if (state.dataset === "mnist" && (state.n === 2048 || state.n === 4096) && state.threshold === "1_3") {
        exp06RhoRow = data.mnist?.budget_300k_rho?.find((r) => r.n === state.n) || null;
      }

      const snapValue = isNum(measuredRho) ? measuredRho : null;
      return { rows, measuredRho, isLB, snapValue, tauNats, tauPerms, exp06RhoRow };
    }

    function syncSliderReadout() {
      const mult = Math.pow(2, state.rhoShift);
      shiftReadout.textContent = `\u03c1 = ${state.rhoShift.toFixed(2)} (${mult.toFixed(2)}\u00d7)`;
      resetBtn.disabled = Math.abs(state.rhoShift) < 0.005;
      resetBtn.style.opacity = resetBtn.disabled ? "0.55" : "1";
    }

    function syncSliderBoundsAndButtons() {
      const info = getMeasuredRhoInfo();
      const neededMax = info.snapValue !== null && info.snapValue > 1.5 ? 1.8 : 1.5;
      rhoSlider.max = neededMax.toFixed(2);
      if (state.rhoShift > neededMax) {
        state.rhoShift = neededMax;
        rhoSlider.value = state.rhoShift.toFixed(2);
      }
      if (info.snapValue === null) {
        snapBtn.disabled = true;
        snapBtn.textContent = "Snap (\u03c1 censored)";
        snapBtn.style.opacity = "0.55";
      } else {
        snapBtn.disabled = false;
        snapBtn.textContent = info.isLB
          ? `Snap to \u03c1_lb (${info.snapValue.toFixed(2)})`
          : `Snap to measured \u03c1 (${info.snapValue.toFixed(2)})`;
        snapBtn.style.opacity = "1";
      }
      syncSliderReadout();
    }

    function buildSpeedLegend(hasExp06) {
      const T = K.theme();
      K.clear(legendWrap);
      const items = [
        { label: "Natural arm", swatch: { lineColor: T.nat } },
        {
          label:
            state.rhoShift > 0.02
              ? `Permuted shifted (step / 2^${state.rhoShift.toFixed(2)})`
              : "Permuted arm",
          swatch: { lineColor: T.perm, dash: "5,3" },
        },
      ];
      if (state.rhoShift > 0.02) {
        items.push({
          label: "Unshifted permuted ghost (rep0)",
          swatch: { lineColor: T.perm, dash: "2,3" },
          opacity: 0.45,
        });
      }
      items.push(
        { label: "rep0", swatch: { shape: "circle", fill: T.muted, stroke: T.surface, width: 14 } },
        { label: "repA", swatch: { shape: "square", fill: T.muted, stroke: T.surface, width: 14 } },
        { label: "repB", swatch: { shape: "triangle", fill: T.muted, stroke: T.surface, width: 14 } }
      );
      if (hasExp06) {
        items.push({
          label: "rep0 (300k ext., exp06)",
          swatch: { lineColor: T.text, shape: "diamond", fill: T.muted, stroke: T.surface, width: 18 },
        });
      }
      for (const it of items) {
        const node = K.el("span", { class: "viz-legend__item" }, [
          legendSvgSwatch(it.swatch),
          K.el("span", { text: it.label }),
        ]);
        if (it.opacity) node.style.opacity = String(it.opacity);
        legendWrap.appendChild(node);
      }
    }

    function redraw() {
      const T = K.theme();
      K.clear(svgWrap);

      const { repTrajs, exp06 } = getTrajBundle();
      const info = getMeasuredRhoInfo();
      const factor = Math.pow(2, state.rhoShift);
      const dsName = state.dataset === "mnist" ? "MNIST" : "FashionMNIST";
      const thrLabel = state.threshold === "1_3" ? "d\u2081/d\u2082 < 1/3" : "d\u2081/d\u2082 < 1/2";

      panelTitleLeft.textContent = `${dsName} \u00b7 n = ${fmtInt(state.n)} \u00b7 ${thrLabel}`;

      // Compute live Collapse Residual RMSE on rep0
      const rep0Entry = repTrajs.find((r) => r.rep.key === "rep0");
      const rmse = rep0Entry
        ? computeCollapseRMSE(rep0Entry.traj.nat, rep0Entry.traj.perm, state.rhoShift)
        : null;
      const rmseBase = rep0Entry
        ? computeCollapseRMSE(rep0Entry.traj.nat, rep0Entry.traj.perm, 0)
        : null;

      // Update header badge
      if (state.rhoShift <= 0.02) {
        panelBadge.className = "viz-badge";
        panelBadge.textContent = "Unshifted (1.00\u00d7)";
      } else if (info.measuredRho !== null && Math.abs(state.rhoShift - info.measuredRho) <= 0.06) {
        panelBadge.className = "viz-badge viz-badge--nat";
        panelBadge.textContent = info.isLB
          ? `Aligned near lower bound (${factor.toFixed(2)}\u00d7)`
          : `Collapsed at \u03c1 \u2248 ${state.rhoShift.toFixed(2)} (${factor.toFixed(2)}\u00d7)`;
      } else {
        panelBadge.className = "viz-badge viz-badge--perm";
        panelBadge.textContent = `Shifted \u2212${state.rhoShift.toFixed(2)} oct (${factor.toFixed(2)}\u00d7)`;
      }

      // Update stat strip
      const exp06Note = info.exp06RhoRow && info.tauNats.every((v) => typeof v === "string") ? " (>300k ext.)" : "";
      statTauNat.set(
        fmtTau(info.tauNats),
        exp06Note ? "exp06: > 300k" : "m = 0.5 crossing",
        T.nat
      );
      statTauPerm.set(
        fmtTau(info.tauPerms),
        exp06Note ? "exp06: > 300k" : "unshifted perm",
        T.perm
      );
      if (info.measuredRho !== null) {
        statMeasRho.set(
          `${info.isLB ? "> " : ""}${info.measuredRho.toFixed(2)} oct`,
          info.isLB ? `> ${Math.pow(2, info.measuredRho).toFixed(2)}\u00d7 (300k LB)` : `${Math.pow(2, info.measuredRho).toFixed(2)}\u00d7 slower`
        );
      } else {
        statMeasRho.set("Censored", "\u03c4 > budget");
      }
      statShift.set(
        `${factor.toFixed(2)}\u00d7`,
        `2^${state.rhoShift.toFixed(2)}`
      );
      if (isNum(rmse)) {
        const goodCollapse = rmse <= 0.045;
        const pctDrop = isNum(rmseBase) && rmseBase > 0 ? Math.round(((rmseBase - rmse) / rmseBase) * 100) : null;
        statRmse.set(
          rmse.toFixed(3),
          state.rhoShift > 0.02 && pctDrop !== null
            ? `${pctDrop >= 0 ? "\u2212" + pctDrop : "+" + Math.abs(pctDrop)}% vs \u03c1=0`
            : "rep0 overlap window",
          goodCollapse ? T.nat : ""
        );
      } else {
        statRmse.set("—", "");
      }

      buildSpeedLegend(Boolean(exp06));

      // ---- Build SVG
      const w = Math.max(300, Math.floor(svgWrap.clientWidth || container.clientWidth - 44 || 860));
      const h = w < 560 ? 280 : 340;
      const m = { l: 54, r: 22, t: 24, b: 40 };
      const pw = w - m.l - m.r;
      const ph = h - m.t - m.b;

      const xScale = K.scaleLog([1500, 350000], [m.l, m.l + pw], 2);
      const yScale = K.scaleLinear([0, 1], [m.t + ph, m.t]);
      const xTicks =
        w < 540
          ? [2000, 5000, 10000, 30000, 100000, 300000]
          : [2000, 5000, 10000, 20000, 50000, 100000, 200000, 300000];

      const svg = K.makeSvg(w, h);

      // Clip path so shifted curves never poke left of y-axis
      const clipId = `clip-speed-${Math.random().toString(36).slice(2, 9)}`;
      const defs = K.svgEl("defs");
      const clipPath = K.svgEl("clipPath", { id: clipId });
      clipPath.appendChild(
        K.svgEl("rect", { x: m.l, y: m.t - 8, width: pw + 8, height: ph + 16 })
      );
      defs.appendChild(clipPath);
      svg.appendChild(defs);

      K.drawAxes(svg, {
        w,
        h,
        m,
        xScale,
        yScale,
        xTicks,
        yTicks: [0, 0.25, 0.5, 0.75, 1.0],
        xFormat: fmtStepShort,
        yFormat: (v) => v.toFixed(2).replace(/\.00$/, ".0"),
        xLabel: "Training steps (log scale)",
        yLabel: "Memorised fraction m(step)",
        hLines: [
          {
            y: 0.5,
            color: T.muted,
            dash: "3,3",
            width: 1.2,
            label: "m = 0.5",
            align: "left",
          },
        ],
        vLines: [
          {
            x: 100000,
            color: T.muted,
            dash: "2,4",
            width: 1,
            opacity: 0.45,
            label: "100k",
            dx: 4,
            dy: 12,
          },
        ],
      });

      const plotGroup = K.svgEl("g", { "clip-path": `url(#${clipId})` });
      svg.appendChild(plotGroup);

      // Draw vertical tau indicators if numeric across reps
      const geomTauNat = geomMean(info.tauNats);
      const geomTauPerm = geomMean(info.tauPerms);
      if (isNum(geomTauNat) && geomTauNat >= 1500 && geomTauNat <= 350000) {
        const xn = xScale(geomTauNat);
        plotGroup.appendChild(
          K.svgEl("line", {
            x1: xn,
            x2: xn,
            y1: yScale(0.5),
            y2: yScale(0),
            stroke: T.nat,
            "stroke-width": 1.3,
            "stroke-dasharray": "3,3",
            opacity: 0.65,
          })
        );
        plotGroup.appendChild(
          K.svgEl("text", {
            x: xn - 5,
            y: yScale(0.03),
            "text-anchor": "end",
            fill: T.nat,
            "font-family": T.mono,
            "font-size": 10.5,
            "font-weight": 600,
            style: `paint-order:stroke;stroke:${T.surface};stroke-width:3px;`,
            text: `\u03c4_nat=${fmtStepStat(geomTauNat)}`,
          })
        );
      }

      if (isNum(geomTauPerm)) {
        const shiftedTau = geomTauPerm / factor;
        if (shiftedTau >= 1500 && shiftedTau <= 350000) {
          const xp = xScale(shiftedTau);
          plotGroup.appendChild(
            K.svgEl("line", {
              x1: xp,
              x2: xp,
              y1: yScale(0.5),
              y2: yScale(0),
              stroke: T.perm,
              "stroke-width": 1.3,
              "stroke-dasharray": "3,3",
              opacity: 0.7,
            })
          );
          plotGroup.appendChild(
            K.svgEl("text", {
              x: xp + 5,
              y: yScale(0.03),
              "text-anchor": "start",
              fill: T.perm,
              "font-family": T.mono,
              "font-size": 10.5,
              "font-weight": 600,
              style: `paint-order:stroke;stroke:${T.surface};stroke-width:3px;`,
              text:
                state.rhoShift > 0.02
                  ? `\u03c4_perm/2^\u03c1=${fmtStepStat(shiftedTau)}`
                  : `\u03c4_perm=${fmtStepStat(geomTauPerm)}`,
            })
          );
          if (state.rhoShift > 0.02 && geomTauPerm <= 350000) {
            const xOrig = xScale(geomTauPerm);
            const yMid = yScale(0.5);
            if (xOrig - xp > 12) {
              plotGroup.appendChild(
                K.svgEl("line", {
                  x1: xOrig,
                  x2: xp + 4,
                  y1: yMid,
                  y2: yMid,
                  stroke: T.perm,
                  "stroke-width": 1.8,
                  opacity: 0.65,
                })
              );
              plotGroup.appendChild(
                K.svgEl("polygon", {
                  points: `${xp},${yMid} ${xp + 6},${yMid - 3.5} ${xp + 6},${yMid + 3.5}`,
                  fill: T.perm,
                  opacity: 0.75,
                })
              );
            }
          }
        }
      }

      // Faint unshifted ghost outline of permuted rep0 (and exp06 if present) when rhoShift > 0.02
      if (state.rhoShift > 0.02) {
        if (rep0Entry) {
          const ghostPts = rep0Entry.traj.perm.map(([s, val]) => [xScale(s), yScale(val)]);
          plotGroup.appendChild(
            K.svgEl("path", {
              d: K.linePath(ghostPts),
              fill: "none",
              stroke: T.perm,
              "stroke-width": 2,
              "stroke-dasharray": "3,3",
              opacity: 0.22,
            })
          );
        }
        if (exp06 && exp06.perm) {
          const ghostExtPts = exp06.perm.map(([s, val]) => [xScale(s), yScale(val)]);
          plotGroup.appendChild(
            K.svgEl("path", {
              d: K.linePath(ghostExtPts),
              fill: "none",
              stroke: T.perm,
              "stroke-width": 2,
              "stroke-dasharray": "3,3",
              opacity: 0.22,
            })
          );
        }
      }

      // Draw all replicate curves (Natural solid, Permuted dashed shifted left by 2^rhoShift)
      for (const { rep, traj } of repTrajs) {
        const natPts = traj.nat.map(([s, val]) => [xScale(s), yScale(val)]);
        const permPts = traj.perm.map(([s, val]) => [xScale(s / factor), yScale(val)]);

        plotGroup.appendChild(
          K.svgEl("path", {
            d: K.linePath(natPts),
            fill: "none",
            stroke: T.nat,
            "stroke-width": 1.9,
            opacity: 0.85,
          })
        );
        plotGroup.appendChild(
          K.svgEl("path", {
            d: K.linePath(permPts),
            fill: "none",
            stroke: T.perm,
            "stroke-width": 1.9,
            "stroke-dasharray": "5,3",
            opacity: 0.88,
          })
        );

        natPts.forEach(([cx, cy]) => {
          plotGroup.appendChild(K.drawMarker(svg, cx, cy, rep.shape, 3.2, T.nat, T.surface));
        });
        permPts.forEach(([cx, cy]) => {
          plotGroup.appendChild(K.drawMarker(svg, cx, cy, rep.shape, 3.2, T.perm, T.surface));
        });
      }

      // Draw MNIST exp06 300k extension for n = 2048 & n = 4096
      if (exp06) {
        const extNatPts = exp06.nat.map(([s, val]) => [xScale(s), yScale(val)]);
        const extPermPts = exp06.perm.map(([s, val]) => [xScale(s / factor), yScale(val)]);

        plotGroup.appendChild(
          K.svgEl("path", {
            d: K.linePath(extNatPts),
            fill: "none",
            stroke: T.nat,
            "stroke-width": 2.3,
            opacity: 0.96,
          })
        );
        plotGroup.appendChild(
          K.svgEl("path", {
            d: K.linePath(extPermPts),
            fill: "none",
            stroke: T.perm,
            "stroke-width": 2.3,
            "stroke-dasharray": "6,3",
            opacity: 0.96,
          })
        );

        extNatPts.forEach(([cx, cy], idx) => {
          if (exp06.nat[idx][0] >= 100000) {
            plotGroup.appendChild(K.drawMarker(svg, cx, cy, "diamond", 3.8, T.nat, T.surface));
          }
        });
        extPermPts.forEach(([cx, cy], idx) => {
          if (exp06.perm[idx][0] >= 100000) {
            plotGroup.appendChild(K.drawMarker(svg, cx, cy, "diamond", 3.8, T.perm, T.surface));
          }
        });
      }

      // ---- Interactive Vertical Hover Crosshair
      const crosshair = K.svgEl("line", {
        x1: 0,
        x2: 0,
        y1: m.t,
        y2: m.t + ph,
        stroke: T.text,
        "stroke-width": 1.1,
        "stroke-dasharray": "2,2",
        opacity: 0,
        style: "pointer-events:none;",
      });
      const dotNat = K.svgEl("circle", {
        r: 4.5,
        fill: T.nat,
        stroke: T.surface,
        "stroke-width": 1.5,
        opacity: 0,
        style: "pointer-events:none;",
      });
      const dotPerm = K.svgEl("circle", {
        r: 4.5,
        fill: T.perm,
        stroke: T.surface,
        "stroke-width": 1.5,
        opacity: 0,
        style: "pointer-events:none;",
      });
      const dotGhost = K.svgEl("circle", {
        r: 3.8,
        fill: T.surface,
        stroke: T.perm,
        "stroke-width": 1.5,
        opacity: 0,
        style: "pointer-events:none;",
      });

      svg.appendChild(crosshair);
      svg.appendChild(dotGhost);
      svg.appendChild(dotNat);
      svg.appendChild(dotPerm);

      // Prefer exp06 when cursor goes > 100k on MNIST 2048/4096, else rep0
      const hoverTrap = K.svgEl("rect", {
        x: m.l,
        y: m.t,
        width: pw,
        height: ph,
        fill: "transparent",
        style: "cursor:crosshair;",
      });

      function onHover(ev) {
        const rect = svg.getBoundingClientRect();
        if (!rect.width) return;
        const svgX = ((ev.clientX - rect.left) / rect.width) * w;
        const clampedX = Math.max(m.l, Math.min(m.l + pw, svgX));
        const cursorStep = xScale.invert(clampedX);

        crosshair.setAttribute("x1", clampedX);
        crosshair.setAttribute("x2", clampedX);
        crosshair.setAttribute("opacity", "0.6");

        // Primary reference series: rep0 (or exp06 when cursor extends beyond rep0 max step)
        const rep0NatMax = rep0Entry ? rep0Entry.traj.nat[rep0Entry.traj.nat.length - 1][0] : 100000;
        const rep0PermMax = rep0Entry ? rep0Entry.traj.perm[rep0Entry.traj.perm.length - 1][0] : 100000;
        const useExp06Nat = Boolean(exp06 && cursorStep > rep0NatMax);
        const origPermStep = cursorStep * factor;
        const useExp06Perm = Boolean(exp06 && origPermStep > rep0PermMax);
        const useExp06Ghost = Boolean(exp06 && cursorStep > rep0PermMax);

        const refNatArr = useExp06Nat ? exp06.nat : rep0Entry?.traj.nat;
        const refPermArr = useExp06Perm ? exp06.perm : rep0Entry?.traj.perm;
        const refGhostArr = useExp06Ghost ? exp06.perm : rep0Entry?.traj.perm;

        const mNat = interpLogStep(refNatArr, cursorStep);
        const mPermShifted = interpLogStep(refPermArr, origPermStep);
        const mPermUnshifted = interpLogStep(refGhostArr, cursorStep);

        if (isNum(mNat)) {
          dotNat.setAttribute("cx", clampedX);
          dotNat.setAttribute("cy", yScale(mNat));
          dotNat.setAttribute("opacity", "1");
        } else {
          dotNat.setAttribute("opacity", "0");
        }

        if (isNum(mPermShifted)) {
          dotPerm.setAttribute("cx", clampedX);
          dotPerm.setAttribute("cy", yScale(mPermShifted));
          dotPerm.setAttribute("opacity", "1");
        } else {
          dotPerm.setAttribute("opacity", "0");
        }

        if (state.rhoShift > 0.02 && isNum(mPermUnshifted)) {
          dotGhost.setAttribute("cx", clampedX);
          dotGhost.setAttribute("cy", yScale(mPermUnshifted));
          dotGhost.setAttribute("opacity", "0.75");
        } else {
          dotGhost.setAttribute("opacity", "0");
        }

        const natTag = useExp06Nat ? "exp06" : "rep0";
        const permTag = useExp06Perm ? "exp06" : "rep0";
        const rows = [
          K.tipRow(`Natural (${natTag})`, isNum(mNat) ? mNat.toFixed(3) : "—", T.nat),
        ];
        if (state.rhoShift > 0.02) {
          rows.push(
            K.tipRow(
              `Perm shifted (${permTag} @ ${fmtStepStat(origPermStep)})`,
              isNum(mPermShifted) ? mPermShifted.toFixed(3) : "—",
              T.perm
            )
          );
          rows.push(
            K.tipRow(
              `Perm unshifted (${useExp06Ghost ? "exp06" : "rep0"})`,
              isNum(mPermUnshifted) ? mPermUnshifted.toFixed(3) : "—",
              T.muted
            )
          );
          if (isNum(mNat) && isNum(mPermShifted)) {
            rows.push(
              K.tipRow("Shifted residual |\u0394m|", Math.abs(mNat - mPermShifted).toFixed(3))
            );
          }
        } else {
          rows.push(
            K.tipRow(
              `Permuted (${useExp06Ghost ? "exp06" : "rep0"})`,
              isNum(mPermUnshifted) ? mPermUnshifted.toFixed(3) : "—",
              T.perm
            )
          );
        }

        if (exp06 && !useExp06Nat && !useExp06Perm) {
          const en = interpLogStep(exp06.nat, cursorStep);
          const ep = interpLogStep(exp06.perm, origPermStep);
          if (isNum(en) || isNum(ep)) {
            rows.push(
              K.tipRow(
                "exp06 300k (nat / perm)",
                `${isNum(en) ? en.toFixed(3) : "—"} / ${isNum(ep) ? ep.toFixed(3) : "—"}`
              )
            );
          }
        }

        K.tooltip.show(
          ev.clientX,
          ev.clientY,
          `<div class="viz-tooltip__title">${dsName} \u00b7 n = ${fmtInt(state.n)} \u00b7 step \u2248 ${fmtInt(cursorStep)}</div>${rows.join("")}`
        );
      }

      hoverTrap.addEventListener("mousemove", onHover);
      hoverTrap.addEventListener("mouseenter", onHover);
      hoverTrap.addEventListener("mouseleave", () => {
        crosshair.setAttribute("opacity", "0");
        dotNat.setAttribute("opacity", "0");
        dotPerm.setAttribute("opacity", "0");
        dotGhost.setAttribute("opacity", "0");
        K.tooltip.hide();
      });

      svg.appendChild(hoverTrap);
      svgWrap.appendChild(svg);
    }

    syncSliderBoundsAndButtons();
    redraw();
    K.mountResponsive(svgWrap, redraw);
  }

  window.DiffMemWidgets.mountCapacity = mountCapacity;
  window.DiffMemWidgets.mountSpeed = mountSpeed;
})();
