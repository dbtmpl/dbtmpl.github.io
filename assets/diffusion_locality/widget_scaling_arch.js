/* Widgets 5 & 6: Delta-vs-Budget, Speedup rho(n), Danger Zone MSE, and Exploratory Scale-Out */
(function () {
  "use strict";

  const K = window.DiffMemKit;
  if (!K) return;

  const { el, svgEl, clear, scaleLinear, scaleLog, makeSvg, drawAxes, linePath, drawMarker, tooltip, tipRow, makeSeg, mountResponsive, fmtSteps } = K;

  function esc(str) {
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function fmtSigned(v, digits = 2) {
    if (v === null || v === undefined || Number.isNaN(v)) return "—";
    const s = Number(v).toFixed(digits);
    return Number(v) > 0 ? `+${s}` : s;
  }

  function fmtNum(v, digits = 2) {
    if (v === null || v === undefined || Number.isNaN(v)) return "—";
    return Number(v).toFixed(digits);
  }

  function buildLegend(items) {
    const wrap = el("div", { class: "viz-legend" });
    for (const it of items) {
      const sw = el("span", {
        class: "viz-legend__swatch",
        style: it.dashed
          ? `background:repeating-linear-gradient(90deg,${it.color} 0 5px,transparent 5px 8px);height:2.5px;`
          : it.dotted
            ? `background:repeating-linear-gradient(90deg,${it.color} 0 2px,transparent 2px 5px);height:2.5px;`
            : `background:${it.color};`,
      });
      wrap.appendChild(el("span", { class: "viz-legend__item" }, [sw, it.label]));
    }
    return wrap;
  }

  function addHoverLayer(svg) {
    const g = svgEl("g", { class: "viz-hover-layer", style: "pointer-events:none;" });
    svg.appendChild(g);
    return g;
  }

  function addTrapRect(svg, m, pw, ph, onMove, onLeave) {
    const trap = svgEl("rect", {
      x: m.l,
      y: m.t,
      width: pw,
      height: ph,
      fill: "transparent",
      style: "cursor:crosshair;",
    });
    trap.addEventListener("mousemove", onMove);
    trap.addEventListener("mouseleave", onLeave);
    svg.appendChild(trap);
    return trap;
  }

  function renderStatStrip(target, entries) {
    clear(target);
    for (const e of entries) {
      const vEl = el("div", {
        class: "viz-stat__v",
        text: e.value,
        style: e.color ? `color:${e.color}` : "",
      });
      target.appendChild(el("div", { class: "viz-stat" }, [
        el("div", { class: "viz-stat__k", text: e.label }),
        vEl,
      ]));
    }
  }

  function drawOpenMarker(cx, cy, r, stroke, surfaceFill) {
    return svgEl("circle", {
      cx,
      cy,
      r,
      fill: surfaceFill,
      stroke,
      "stroke-width": 2,
    });
  }

  function drawUpArrow(cx, cy, color, surfaceFill, len = 15) {
    const g = svgEl("g");
    const yTip = cy - len;
    g.appendChild(svgEl("line", {
      x1: cx,
      y1: cy - 3.5,
      x2: cx,
      y2: yTip + 2,
      stroke: surfaceFill,
      "stroke-width": 3.4,
      "stroke-linecap": "round",
    }));
    g.appendChild(svgEl("line", {
      x1: cx,
      y1: cy - 3.5,
      x2: cx,
      y2: yTip + 2,
      stroke: color,
      "stroke-width": 1.8,
      "stroke-linecap": "round",
    }));
    g.appendChild(svgEl("polygon", {
      points: `${cx},${yTip - 1} ${cx - 3.4},${yTip + 4.5} ${cx + 3.4},${yTip + 4.5}`,
      fill: color,
    }));
    return g;
  }

  function drawGuidePath(svg, pts, stroke, dash, width, opacity = 1) {
    if (pts.length < 2) return;
    svg.appendChild(svgEl("path", {
      d: linePath(pts),
      fill: "none",
      stroke,
      "stroke-width": width,
      "stroke-dasharray": dash || null,
      "stroke-linecap": "round",
      "stroke-linejoin": "round",
      opacity,
    }));
  }

  function drawAreaBetween(svg, upperPts, lowerPts, fill, opacity = 0.14) {
    if (upperPts.length < 2 || upperPts.length !== lowerPts.length) return;
    const rev = lowerPts.slice().reverse();
    const d = linePath(upperPts) + " " + rev.map((p) => `L${p[0].toFixed(2)},${p[1].toFixed(2)}`).join(" ") + " Z";
    svg.appendChild(svgEl("path", {
      d,
      fill,
      opacity,
      stroke: "none",
    }));
  }

  /* =========================================================================
   * WIDGET 5: mountDeltaRho(container, data) -> #viz-delta-rho
   * ========================================================================= */
  function mountDeltaRho(container, data) {
    const state = {
      showGuide: true,
      rightTab: "rho",       // "rho" | "danger"
      showPermSeeds: false,
      dzStep: "100000",      // "10000" | "40000" | "100000"
      dzMetric: "gap",       // "gap" | "mse"
    };

    const reps = ["rep0", "repA", "repB"];
    const mnistReps = reps.map((r) => ({ id: r, rows: data.mnist.unet[r].delta_1_3 }));
    const fmnistReps = reps.map((r) => ({ id: r, rows: data.fmnist.unet[r].delta_1_3 }));
    const mnist300k = (data.mnist.budget_300k_delta || []).filter((d) => d.step >= 100000 && d.delta !== null);

    // ---- Build DOM scaffold once ----
    const header = el("div", { class: "viz-header" }, [
      el("div", null, [
        el("h3", { class: "viz-title", text: "The Speed\u2013Capacity Identity \u0394 \u2248 \u2212\u03c1/a and the Per-Noise \u201cDanger Zone\u201d" }),
        el("p", {
          class: "viz-subtitle",
          text: "Tracking how fixed-budget onset shifts shrink as training runs longer, how \u03c1 grows with n, and why permuted denoising MSE stays higher even after memorisation catches up.",
        }),
      ]),
      el("span", { class: "viz-badge", text: "MNIST & FMNIST \u00b7 rep0/A/B + 300k" }),
    ]);

    // Left panel scaffold
    const guideCb = el("input", { type: "checkbox" });
    guideCb.checked = state.showGuide;
    const guideLabel = el("label", { class: "viz-check" }, [
      guideCb,
      el("span", { text: "Show \u2212\u03c1/a scaling guide (a = 1.0 \u2192 1.7)" }),
    ]);
    const leftControls = el("div", { class: "viz-controls", style: "margin-bottom:8px;" }, [guideLabel]);
    const leftPlot = el("div", { style: "min-height:270px;" });
    const leftLegend = el("div");
    const leftPanel = el("div", { class: "viz-panel" }, [
      el("div", { class: "viz-panel-title" }, [
        el("span", { text: "Onset Shift \u0394(step) vs. Training Budget" }),
        el("span", { class: "viz-readout", text: "5k \u2192 300k" }),
      ]),
      leftControls,
      leftPlot,
      leftLegend,
    ]);

    // Right panel scaffold
    const rightSeg = makeSeg(
      [
        { value: "rho", label: "Slowdown \u03c1(n)" },
        { value: "danger", label: "Danger Zone: Per-t MSE Gap" },
      ],
      state.rightTab,
      (val) => {
        state.rightTab = val;
        syncRightTabControls();
        drawRight();
      }
    );

    // Tab A controls
    const permSeedCb = el("input", { type: "checkbox" });
    permSeedCb.checked = state.showPermSeeds;
    const permSeedLabel = el("label", { class: "viz-check" }, [
      permSeedCb,
      el("span", { text: "Show permutation-identity seeds (1234, 1, 2)" }),
    ]);
    const tabAControls = el("div", { class: "viz-controls", style: "margin-bottom:8px;" }, [permSeedLabel]);

    // Tab B controls
    const dzStepSeg = makeSeg(
      [
        { value: "10000", label: "10k" },
        { value: "40000", label: "40k" },
        { value: "100000", label: "100k" },
      ],
      state.dzStep,
      (val) => {
        state.dzStep = val;
        drawRight();
      }
    );
    const dzMetricSeg = makeSeg(
      [
        { value: "gap", label: "Test \u2212 Train Gap" },
        { value: "mse", label: "Train vs Test MSE" },
      ],
      state.dzMetric,
      (val) => {
        state.dzMetric = val;
        drawRight();
      }
    );
    const tabBControls = el("div", { class: "viz-controls", style: "margin-bottom:8px;gap:8px 10px;" }, [
      el("div", { class: "viz-control-group" }, [
        el("span", { class: "viz-control-label", text: "Step" }),
        dzStepSeg,
      ]),
      dzMetricSeg,
    ]);

    const rightPlot = el("div", { style: "min-height:270px;" });
    const rightLegend = el("div");
    const rightPanel = el("div", { class: "viz-panel" }, [
      el("div", { class: "viz-panel-title", style: "flex-wrap:wrap;" }, [
        rightSeg,
      ]),
      tabAControls,
      tabBControls,
      rightPlot,
      rightLegend,
    ]);

    const grid = el("div", { class: "viz-grid-2" }, [leftPanel, rightPanel]);
    const figcaption = el("div", { class: "viz-figcaption" }, [
      el("b", { text: "Takeaway: " }),
      "Because the memorisation boundary steepens as training proceeds (slope ",
      el("em", { text: "a" }),
      " rises from ~1.0 early to ~1.7\u20133.0 late), a persistent ",
      el("strong", { text: "~2\u00d7 slowdown (\u03c1 \u2248 1.0)" }),
      " projects onto a shrinking horizontal onset shift ",
      el("strong", { text: "\u0394 \u2248 \u2212\u03c1/a \u2192 \u22120.33" }),
      " inside the pre-registered \u00b10.5 NO SHIFT band. Yet on Tab B, low-noise permuted test MSE at ",
      el("em", { text: "t = 50" }),
      " stays nearly ",
      el("strong", { text: "2\u00d7 worse (1.64 vs. 0.90 gap at 100k)" }),
      " even after ",
      el("em", { text: "L\u2082" }),
      " memorisation catches up.",
    ]);

    clear(container);
    container.appendChild(header);
    container.appendChild(grid);
    container.appendChild(figcaption);

    function syncRightTabControls() {
      tabAControls.style.display = state.rightTab === "rho" ? "flex" : "none";
      tabBControls.style.display = state.rightTab === "danger" ? "flex" : "none";
    }
    syncRightTabControls();

    guideCb.addEventListener("change", () => {
      state.showGuide = guideCb.checked;
      drawLeft();
    });
    permSeedCb.addEventListener("change", () => {
      state.showPermSeeds = permSeedCb.checked;
      drawRight();
    });

    // ---- Left Panel Draw: Delta(step) vs Budget ----
    function theoryDelta(step) {
      const p = Math.log10(Math.max(5000, step) / 5000) / Math.log10(300000 / 5000);
      const a = 1.0 + 2.0 * Math.pow(Math.max(0, Math.min(1, p)), 0.8);
      return -1.0 / a;
    }

    function drawLeft() {
      const w = leftPlot.clientWidth;
      if (!w) return;
      const T = K.theme();
      const h = 276;
      const m = { l: 48, r: 22, t: 18, b: 38 };
      const pw = w - m.l - m.r;
      const ph = h - m.t - m.b;

      const xScale = scaleLog([4500, 345000], [m.l, m.l + pw], 10);
      const yScale = scaleLinear([-1.2, 0.65], [m.t + ph, m.t]);

      const svg = makeSvg(w, h);
      drawAxes(svg, {
        w,
        h,
        m,
        xScale,
        yScale,
        xTicks: [5000, 10000, 20000, 50000, 100000, 200000, 300000],
        yTicks: [-1.0, -0.5, 0.0, 0.5],
        xFormat: fmtSteps,
        yFormat: (v) => (v === 0 ? "0.0" : fmtSigned(v, 1)),
        xLabel: "Training budget (steps, log scale)",
        yLabel: "\u0394(step) = log\u2082(n*_perm / n*_nat)",
        hBands: [
          { y0: -0.5, y1: 0.5, fill: T.band, label: "Pre-registered NO SHIFT band" },
        ],
        hLines: [
          { y: 0, color: T.axis, width: 1, dash: "none", opacity: 0.9 },
          { y: -1.0, color: T.perm, width: 1.2, dash: "5,4", opacity: 0.85, label: "SHIFT threshold (\u0394 = \u22121.0)", align: "left" },
        ],
      });

      // Theoretical -rho/a scaling guide
      if (state.showGuide) {
        const denseSteps = [];
        for (let i = 0; i <= 40; i++) {
          denseSteps.push(5000 * Math.pow(300000 / 5000, i / 40));
        }
        const upperPts = denseSteps.map((s) => [xScale(s), yScale(theoryDelta(s) * 0.80)]);
        const lowerPts = denseSteps.map((s) => [xScale(s), yScale(theoryDelta(s) * 1.18)]);
        const midPts = denseSteps.map((s) => [xScale(s), yScale(theoryDelta(s))]);
        drawAreaBetween(svg, upperPts, lowerPts, T.gold, 0.14);
        drawGuidePath(svg, midPts, T.gold, "4,3", 1.9, 0.92);
        const guideLabelPt = upperPts[6] || midPts[6];
        svg.appendChild(svgEl("text", {
          x: guideLabelPt[0] + 6,
          y: guideLabelPt[1] - 12,
          "text-anchor": "start",
          fill: T.gold,
          "font-family": T.sans,
          "font-size": 10.5,
          "font-weight": 600,
          style: `paint-order:stroke;stroke:${T.cardAlt};stroke-width:3.5px;stroke-linejoin:round;`,
          text: "\u2212\u03c1/a guide (\u03c1\u22481)",
        }));
      }

      // FashionMNIST rep0/A/B (dashed)
      for (const rep of fmnistReps) {
        const valid = rep.rows.filter((d) => d.step >= 5000 && d.delta !== null);
        const pts = valid.map((d) => [xScale(d.step), yScale(d.delta)]);
        const isMain = rep.id === "rep0";
        drawGuidePath(svg, pts, T.fm, "5,3", isMain ? 2.1 : 1.4, isMain ? 0.95 : 0.52);
        for (const d of valid) {
          svg.appendChild(drawMarker(svg, xScale(d.step), yScale(d.delta), "triangle", isMain ? 3.2 : 2.3, T.fm, T.surface));
        }
      }

      // MNIST rep0/A/B (solid)
      for (const rep of mnistReps) {
        const valid = rep.rows.filter((d) => d.step >= 5000 && d.delta !== null);
        const pts = valid.map((d) => [xScale(d.step), yScale(d.delta)]);
        const isMain = rep.id === "rep0";
        drawGuidePath(svg, pts, T.mn, null, isMain ? 2.2 : 1.45, isMain ? 0.96 : 0.55);
        for (const d of valid) {
          svg.appendChild(drawMarker(svg, xScale(d.step), yScale(d.delta), "circle", isMain ? 3.2 : 2.3, T.mn, T.surface));
        }
      }

      // MNIST exp06 300k seed-0 (100k - 300k, diamond markers)
      if (mnist300k.length) {
        const pts300 = mnist300k.map((d) => [xScale(d.step), yScale(d.delta)]);
        drawGuidePath(svg, pts300, T.mn, null, 2.4, 1.0);
        for (const d of mnist300k) {
          svg.appendChild(drawMarker(svg, xScale(d.step), yScale(d.delta), "diamond", 4.0, T.mn, T.surface));
        }
        const last300 = mnist300k[mnist300k.length - 1];
        svg.appendChild(svgEl("text", {
          x: xScale(last300.step) - 2,
          y: yScale(last300.delta) + 16,
          "text-anchor": "end",
          fill: T.mn,
          "font-family": T.sans,
          "font-size": 10.5,
          "font-weight": 600,
          style: `paint-order:stroke;stroke:${T.cardAlt};stroke-width:3.5px;stroke-linejoin:round;`,
          text: `exp06 300k (${fmtSigned(last300.delta, 2)})`,
        }));
      }

      // Hover interaction
      const allSteps = [10000, 15000, 20000, 30000, 40000, 50000, 70000, 100000, 150000, 200000, 250000, 300000];
      const hoverG = addHoverLayer(svg);

      function lookupStep(rows, step) {
        const found = rows.find((r) => r.step === step);
        return found && found.delta !== null ? found.delta : null;
      }

      addTrapRect(
        svg,
        m,
        pw,
        ph,
        (e) => {
          const rect = svg.getBoundingClientRect();
          const mx = ((e.clientX - rect.left) / rect.width) * w;
          let bestStep = allSteps[0];
          let bestDist = Infinity;
          for (const s of allSteps) {
            const dist = Math.abs(xScale(s) - mx);
            if (dist < bestDist) {
              bestDist = dist;
              bestStep = s;
            }
          }
          clear(hoverG);
          const hx = xScale(bestStep);
          hoverG.appendChild(svgEl("line", {
            x1: hx,
            x2: hx,
            y1: m.t,
            y2: m.t + ph,
            stroke: T.muted,
            "stroke-width": 1,
            "stroke-dasharray": "3,3",
          }));

          const mnVals = mnistReps.map((r) => lookupStep(r.rows, bestStep));
          const fmVals = fmnistReps.map((r) => lookupStep(r.rows, bestStep));
          const d300Row = mnist300k.find((r) => r.step === bestStep);
          const d300Val = d300Row ? d300Row.delta : null;

          const highlight = (val, color) => {
            if (val === null) return;
            hoverG.appendChild(svgEl("circle", {
              cx: hx,
              cy: yScale(val),
              r: 5.5,
              fill: "none",
              stroke: color,
              "stroke-width": 2,
            }));
          };
          mnVals.forEach((v) => highlight(v, T.mn));
          fmVals.forEach((v) => highlight(v, T.fm));
          highlight(d300Val, T.mn);

          const fmtTriple = (arr) => arr.map((v) => (v === null ? "\u2014" : fmtSigned(v, 2))).join(" / ");
          let html = `<div class="viz-tooltip__title">Budget T = ${fmtSteps(bestStep)} steps</div>`;
          html += tipRow("MNIST \u0394 (rep0/A/B)", fmtTriple(mnVals), T.mn);
          html += tipRow("FMNIST \u0394 (rep0/A/B)", fmtTriple(fmVals), T.fm);
          html += tipRow("300k seed-0 \u0394 (exp06)", d300Val !== null ? fmtSigned(d300Val, 3) : "\u2014", T.mn);
          if (state.showGuide) {
            html += tipRow("\u2212\u03c1/a guide (\u03c1=1)", fmtSigned(theoryDelta(bestStep), 2), T.gold);
          }
          tooltip.show(e.clientX, e.clientY, html);
        },
        () => {
          clear(hoverG);
          tooltip.hide();
        }
      );

      // Legend
      clear(leftLegend);
      const legItems = [
        { label: "MNIST U-Net (rep0/A/B)", color: T.mn },
        { label: "MNIST exp06 300k (\u25c6)", color: T.mn },
        { label: "FashionMNIST (rep0/A/B)", color: T.fm, dashed: true },
      ];
      if (state.showGuide) legItems.push({ label: "\u2212\u03c1/a scaling guide", color: T.gold, dashed: true });
      leftLegend.appendChild(buildLegend(legItems));

      clear(leftPlot);
      leftPlot.appendChild(svg);
    }

    // ---- Right Panel Draw: Tab A (Slowdown rho(n)) or Tab B (Danger Zone) ----
    function drawRight() {
      if (state.rightTab === "rho") {
        drawRightRho();
      } else {
        drawRightDanger();
      }
    }

    function drawRightRho() {
      const w = rightPlot.clientWidth;
      if (!w) return;
      const T = K.theme();
      const h = 276;
      const m = { l: 48, r: 22, t: 18, b: 38 };
      const pw = w - m.l - m.r;
      const ph = h - m.t - m.b;

      const xScale = scaleLog([430, 4800], [m.l, m.l + pw], 2);
      const yScale = scaleLinear([-0.2, 1.5], [m.t + ph, m.t]);

      const svg = makeSvg(w, h);
      drawAxes(svg, {
        w,
        h,
        m,
        xScale,
        yScale,
        xTicks: [512, 1024, 2048, 4096],
        yTicks: [-0.2, 0.0, 0.5, 1.0, 1.4],
        xFormat: (v) => String(v),
        yFormat: (v) => (v === 0 ? "0.0" : fmtNum(v, 1)),
        xLabel: "Dataset size n (log\u2082 scale)",
        yLabel: "Slowdown \u03c1 = log\u2082(\u03c4_perm / \u03c4_nat)",
        hBands: [
          { y0: -0.5, y1: 0.5, fill: T.band, label: "No speed effect" },
        ],
        hLines: [
          { y: 0, color: T.axis, width: 1, dash: "none", opacity: 0.85 },
          { y: 1.0, color: T.muted, width: 1.2, dash: "2,3", opacity: 0.9, label: "2\u00d7 slower (\u03c1 = 1.0)", align: "right" },
        ],
        vLines: [
          { x: 4096, color: T.axis, width: 1, dash: "3,4", opacity: 0.55, label: "\u03c4 > 100k", anchor: "end", dx: -6, dy: ph - 16 },
        ],
      });

      const mnRhoReps = reps.map((r) => ({ id: r, rows: data.mnist.unet[r].rho_1_3 }));
      const fmRhoReps = reps.map((r) => ({ id: r, rows: data.fmnist.unet[r].rho_1_3 }));

      // FashionMNIST curves + censored lower-bound arrows at n=2048
      const fmOffsets = { repA: -6, rep0: 0, repB: 6 };
      for (const rep of fmRhoReps) {
        const valid = rep.rows.filter((d) => d.rho !== null || d.rho_lb !== null);
        const pts = valid.map((d) => {
          const val = d.rho !== null ? d.rho : d.rho_lb;
          const xOff = d.rho === null && d.rho_lb !== null ? (fmOffsets[rep.id] || 0) : 0;
          return [xScale(d.n) + xOff, yScale(val)];
        });
        const isMain = rep.id === "rep0";
        drawGuidePath(svg, pts, T.fm, "5,3", isMain ? 2.1 : 1.4, isMain ? 0.95 : 0.55);
        for (const d of valid) {
          if (d.rho !== null) {
            svg.appendChild(drawMarker(svg, xScale(d.n), yScale(d.rho), "triangle", isMain ? 3.2 : 2.3, T.fm, T.surface));
          } else if (d.rho_lb !== null) {
            const cx = xScale(d.n) + (fmOffsets[rep.id] || 0);
            const cy = yScale(d.rho_lb);
            svg.appendChild(drawUpArrow(cx, cy, T.fm, T.surface, 15));
            svg.appendChild(drawOpenMarker(cx, cy, isMain ? 4.0 : 3.3, T.fm, T.surface));
          }
        }
      }
      // Subtle label for censored FMNIST lower bounds
      svg.appendChild(svgEl("text", {
        x: xScale(2048),
        y: yScale(1.20) - 24,
        "text-anchor": "middle",
        fill: T.fm,
        "font-family": T.sans,
        "font-size": 10,
        "font-weight": 600,
        style: `paint-order:stroke;stroke:${T.cardAlt};stroke-width:3.5px;stroke-linejoin:round;`,
        text: "FMNIST lb (>0.97\u2026>1.20)",
      }));

      // MNIST rep0/A/B curves
      for (const rep of mnRhoReps) {
        const valid = rep.rows.filter((d) => d.rho !== null);
        const pts = valid.map((d) => [xScale(d.n), yScale(d.rho)]);
        const isMain = rep.id === "rep0";
        drawGuidePath(svg, pts, T.mn, null, isMain ? 2.2 : 1.45, isMain ? 0.96 : 0.55);
        for (const d of valid) {
          svg.appendChild(drawMarker(svg, xScale(d.n), yScale(d.rho), "circle", isMain ? 3.3 : 2.3, T.mn, T.surface));
        }
      }

      // Optional permutation-identity seeds (1234, 1, 2)
      const permSeedList = [
        { seed: "1234", label: "seed 1234 (=rep0)", color: T.mn, shape: "circle", dash: null },
        { seed: "1", label: "seed 1 (1.16 @ 2048)", color: T.gold, shape: "diamond", dash: "3,2" },
        { seed: "2", label: "seed 2 (0.99 @ 2048)", color: T.perm, shape: "square", dash: "2,2" },
      ];
      if (state.showPermSeeds && data.permctl) {
        for (const ps of permSeedList) {
          const entry = data.permctl[ps.seed];
          if (!entry || !entry.rho_table) continue;
          const valid = entry.rho_table.filter((d) => d.rho !== null);
          const pts = valid.map((d) => [xScale(d.n), yScale(d.rho)]);
          drawGuidePath(svg, pts, ps.color, ps.dash, ps.seed === "1234" ? 2.4 : 1.9, 0.95);
          for (const d of valid) {
            svg.appendChild(drawMarker(svg, xScale(d.n), yScale(d.rho), ps.shape, 3.6, ps.color, T.surface));
          }
        }
        // Highlight seed 2 = 0.993 (0.01 miss)
        const s2 = data.permctl["2"] && data.permctl["2"].rho["2048"];
        if (s2 !== undefined) {
          svg.appendChild(svgEl("text", {
            x: xScale(2048) + 8,
            y: yScale(s2) + 14,
            fill: T.perm,
            "font-family": T.sans,
            "font-size": 10,
            "font-weight": 600,
            text: `seed 2: ${fmtNum(s2, 2)} (0.01 miss)`,
          }));
        }
      }

      // Hover
      const nTicks = [512, 1024, 2048, 4096];
      const hoverG = addHoverLayer(svg);
      addTrapRect(
        svg,
        m,
        pw,
        ph,
        (e) => {
          const rect = svg.getBoundingClientRect();
          const mx = ((e.clientX - rect.left) / rect.width) * w;
          let bestN = nTicks[0];
          let bestDist = Infinity;
          for (const nv of nTicks) {
            const dist = Math.abs(xScale(nv) - mx);
            if (dist < bestDist) {
              bestDist = dist;
              bestN = nv;
            }
          }
          clear(hoverG);
          const hx = xScale(bestN);
          hoverG.appendChild(svgEl("line", {
            x1: hx,
            x2: hx,
            y1: m.t,
            y2: m.t + ph,
            stroke: T.muted,
            "stroke-width": 1,
            "stroke-dasharray": "3,3",
          }));

          const mnStrs = mnRhoReps.map((r) => {
            const row = r.rows.find((x) => x.n === bestN);
            if (!row || row.rho === null) return "\u2014";
            hoverG.appendChild(svgEl("circle", { cx: hx, cy: yScale(row.rho), r: 5.5, fill: "none", stroke: T.mn, "stroke-width": 1.8 }));
            return fmtNum(row.rho, 2);
          });
          const fmStrs = fmRhoReps.map((r) => {
            const row = r.rows.find((x) => x.n === bestN);
            if (!row) return "\u2014";
            if (row.rho !== null) {
              hoverG.appendChild(svgEl("circle", { cx: hx, cy: yScale(row.rho), r: 5.5, fill: "none", stroke: T.fm, "stroke-width": 1.8 }));
              return fmtNum(row.rho, 2);
            }
            if (row.rho_lb !== null) {
              const cx = hx + (fmOffsets[r.id] || 0);
              hoverG.appendChild(svgEl("circle", { cx, cy: yScale(row.rho_lb), r: 6, fill: "none", stroke: T.fm, "stroke-width": 1.8 }));
              return `>${fmtNum(row.rho_lb, 2)} (lb)`;
            }
            return "\u2014";
          });

          let html = `<div class="viz-tooltip__title">Dataset size n = ${bestN}</div>`;
          if (bestN === 4096) {
            html += tipRow("MNIST / FMNIST", "\u03c4 > 100k (censored)", T.muted);
          } else {
            html += tipRow("MNIST \u03c1 (rep0/A/B)", mnStrs.join(" / "), T.mn);
            html += tipRow("FMNIST \u03c1 (rep0/A/B)", fmStrs.join(" / "), T.fm);
            if (bestN === 2048 && data.mnist.budget_300k_rho) {
              const r300 = data.mnist.budget_300k_rho.find((x) => x.n === 2048);
              if (r300 && r300.rho !== null) {
                html += tipRow("MNIST 300k (exp06 s0)", fmtNum(r300.rho, 2), T.mn);
              }
            }
            if (state.showPermSeeds && data.permctl) {
              const s1234 = data.permctl["1234"].rho[String(bestN)];
              const s1 = data.permctl["1"].rho[String(bestN)];
              const s2 = data.permctl["2"].rho[String(bestN)];
              html += tipRow("Perm seeds (1234/1/2)", [s1234, s1, s2].map((v) => (v == null ? "\u2014" : fmtNum(v, 2))).join(" / "), T.gold);
            }
          }
          tooltip.show(e.clientX, e.clientY, html);
        },
        () => {
          clear(hoverG);
          tooltip.hide();
        }
      );

      clear(rightLegend);
      const legItems = [
        { label: "MNIST (rep0/A/B)", color: T.mn },
        { label: "FMNIST (rep0/A/B, \u25cb\u2191 lb)", color: T.fm, dashed: true },
      ];
      if (state.showPermSeeds) {
        legItems.push({ label: "Perm seed 1", color: T.gold, dashed: true });
        legItems.push({ label: "Perm seed 2", color: T.perm, dotted: true });
      }
      rightLegend.appendChild(buildLegend(legItems));

      clear(rightPlot);
      rightPlot.appendChild(svg);
    }

    function drawRightDanger() {
      const w = rightPlot.clientWidth;
      if (!w) return;
      const T = K.theme();
      const h = 276;
      const m = { l: 48, r: 22, t: 18, b: 38 };
      const pw = w - m.l - m.r;
      const ph = h - m.t - m.b;

      const stepData = (data.danger_zone && data.danger_zone[state.dzStep]) || { nat: [], perm: [] };
      const natRows = stepData.nat || [];
      const permRows = stepData.perm || [];

      const maxY = state.dzStep === "10000" ? 0.5 : state.dzStep === "40000" ? 1.2 : 1.8;
      const yTicks = state.dzStep === "10000"
        ? [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        : state.dzStep === "40000"
          ? [0.0, 0.3, 0.6, 0.9, 1.2]
          : [0.0, 0.4, 0.8, 1.2, 1.6];

      const xScale = scaleLinear([0, 940], [m.l, m.l + pw]);
      const yScale = scaleLinear([-0.04, maxY], [m.t + ph, m.t]);

      const svg = makeSvg(w, h);
      drawAxes(svg, {
        w,
        h,
        m,
        xScale,
        yScale,
        xTicks: [50, 150, 300, 500, 700, 900],
        yTicks,
        xFormat: (v) => String(v),
        yFormat: (v) => fmtNum(v, 1),
        xLabel: "Diffusion noise timestep t (low noise on left)",
        yLabel: state.dzMetric === "gap" ? "Test \u2212 Train MSE Gap" : "Denoising MSE (Train & Test)",
        hLines: [
          { y: 0, color: T.axis, width: 1, dash: "none", opacity: 0.85 },
        ],
      });

      // Vertical low-noise danger zone shading (t = 20..200)
      const dzX0 = xScale(20);
      const dzX1 = xScale(200);
      svg.appendChild(svgEl("rect", {
        x: dzX0,
        y: m.t,
        width: dzX1 - dzX0,
        height: ph,
        fill: T.permSoft,
        opacity: 0.6,
      }));
      svg.appendChild(svgEl("text", {
        x: dzX1 + 6,
        y: m.t + 13,
        fill: T.perm,
        "font-family": T.sans,
        "font-size": 10,
        "font-weight": 600,
        style: `paint-order:stroke;stroke:${T.cardAlt};stroke-width:3px;stroke-linejoin:round;`,
        text: "Danger Zone (low t)",
      }));

      if (state.dzMetric === "gap") {
        const natPts = natRows.map((r) => [xScale(r.t), yScale(r.gap)]);
        const permPts = permRows.map((r) => [xScale(r.t), yScale(r.gap)]);
        drawAreaBetween(svg, permPts, natPts, T.perm, 0.14);
        drawGuidePath(svg, natPts, T.nat, null, 2.2, 0.96);
        drawGuidePath(svg, permPts, T.perm, null, 2.2, 0.96);
        for (const r of natRows) {
          svg.appendChild(drawMarker(svg, xScale(r.t), yScale(r.gap), "circle", 3.6, T.nat, T.surface));
        }
        for (const r of permRows) {
          svg.appendChild(drawMarker(svg, xScale(r.t), yScale(r.gap), "square", 3.5, T.perm, T.surface));
        }
        // Callout at t=50
        const p50 = permRows[0];
        const n50 = natRows[0];
        if (p50 && n50) {
          svg.appendChild(svgEl("text", {
            x: xScale(50) + 10,
            y: yScale(p50.gap) + 4,
            fill: T.perm,
            "font-family": T.mono,
            "font-size": 10.5,
            "font-weight": 600,
            style: `paint-order:stroke;stroke:${T.cardAlt};stroke-width:3px;stroke-linejoin:round;`,
            text: `Perm: ${fmtNum(p50.gap, 2)}`,
          }));
          svg.appendChild(svgEl("text", {
            x: xScale(50) + 10,
            y: yScale(n50.gap) + (p50.gap - n50.gap < 0.1 ? 14 : 4),
            fill: T.nat,
            "font-family": T.mono,
            "font-size": 10.5,
            "font-weight": 600,
            style: `paint-order:stroke;stroke:${T.cardAlt};stroke-width:3px;stroke-linejoin:round;`,
            text: `Nat: ${fmtNum(n50.gap, 2)}`,
          }));
        }
      } else {
        // Test MSE (solid) vs Train MSE (dashed)
        const natTest = natRows.map((r) => [xScale(r.t), yScale(r.test)]);
        const permTest = permRows.map((r) => [xScale(r.t), yScale(r.test)]);
        const natTrain = natRows.map((r) => [xScale(r.t), yScale(r.train)]);
        const permTrain = permRows.map((r) => [xScale(r.t), yScale(r.train)]);
        drawGuidePath(svg, natTest, T.nat, null, 2.2, 0.96);
        drawGuidePath(svg, permTest, T.perm, null, 2.2, 0.96);
        drawGuidePath(svg, natTrain, T.nat, "4,3", 1.7, 0.85);
        drawGuidePath(svg, permTrain, T.perm, "4,3", 1.7, 0.85);
        for (const r of natRows) {
          svg.appendChild(drawMarker(svg, xScale(r.t), yScale(r.test), "circle", 3.5, T.nat, T.surface));
          svg.appendChild(drawOpenMarker(xScale(r.t), yScale(r.train), 2.8, T.nat, T.surface));
        }
        for (const r of permRows) {
          svg.appendChild(drawMarker(svg, xScale(r.t), yScale(r.test), "square", 3.4, T.perm, T.surface));
          svg.appendChild(drawOpenMarker(xScale(r.t), yScale(r.train), 2.8, T.perm, T.surface));
        }
      }

      // Hover
      const tVals = [50, 150, 300, 500, 700, 900];
      const hoverG = addHoverLayer(svg);
      addTrapRect(
        svg,
        m,
        pw,
        ph,
        (e) => {
          const rect = svg.getBoundingClientRect();
          const mx = ((e.clientX - rect.left) / rect.width) * w;
          let bestT = tVals[0];
          let bestDist = Infinity;
          for (const tv of tVals) {
            const dist = Math.abs(xScale(tv) - mx);
            if (dist < bestDist) {
              bestDist = dist;
              bestT = tv;
            }
          }
          clear(hoverG);
          const hx = xScale(bestT);
          hoverG.appendChild(svgEl("line", {
            x1: hx,
            x2: hx,
            y1: m.t,
            y2: m.t + ph,
            stroke: T.muted,
            "stroke-width": 1,
            "stroke-dasharray": "3,3",
          }));

          const nr = natRows.find((r) => r.t === bestT);
          const pr = permRows.find((r) => r.t === bestT);
          if (nr && pr) {
            if (state.dzMetric === "gap") {
              hoverG.appendChild(svgEl("circle", { cx: hx, cy: yScale(nr.gap), r: 6, fill: "none", stroke: T.nat, "stroke-width": 2 }));
              hoverG.appendChild(svgEl("circle", { cx: hx, cy: yScale(pr.gap), r: 6, fill: "none", stroke: T.perm, "stroke-width": 2 }));
            } else {
              hoverG.appendChild(svgEl("circle", { cx: hx, cy: yScale(nr.test), r: 6, fill: "none", stroke: T.nat, "stroke-width": 2 }));
              hoverG.appendChild(svgEl("circle", { cx: hx, cy: yScale(pr.test), r: 6, fill: "none", stroke: T.perm, "stroke-width": 2 }));
            }
            const ratioStr = nr.gap > 0.01 ? `${(pr.gap / nr.gap).toFixed(2)}\u00d7` : "\u2014";
            let html = `<div class="viz-tooltip__title">MNIST n=2048 \u00b7 T=${fmtSteps(Number(state.dzStep))} \u00b7 t=${bestT}</div>`;
            html += tipRow("Natural gap (test \u2212 train)", `${fmtNum(nr.gap, 3)} (${fmtNum(nr.test, 3)} \u2212 ${fmtNum(nr.train, 3)})`, T.nat);
            html += tipRow("Permuted gap (test \u2212 train)", `${fmtNum(pr.gap, 3)} (${fmtNum(pr.test, 3)} \u2212 ${fmtNum(pr.train, 3)})`, T.perm);
            html += tipRow("Perm / Nat gap ratio", ratioStr, T.gold);
            tooltip.show(e.clientX, e.clientY, html);
          }
        },
        () => {
          clear(hoverG);
          tooltip.hide();
        }
      );

      clear(rightLegend);
      const legItems = state.dzMetric === "gap"
        ? [
            { label: "Natural MNIST (n=2048)", color: T.nat },
            { label: "Permuted MNIST (n=2048)", color: T.perm },
          ]
        : [
            { label: "Natural Test MSE", color: T.nat },
            { label: "Natural Train MSE", color: T.nat, dashed: true },
            { label: "Permuted Test MSE", color: T.perm },
            { label: "Permuted Train MSE", color: T.perm, dashed: true },
          ];
      rightLegend.appendChild(buildLegend(legItems));

      clear(rightPlot);
      rightPlot.appendChild(svg);
    }

    drawLeft();
    drawRight();
    mountResponsive(leftPlot, drawLeft);
    mountResponsive(rightPlot, drawRight);
  }

  /* =========================================================================
   * WIDGET 6: mountScaleArch(container, data) -> #viz-scale-arch
   * ========================================================================= */
  function mountScaleArch(container, data) {
    const sa = data.scale_arch || {};
    const widths = sa.widths || [];
    const archs = sa.architectures || [];
    const ditTrajs = sa.dit_trajs || {};
    const cifar = sa.cifar10 || {};

    const state = {
      topTab: "arch",          // "arch" | "cifar"
      widthView: "rho_delta",  // "rho_delta" | "bonnaire"
      selectedArchId: "unet_x1",
      archCurveMode: "mn",     // "mn" | "mstep"
      cifarFocusN: "all",      // "all" | "128" | "256" | "512"
      cifarXScale: "linear",   // "linear" | "log"
    };

    // Compute real MLP repA/repB mean m(n, 100k) if user selects mlp_null
    const mlpCurves = (() => {
      if (!data.mnist || !data.mnist.mlp) return null;
      const rA = (data.mnist.mlp.repA.delta_1_3 || []).find((r) => r.step === 100000);
      const rB = (data.mnist.mlp.repB.delta_1_3 || []).find((r) => r.step === 100000);
      if (!rA || !rB) return null;
      const m_nat = {};
      const m_perm = {};
      for (const k of Object.keys(rA.m_nat)) {
        m_nat[k] = 0.5 * (rA.m_nat[k] + rB.m_nat[k]);
        m_perm[k] = 0.5 * (rA.m_perm[k] + rB.m_perm[k]);
      }
      return { m_nat, m_perm };
    })();

    // ---- Header & Top Seg ----
    const header = el("div", { class: "viz-header" }, [
      el("div", null, [
        el("h3", { class: "viz-title", text: "Exploratory Scale-Out: U-Net Width, Transformer Patch Size & CIFAR-10 at 300k" }),
        el("p", {
          class: "viz-subtitle",
          text: "Single-seed exploratory extensions beyond the main MNIST/FMNIST protocol \u2014 click an architecture or switch tabs to inspect curves and real CIFAR-10 outputs.",
        }),
      ]),
      el("span", { class: "viz-badge viz-badge--warn", text: "Single-seed exploratory" }),
    ]);

    const topSeg = makeSeg(
      [
        { value: "arch", label: "Width Sweep & Architecture Dial" },
        { value: "cifar", label: "CIFAR-10: 100k \u2192 300k & Real Samples" },
      ],
      state.topTab,
      (val) => {
        state.topTab = val;
        syncTopTabs();
      }
    );
    const topControls = el("div", { class: "viz-controls" }, [topSeg]);

    // ---- TAB 1: Width Sweep & Architecture Dial ----
    const widthSubSeg = makeSeg(
      [
        { value: "rho_delta", label: "Slowdown \u03c1 & \u0394 vs Width" },
        { value: "bonnaire", label: "Bonnaire Onset Scaling (log\u2082 n* vs Width)" },
      ],
      state.widthView,
      (val) => {
        state.widthView = val;
        drawWidth();
      }
    );
    const widthPlot = el("div", { style: "min-height:270px;" });
    const widthLegend = el("div");
    const widthPanel = el("div", { class: "viz-panel" }, [
      el("div", { class: "viz-panel-title" }, [
        el("span", { text: "U-Net Width Sweep (MNIST, 1.15M \u2192 14.2M params)" }),
      ]),
      el("div", { class: "viz-controls", style: "margin-bottom:8px;" }, [widthSubSeg]),
      widthPlot,
      widthLegend,
    ]);

    // Right panel: Architecture Dial + Mini Trajectory Inset
    const dialPlot = el("div", { style: "min-height:178px;" });
    const quickArchOpts = [
      { value: "unet_x1", label: "U-Net \u00d71" },
      { value: "D2_p2", label: "DiT-p2" },
      { value: "D3_p4", label: "DiT-p4" },
    ];
    const quickArchSeg = makeSeg(quickArchOpts, state.selectedArchId, (val) => {
      selectArch(val);
    });
    const insetModeSeg = makeSeg(
      [
        { value: "mn", label: "m(n, 100k)" },
        { value: "mstep", label: "m(step)" },
      ],
      state.archCurveMode,
      (val) => {
        state.archCurveMode = val;
        drawArchInset();
      }
    );
    const insetTitleLabel = el("span", { style: "font-size:12px;font-weight:600;color:var(--text);", text: "Conv U-Net \u00d71" });
    const insetTitleReadout = el("span", { class: "viz-readout", text: "" });
    const archInsetPlot = el("div", { style: "min-height:152px;" });
    const archStatStrip = el("div", { class: "viz-stat-strip viz-stat-strip--2col" });

    const archPanel = el("div", { class: "viz-panel" }, [
      el("div", { class: "viz-panel-title" }, [
        el("span", { text: "Architecture Locality Dial (click a row to inspect)" }),
      ]),
      dialPlot,
      el("div", {
        style: "margin-top:8px;padding-top:10px;border-top:1px solid var(--border);",
      }, [
        el("div", { class: "viz-controls", style: "justify-content:space-between;margin-bottom:6px;gap:6px;" }, [
          quickArchSeg,
          insetModeSeg,
        ]),
        el("div", { style: "display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;gap:8px;flex-wrap:wrap;" }, [
          insetTitleLabel,
          insetTitleReadout,
        ]),
        archInsetPlot,
        archStatStrip,
      ]),
    ]);

    const tab1Wrap = el("div", { class: "viz-grid-2" }, [widthPanel, archPanel]);

    // ---- TAB 2: CIFAR-10 100k -> 300k & Real Samples ----
    const cifarNSeg = makeSeg(
      [
        { value: "all", label: "All n" },
        { value: "128", label: "n = 128" },
        { value: "256", label: "n = 256" },
        { value: "512", label: "n = 512" },
      ],
      state.cifarFocusN,
      (val) => {
        state.cifarFocusN = val;
        drawCifarTraj();
      }
    );
    const cifarScaleSeg = makeSeg(
      [
        { value: "linear", label: "Linear steps" },
        { value: "log", label: "Log steps" },
      ],
      state.cifarXScale,
      (val) => {
        state.cifarXScale = val;
        drawCifarTraj();
      }
    );
    const cifarTrajPlot = el("div", { style: "min-height:252px;" });
    const cifarTrajLegend = el("div");
    const cifarTrajStats = el("div", { class: "viz-stat-strip viz-stat-strip--2col" });

    const cifarLeftPanel = el("div", { class: "viz-panel" }, [
      el("div", { class: "viz-panel-title" }, [
        el("span", { text: "CIFAR-10 U-Net Trajectories m(step), 2k \u2192 300k" }),
        el("span", { class: "viz-badge viz-badge--warn", text: cifar.primary_300k || "INCONCLUSIVE" }),
      ]),
      el("div", { class: "viz-controls", style: "justify-content:space-between;margin-bottom:8px;gap:6px;" }, [
        cifarNSeg,
        cifarScaleSeg,
      ]),
      cifarTrajPlot,
      cifarTrajLegend,
      cifarTrajStats,
    ]);

    const cifarMnPlot = el("div", { style: "min-height:164px;" });
    const cifarMnLegend = el("div");
    const cifarSpriteWrap = el("div", { class: "cifar-grid-strip", style: "grid-template-columns:repeat(auto-fit,minmax(195px,1fr));" });

    const cifarRightPanel = el("div", { class: "viz-panel" }, [
      el("div", { class: "viz-panel-title" }, [
        el("span", { text: "CIFAR-10 Onset m(n) at 100k & Real Sample Sheet" }),
        el("span", { class: "viz-readout", text: `\u0394(100k) = ${fmtSigned(cifar.delta_100k, 2)}` }),
      ]),
      cifarMnPlot,
      cifarMnLegend,
      el("div", {
        style: "margin-top:10px;padding-top:10px;border-top:1px solid var(--border);",
      }, [
        el("div", { style: "display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap;" }, [
          el("span", { style: "font-size:12px;font-weight:600;color:var(--text);", text: "Generated CIFAR-10 Samples (n = 128) \u2014 Hover any tile for d\u2081, d\u2082" }),
          el("span", { style: "font-size:11px;color:var(--muted);", text: "Teal dot = Memorised (d\u2081/d\u2082 < 1/3) \u00b7 Orange = Novel" }),
        ]),
        cifarSpriteWrap,
      ]),
    ]);

    const tab2Wrap = el("div", { class: "viz-grid-2" }, [cifarLeftPanel, cifarRightPanel]);

    const figcaption = el("div", { class: "viz-figcaption" }, [
      el("b", { text: "Exploratory summary: " }),
      "Doubling U-Net width reproduces Bonnaire et al.\u2019s +1 octave/2\u00d7W capacity scaling while driving speed slowdown \u03c1 toward zero by \u00d72 (saturation); replacing 3\u00d73 convolutions with 2\u00d72 or 4\u00d74 transformer patches or 1\u00d71 pixel tokens dials \u03c1 smoothly from 0.93 down to 0.00. On CIFAR-10 at 300k, ",
      el("em", { text: "n = 128" }),
      " catches up (0.933, gap 0.067), whereas ",
      el("em", { text: "n = 256" }),
      " plateaus near 0.798 vs. 0.982 (gap 0.184).",
    ]);

    clear(container);
    container.appendChild(header);
    container.appendChild(topControls);
    container.appendChild(tab1Wrap);
    container.appendChild(tab2Wrap);
    container.appendChild(figcaption);

    function selectArch(id) {
      state.selectedArchId = id;
      const segButtons = quickArchSeg.querySelectorAll("button");
      quickArchOpts.forEach((opt, idx) => {
        if (segButtons[idx]) {
          segButtons[idx].setAttribute("aria-pressed", opt.value === id ? "true" : "false");
        }
      });
      const hasTraj = Boolean(ditTrajs[id]);
      insetModeSeg.style.display = hasTraj ? "inline-flex" : "none";
      drawDial();
      drawArchInset();
    }

    function syncTopTabs() {
      tab1Wrap.style.display = state.topTab === "arch" ? "grid" : "none";
      tab2Wrap.style.display = state.topTab === "cifar" ? "grid" : "none";
      requestAnimationFrame(() => {
        if (state.topTab === "arch") {
          drawWidth();
          drawDial();
          drawArchInset();
        } else {
          drawCifarTraj();
          drawCifarMn();
        }
      });
    }

    // ---- Build CIFAR-10 sprite cards once ----
    function buildCifarSprites() {
      clear(cifarSpriteWrap);
      const assetBase = window.POST_ASSET_BASE || "/assets/diffusion_locality/";
      const spriteUrl = `${assetBase}cifar_samples.png`;
      const rowsMeta = [
        { key: "nat_100k", label: "Natural 100k", badgeCls: "viz-badge--nat" },
        { key: "perm_100k", label: "Permuted 100k", badgeCls: "viz-badge--perm" },
        { key: "nat_300k", label: "Natural 300k", badgeCls: "viz-badge--nat" },
        { key: "perm_300k", label: "Permuted 300k", badgeCls: "viz-badge--perm" },
      ];
      const sprites = cifar.sprites || {};
      // 1x sheet geometry: cols=6, rows=4, tile=32, pad=2 -> W=202, H=134 (scaled 3x in cifar_samples.png: 606x402)
      const bgSizeX = (202 / 32) * 100; // 631.25%
      const bgSizeY = (134 / 32) * 100; // 418.75%

      for (const rm of rowsMeta) {
        const group = sprites[rm.key];
        if (!group) continue;
        const tilesRow = el("div", { class: "cifar-strip-tiles" });
        for (const item of group.items || []) {
          const posX = item.col * 20;           // c * 34 / (202 - 32) * 100
          const posY = (item.row * 100) / 3;    // r * 34 / (134 - 32) * 100
          const dotColor = item.memorized ? "var(--viz-nat)" : "var(--viz-perm)";
          const tile = el("div", {
            class: "cifar-sprite-thumb",
            tabindex: "0",
            role: "img",
            "aria-label": `${rm.label} sample ${item.idx + 1}: ${item.memorized ? "Memorised" : "Novel"}`,
            style: `position:relative;background-image:url("${spriteUrl}");background-size:${bgSizeX}% ${bgSizeY}%;background-position:${posX.toFixed(4)}% ${posY.toFixed(4)}%;`,
          }, [
            el("span", {
              class: "hero-sample-dot",
              style: `top:3px;right:3px;width:6px;height:6px;background:${dotColor};`,
            }),
          ]);

          const showTip = (clientX, clientY) => {
            const statusText = item.memorized ? "Memorised (d\u2081/d\u2082 < 1/3)" : "Novel (d\u2081/d\u2082 \u2265 1/3)";
            const statusColor = item.memorized ? K.theme().nat : K.theme().perm;
            let html = `<div class="viz-tooltip__title">${esc(rm.label)} \u00b7 Sample #${item.idx + 1}</div>`;
            html += tipRow("Nearest train d\u2081", fmtNum(item.d1, 2));
            html += tipRow("2nd nearest d\u2082", fmtNum(item.d2, 2));
            html += tipRow("Ratio d\u2081 / d\u2082", fmtNum(item.ratio, 3));
            html += tipRow("Verdict", statusText, statusColor);
            tooltip.show(clientX, clientY, html);
          };

          tile.addEventListener("mouseenter", (e) => showTip(e.clientX, e.clientY));
          tile.addEventListener("mousemove", (e) => showTip(e.clientX, e.clientY));
          tile.addEventListener("mouseleave", () => tooltip.hide());
          tile.addEventListener("focus", () => {
            const r = tile.getBoundingClientRect();
            showTip(r.left + r.width / 2, r.top);
          });
          tile.addEventListener("blur", () => tooltip.hide());
          tilesRow.appendChild(tile);
        }

        const card = el("div", { class: "cifar-strip-card" }, [
          el("div", { style: "display:flex;justify-content:space-between;align-items:center;gap:6px;" }, [
            el("span", { style: "font-size:12px;font-weight:600;color:var(--text);", text: rm.label }),
            el("span", { class: `viz-badge ${rm.badgeCls}`, style: "padding:1px 7px;font-size:11px;", text: `m = ${fmtNum(group.m13, 3)}` }),
          ]),
          tilesRow,
        ]);
        cifarSpriteWrap.appendChild(card);
      }
    }
    buildCifarSprites();

    // ---- Draw Width Sweep ----
    function drawWidth() {
      const w = widthPlot.clientWidth;
      if (!w || !widths.length) return;
      const T = K.theme();
      const h = 274;
      const m = { l: 56, r: 24, t: 20, b: 40 };
      const pw = w - m.l - m.r;
      const ph = h - m.t - m.b;

      const svg = makeSvg(w, h);
      const hoverG = addHoverLayer(svg);

      if (state.widthView === "rho_delta") {
        const xScale = scaleLog([0.85, 20], [m.l, m.l + pw], 10);
        const yScale = scaleLinear([-0.85, 1.35], [m.t + ph, m.t]);

        drawAxes(svg, {
          w,
          h,
          m,
          xScale,
          yScale,
          xTicks: widths.map((d) => d.params_m),
          yTicks: [-0.5, 0.0, 0.5, 1.0],
          xFormat: (v) => {
            const found = widths.find((d) => d.params_m === v);
            const multStr = found ? (found.width_mult === 0.5 ? "\u00d7\u00bd" : `\u00d7${found.width_mult}`) : "";
            return `${v}M (${multStr})`;
          },
          yFormat: (v) => (v === 0 ? "0.0" : fmtSigned(v, 1)),
          xLabel: "Model parameters (M, log\u2081\u2080 scale)",
          yLabel: "Mean \u03c1 and Onset Shift \u0394(100k)",
          hBands: [
            { y0: -0.5, y1: 0.5, fill: T.band },
          ],
          hLines: [
            { y: 0, color: T.axis, width: 1, dash: "none", opacity: 0.85 },
            { y: 1.0, color: T.muted, width: 1.1, dash: "2,3", opacity: 0.9, label: "2\u00d7 slower (\u03c1 = 1.0)", align: "right" },
          ],
        });

        svg.appendChild(svgEl("text", {
          x: m.l + 8,
          y: yScale(0.5) + 13,
          "text-anchor": "start",
          fill: T.muted,
          "font-family": T.sans,
          "font-size": 10.5,
          text: "\u00b10.5 no-effect / NO SHIFT band",
        }));

        const rhoPts = widths.map((d) => [xScale(d.params_m), yScale(d.rho_mean)]);
        const delPts = widths.map((d) => [xScale(d.params_m), yScale(d.delta_100k)]);
        drawGuidePath(svg, rhoPts, T.perm, null, 2.3, 0.96);
        drawGuidePath(svg, delPts, T.nat, null, 2.3, 0.96);

        widths.forEach((d, idx) => {
          const cx = xScale(d.params_m);
          const cyR = yScale(d.rho_mean);
          const cyD = yScale(d.delta_100k);
          svg.appendChild(drawMarker(svg, cx, cyR, "square", 4.2, T.perm, T.surface));
          svg.appendChild(drawMarker(svg, cx, cyD, "circle", 4.2, T.nat, T.surface));
          const anchor = idx === 0 ? "start" : idx === widths.length - 1 ? "end" : "middle";
          const dx = idx === 0 ? 7 : idx === widths.length - 1 ? -7 : 0;
          svg.appendChild(svgEl("text", {
            x: cx + dx,
            y: cyR - 9,
            "text-anchor": anchor,
            fill: T.perm,
            "font-family": T.mono,
            "font-size": 10.5,
            "font-weight": 600,
            style: `paint-order:stroke;stroke:${T.cardAlt};stroke-width:3.5px;stroke-linejoin:round;`,
            text: fmtNum(d.rho_mean, 2),
          }));
          svg.appendChild(svgEl("text", {
            x: cx + dx,
            y: cyD + 16,
            "text-anchor": anchor,
            fill: T.nat,
            "font-family": T.mono,
            "font-size": 10.5,
            "font-weight": 600,
            style: `paint-order:stroke;stroke:${T.cardAlt};stroke-width:3.5px;stroke-linejoin:round;`,
            text: fmtSigned(d.delta_100k, 2),
          }));
        });

        addTrapRect(
          svg,
          m,
          pw,
          ph,
          (e) => {
            const rect = svg.getBoundingClientRect();
            const mx = ((e.clientX - rect.left) / rect.width) * w;
            let best = widths[0];
            let bestDist = Infinity;
            for (const d of widths) {
              const dist = Math.abs(xScale(d.params_m) - mx);
              if (dist < bestDist) {
                bestDist = dist;
                best = d;
              }
            }
            clear(hoverG);
            const hx = xScale(best.params_m);
            hoverG.appendChild(svgEl("line", { x1: hx, x2: hx, y1: m.t, y2: m.t + ph, stroke: T.muted, "stroke-width": 1, "stroke-dasharray": "3,3" }));
            hoverG.appendChild(svgEl("circle", { cx: hx, cy: yScale(best.rho_mean), r: 6.5, fill: "none", stroke: T.perm, "stroke-width": 2 }));
            hoverG.appendChild(svgEl("circle", { cx: hx, cy: yScale(best.delta_100k), r: 6.5, fill: "none", stroke: T.nat, "stroke-width": 2 }));

            const nRangeStr = `${best.n_range[0]}\u2013${best.n_range[best.n_range.length - 1]}`;
            const rhoDetail = Object.entries(best.rho || {}).map(([n, v]) => `${n}: ${fmtNum(v, 2)}`).join(", ");
            let html = `<div class="viz-tooltip__title">${esc(best.label)} (${best.params_m}M)</div>`;
            html += tipRow("Mean slowdown \u03c1", fmtNum(best.rho_mean, 3), T.perm);
            html += tipRow("Onset shift \u0394(100k)", fmtSigned(best.delta_100k, 3), T.nat);
            html += tipRow("n range for \u03c1", nRangeStr);
            if (rhoDetail) html += tipRow("Per-n \u03c1", rhoDetail);
            html += tipRow("Verdicts", `${best.capacity} / ${best.speed}`);
            tooltip.show(e.clientX, e.clientY, html);
          },
          () => {
            clear(hoverG);
            tooltip.hide();
          }
        );

        clear(widthLegend);
        widthLegend.appendChild(buildLegend([
          { label: "Mean slowdown \u03c1 (\u25a0)", color: T.perm },
          { label: "Fixed-budget \u0394(100k) (\u25cf)", color: T.nat },
        ]));
      } else {
        // Bonnaire Onset Scaling (log2 n* vs Width)
        const xScale = scaleLog([0.38, 2.6], [m.l, m.l + pw], 2);
        const yScale = scaleLinear([9.3, 12.8], [m.t + ph, m.t]);

        drawAxes(svg, {
          w,
          h,
          m,
          xScale,
          yScale,
          xTicks: [0.5, 1, 2],
          yTicks: [9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5],
          xFormat: (v) => (v === 0.5 ? "\u00d7\u00bd (1.15M)" : v === 1 ? "\u00d71 (3.83M)" : "\u00d72 (14.2M)"),
          yFormat: (v) => fmtNum(v, 1),
          xLabel: "U-Net channel width multiplier W (log\u2082 scale)",
          yLabel: "Critical onset log\u2082 n*(100k)",
        });

        // Slope +1 octave per width doubling reference line (anchored to natural mean)
        const meanNat = widths.reduce((acc, d) => acc + d.log2_nstar_nat_100k, 0) / widths.length;
        const refPts = [0.42, 1, 2.35].map((wm) => [xScale(wm), yScale(meanNat + Math.log2(wm))]);
        drawGuidePath(svg, refPts, T.gold, "2,3", 1.8, 0.9);
        svg.appendChild(svgEl("text", {
          x: xScale(0.45),
          y: yScale(meanNat + Math.log2(0.45)) - 10,
          fill: T.gold,
          "font-family": T.sans,
          "font-size": 10.5,
          "font-weight": 600,
          text: "Slope +1 oct / 2\u00d7W (\u03c4_mem \u221d n\u00b7W\u207b\u00b9)",
        }));

        const natPts = widths.map((d) => [xScale(d.width_mult), yScale(d.log2_nstar_nat_100k)]);
        const permPts = widths.map((d) => [xScale(d.width_mult), yScale(d.log2_nstar_perm_100k)]);
        drawGuidePath(svg, natPts, T.nat, null, 2.3, 0.96);
        drawGuidePath(svg, permPts, T.perm, "5,3", 2.2, 0.96);

        widths.forEach((d, idx) => {
          const cx = xScale(d.width_mult);
          const cyN = yScale(d.log2_nstar_nat_100k);
          const cyP = yScale(d.log2_nstar_perm_100k);
          svg.appendChild(drawMarker(svg, cx, cyN, "circle", 4.2, T.nat, T.surface));
          svg.appendChild(drawMarker(svg, cx, cyP, "square", 4.0, T.perm, T.surface));
          const anchor = idx === 0 ? "start" : idx === widths.length - 1 ? "end" : "middle";
          const dx = idx === 0 ? 7 : idx === widths.length - 1 ? -7 : 0;
          svg.appendChild(svgEl("text", {
            x: cx + dx,
            y: cyN - 9,
            "text-anchor": anchor,
            fill: T.nat,
            "font-family": T.mono,
            "font-size": 10.5,
            "font-weight": 600,
            text: fmtNum(d.log2_nstar_nat_100k, 2),
          }));
          svg.appendChild(svgEl("text", {
            x: cx + dx,
            y: cyP + 16,
            "text-anchor": anchor,
            fill: T.perm,
            "font-family": T.mono,
            "font-size": 10.5,
            "font-weight": 600,
            text: fmtNum(d.log2_nstar_perm_100k, 2),
          }));
        });

        addTrapRect(
          svg,
          m,
          pw,
          ph,
          (e) => {
            const rect = svg.getBoundingClientRect();
            const mx = ((e.clientX - rect.left) / rect.width) * w;
            let best = widths[0];
            let bestDist = Infinity;
            for (const d of widths) {
              const dist = Math.abs(xScale(d.width_mult) - mx);
              if (dist < bestDist) {
                bestDist = dist;
                best = d;
              }
            }
            clear(hoverG);
            const hx = xScale(best.width_mult);
            hoverG.appendChild(svgEl("line", { x1: hx, x2: hx, y1: m.t, y2: m.t + ph, stroke: T.muted, "stroke-width": 1, "stroke-dasharray": "3,3" }));
            hoverG.appendChild(svgEl("circle", { cx: hx, cy: yScale(best.log2_nstar_nat_100k), r: 6.5, fill: "none", stroke: T.nat, "stroke-width": 2 }));
            hoverG.appendChild(svgEl("circle", { cx: hx, cy: yScale(best.log2_nstar_perm_100k), r: 6.5, fill: "none", stroke: T.perm, "stroke-width": 2 }));

            let html = `<div class="viz-tooltip__title">${esc(best.label)} (${best.params_m}M)</div>`;
            html += tipRow("log\u2082 n*_nat (100k)", fmtNum(best.log2_nstar_nat_100k, 2), T.nat);
            html += tipRow("log\u2082 n*_perm (100k)", fmtNum(best.log2_nstar_perm_100k, 2), T.perm);
            html += tipRow("\u0394(100k)", fmtSigned(best.delta_100k, 3));
            html += tipRow("Bonnaire +1 oct/2\u00d7W ref", fmtNum(meanNat + Math.log2(best.width_mult), 2), T.gold);
            tooltip.show(e.clientX, e.clientY, html);
          },
          () => {
            clear(hoverG);
            tooltip.hide();
          }
        );

        clear(widthLegend);
        widthLegend.appendChild(buildLegend([
          { label: "Natural log\u2082 n*(100k)", color: T.nat },
          { label: "Permuted log\u2082 n*(100k)", color: T.perm, dashed: true },
          { label: "Slope +1 oct/2\u00d7W (Bonnaire et al. 2025)", color: T.gold, dotted: true },
        ]));
      }

      clear(widthPlot);
      widthPlot.appendChild(svg);
    }

    // ---- Draw Architecture Locality Dial (5-row horizontal lollipop) ----
    function drawDial() {
      const w = dialPlot.clientWidth;
      if (!w || !archs.length) return;
      const T = K.theme();
      const rowH = 30;
      const m = { t: 24, b: 24, l: Math.min(118, Math.round(w * 0.28)), r: 12 };
      const h = m.t + archs.length * rowH + m.b;
      const gap = 26;
      const tracksW = Math.max(120, w - m.l - m.r - gap);
      const t1W = tracksW * 0.54;
      const t2W = tracksW * 0.46;
      const t1X0 = m.l;
      const t2X0 = m.l + t1W + gap;

      const xRho = scaleLinear([-0.68, 1.48], [t1X0, t1X0 + t1W]);
      const xDel = scaleLinear([-0.82, 0.50], [t2X0, t2X0 + t2W]);

      const svg = makeSvg(w, h);

      // Column headers
      svg.appendChild(svgEl("text", {
        x: t1X0 + t1W / 2,
        y: 13,
        "text-anchor": "middle",
        fill: T.text,
        "font-family": T.sans,
        "font-size": 11,
        "font-weight": 600,
        text: "Slowdown \u03c1 (mean)",
      }));
      svg.appendChild(svgEl("text", {
        x: t2X0 + t2W / 2,
        y: 13,
        "text-anchor": "middle",
        fill: T.text,
        "font-family": T.sans,
        "font-size": 11,
        "font-weight": 600,
        text: "Onset Shift \u0394(100k)",
      }));

      const yTop = m.t;
      const yBot = m.t + archs.length * rowH;

      // [-0.5, 0.5] bands on both tracks
      svg.appendChild(svgEl("rect", {
        x: xRho(-0.5),
        y: yTop,
        width: xRho(0.5) - xRho(-0.5),
        height: yBot - yTop,
        fill: T.band,
      }));
      svg.appendChild(svgEl("rect", {
        x: xDel(-0.5),
        y: yTop,
        width: xDel(0.4) - xDel(-0.5),
        height: yBot - yTop,
        fill: T.band,
      }));

      // Vertical reference lines
      for (const tx of [0, 1.0]) {
        const x = xRho(tx);
        svg.appendChild(svgEl("line", {
          x1: x,
          x2: x,
          y1: yTop,
          y2: yBot,
          stroke: tx === 0 ? T.axis : T.muted,
          "stroke-width": 1,
          "stroke-dasharray": tx === 0 ? null : "2,3",
        }));
        svg.appendChild(svgEl("text", {
          x,
          y: yBot + 14,
          "text-anchor": "middle",
          fill: T.muted,
          "font-family": T.mono,
          "font-size": 10,
          text: tx === 0 ? "0" : "1.0 (2\u00d7)",
        }));
      }
      for (const tx of [-0.5, 0]) {
        const x = xDel(tx);
        svg.appendChild(svgEl("line", {
          x1: x,
          x2: x,
          y1: yTop,
          y2: yBot,
          stroke: tx === 0 ? T.axis : T.grid,
          "stroke-width": 1,
          "stroke-dasharray": tx === 0 ? null : "3,3",
        }));
        svg.appendChild(svgEl("text", {
          x,
          y: yBot + 14,
          "text-anchor": "middle",
          fill: T.muted,
          "font-family": T.mono,
          "font-size": 10,
          text: tx === 0 ? "0" : "\u22120.5",
        }));
      }

      const shortName = {
        unet_x1: "Conv U-Net \u00d71",
        D2_p2: "DiT patch 2\u00d72",
        D3_p4: "DiT patch 4\u00d74",
        D1_p1: "DiT patch 1\u00d71",
        mlp_null: "Residual MLP",
      };

      archs.forEach((arch, idx) => {
        const ry = m.t + idx * rowH;
        const cy = ry + rowH / 2;
        const isSel = arch.id === state.selectedArchId;

        const rowG = svgEl("g", { style: "cursor:pointer;" });
        const bgRect = svgEl("rect", {
          x: 2,
          y: ry + 2,
          width: w - 4,
          height: rowH - 4,
          rx: 6,
          fill: isSel ? T.natSoft : "transparent",
          stroke: isSel ? T.nat : "transparent",
          "stroke-width": 1.2,
        });
        rowG.appendChild(bgRect);

        // Label
        rowG.appendChild(svgEl("text", {
          x: m.l - 8,
          y: cy + 3.5,
          "text-anchor": "end",
          fill: isSel ? T.text : T.muted,
          "font-family": T.sans,
          "font-size": 11.5,
          "font-weight": isSel ? 700 : 500,
          text: shortName[arch.id] || arch.label,
        }));

        // Track 1: rho lollipop
        const rColor = arch.speed === "SPEED EFFECT" ? T.perm : T.muted;
        const rx0 = xRho(0);
        const rx1 = xRho(arch.rho_mean);
        rowG.appendChild(svgEl("line", {
          x1: rx0,
          x2: rx1,
          y1: cy,
          y2: cy,
          stroke: rColor,
          "stroke-width": 2.4,
          opacity: 0.9,
        }));
        if (arch.id === "D1_p1") {
          rowG.appendChild(drawMarker(svg, rx1, cy, "diamond", 4.5, T.surface, T.muted));
        } else {
          rowG.appendChild(drawMarker(svg, rx1, cy, "circle", 4.2, rColor, T.surface));
        }
        const rTxt = arch.id === "D1_p1" ? "0.00 (exact)" : fmtNum(arch.rho_mean, 2);
        const rRight = arch.rho_mean >= 0;
        // For negative bars, place label right of the 0 axis so it never enters the architecture label column
        const rTextX = rRight ? rx1 + 7 : rx0 + 7;
        rowG.appendChild(svgEl("text", {
          x: rTextX,
          y: cy + 3.5,
          "text-anchor": "start",
          fill: rColor,
          "font-family": T.mono,
          "font-size": 10.5,
          "font-weight": 600,
          style: `paint-order:stroke;stroke:${T.cardAlt};stroke-width:3px;stroke-linejoin:round;`,
          text: rTxt,
        }));

        // Track 2: delta lollipop
        const dColor = arch.id === "D1_p1" ? T.muted : T.nat;
        const dx0 = xDel(0);
        const dx1 = xDel(arch.delta_100k);
        rowG.appendChild(svgEl("line", {
          x1: dx0,
          x2: dx1,
          y1: cy,
          y2: cy,
          stroke: dColor,
          "stroke-width": 2.4,
          opacity: 0.9,
        }));
        rowG.appendChild(drawMarker(svg, dx1, cy, arch.id === "D1_p1" ? "diamond" : "circle", 4.2, arch.id === "D1_p1" ? T.surface : dColor, arch.id === "D1_p1" ? T.muted : T.surface));
        const dRight = arch.delta_100k >= 0;
        rowG.appendChild(svgEl("text", {
          x: dRight ? dx1 + 7 : dx1 - 7,
          y: cy + 3.5,
          "text-anchor": dRight ? "start" : "end",
          fill: dColor,
          "font-family": T.mono,
          "font-size": 10.5,
          "font-weight": 600,
          text: fmtSigned(arch.delta_100k, 2),
        }));

        rowG.addEventListener("click", () => selectArch(arch.id));
        const showTip = (e) => {
          const perN = Object.entries(arch.rho || {}).map(([n, v]) => `${n}: ${fmtNum(v, 2)}`).join(", ");
          let html = `<div class="viz-tooltip__title">${esc(arch.label)} (${arch.params_m}M)</div>`;
          html += `<div style="margin-bottom:4px;opacity:0.86;">${esc(arch.locality_desc)}</div>`;
          html += tipRow("Mean slowdown \u03c1", arch.id === "D1_p1" ? "0.000 (exact)" : fmtNum(arch.rho_mean, 3), rColor);
          if (perN) html += tipRow("Per-n \u03c1", perN);
          html += tipRow("Onset shift \u0394(100k)", fmtSigned(arch.delta_100k, 3), dColor);
          html += tipRow("Verdict", `${arch.capacity} / ${arch.speed}`);
          tooltip.show(e.clientX, e.clientY, html);
        };
        rowG.addEventListener("mouseenter", showTip);
        rowG.addEventListener("mousemove", showTip);
        rowG.addEventListener("mouseleave", () => tooltip.hide());

        svg.appendChild(rowG);
      });

      clear(dialPlot);
      dialPlot.appendChild(svg);
    }

    // ---- Draw Mini Trajectory Inset for Selected Architecture ----
    function drawArchInset() {
      const w = archInsetPlot.clientWidth;
      if (!w) return;
      const T = K.theme();
      const arch = archs.find((a) => a.id === state.selectedArchId) || archs[0];
      if (!arch) return;
      insetTitleLabel.textContent = arch.label;

      // Update stats strip (avoid cross-arch n* comparisons; focus on within-arch params, delta, rho, verdict)
      renderStatStrip(archStatStrip, [
        { label: "Params", value: `${arch.params_m}M` },
        { label: "\u0394(100k)", value: fmtSigned(arch.delta_100k, 2), color: T.nat },
        { label: "Mean \u03c1", value: arch.id === "D1_p1" ? "0.00 (exact)" : fmtNum(arch.rho_mean, 2), color: arch.speed === "SPEED EFFECT" ? T.perm : T.muted },
        { label: "Speed / Cap", value: arch.speed, color: arch.speed === "SPEED EFFECT" ? T.perm : T.nat },
      ]);

      // Special case: DiT-p1 exact equivariance (no m_nat/m_perm curves needed; exact symmetry)
      if (arch.id === "D1_p1") {
        insetTitleReadout.textContent = "Exact permutation equivariance";
        clear(archInsetPlot);
        archInsetPlot.appendChild(el("div", {
          style: `background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:14px 16px;font-size:12.5px;line-height:1.5;color:var(--text);`,
        }, [
          el("div", { style: "font-weight:700;margin-bottom:4px;color:var(--text);", text: "DiT patch 1\u00d71 (learned positional embedding): Exact Symmetry Null" }),
          el("div", { style: "color:var(--muted);", text: "Each of the 784 pixels is its own token with a learned positional vector. Permuting input pixels permutes the token embeddings identically, so natural and permuted trajectories are mathematically indistinguishable across every n and every step." }),
          el("div", { style: "margin-top:8px;font-family:var(--mono);font-size:12px;color:var(--viz-nat);", text: "Measured max |f(Px) \u2212 P f(x)| = 2.4 \u00d7 10\u207b\u2077 \u00b7 \u0394 = 0.000 \u00b7 \u03c1 = 0.000" }),
        ]));
        return;
      }

      const h = 156;
      const m = { l: 48, r: 16, t: 14, b: 32 };
      const pw = w - m.l - m.r;
      const ph = h - m.t - m.b;
      const svg = makeSvg(w, h);
      const hoverG = addHoverLayer(svg);

      const useTraj = state.archCurveMode === "mstep" && Boolean(ditTrajs[arch.id]);
      if (useTraj) {
        const focusN = arch.id === "unet_x1" ? "2048" : "256";
        const pair = ditTrajs[arch.id][focusN];
        insetTitleReadout.textContent = `m(step) at n = ${focusN}`;
        const maxStep = arch.id === "unet_x1" && Number(focusN) <= 512 ? 30000 : 100000;
        const xScale = scaleLinear([0, maxStep], [m.l, m.l + pw]);
        const yScale = scaleLinear([0, 1.04], [m.t + ph, m.t]);

        drawAxes(svg, {
          w,
          h,
          m,
          xScale,
          yScale,
          xTicks: [0, 20000, 40000, 60000, 80000, 100000],
          yTicks: [0, 0.5, 1.0],
          xFormat: fmtSteps,
          yFormat: (v) => fmtNum(v, 1),
          xLabel: `Training step (n = ${focusN})`,
          yLabel: "m",
          hLines: [{ y: 0.5, color: T.muted, width: 1, dash: "3,3", opacity: 0.65 }],
        });

        const natPts = pair.nat.map(([s, v]) => [xScale(s), yScale(v)]);
        const permPts = pair.perm.map(([s, v]) => [xScale(s), yScale(v)]);
        drawGuidePath(svg, natPts, T.nat, null, 2.2, 0.96);
        drawGuidePath(svg, permPts, T.perm, "4,3", 2.2, 0.96);
        for (const [s, v] of pair.nat) svg.appendChild(drawMarker(svg, xScale(s), yScale(v), "circle", 3.0, T.nat, T.surface));
        for (const [s, v] of pair.perm) svg.appendChild(drawMarker(svg, xScale(s), yScale(v), "square", 2.9, T.perm, T.surface));

        addTrapRect(
          svg,
          m,
          pw,
          ph,
          (e) => {
            const rect = svg.getBoundingClientRect();
            const mx = ((e.clientX - rect.left) / rect.width) * w;
            let bestIdx = 0;
            let bestDist = Infinity;
            pair.nat.forEach(([s], i) => {
              const dist = Math.abs(xScale(s) - mx);
              if (dist < bestDist) {
                bestDist = dist;
                bestIdx = i;
              }
            });
            clear(hoverG);
            const [st, nv] = pair.nat[bestIdx];
            const pv = pair.perm[bestIdx] ? pair.perm[bestIdx][1] : null;
            const hx = xScale(st);
            hoverG.appendChild(svgEl("line", { x1: hx, x2: hx, y1: m.t, y2: m.t + ph, stroke: T.muted, "stroke-width": 1, "stroke-dasharray": "3,3" }));
            hoverG.appendChild(svgEl("circle", { cx: hx, cy: yScale(nv), r: 5, fill: "none", stroke: T.nat, "stroke-width": 1.8 }));
            if (pv !== null) hoverG.appendChild(svgEl("circle", { cx: hx, cy: yScale(pv), r: 5, fill: "none", stroke: T.perm, "stroke-width": 1.8 }));
            let html = `<div class="viz-tooltip__title">${esc(arch.label)} \u00b7 n=${focusN} \u00b7 ${fmtSteps(st)}</div>`;
            html += tipRow("Natural m", fmtNum(nv, 3), T.nat);
            if (pv !== null) html += tipRow("Permuted m", fmtNum(pv, 3), T.perm);
            tooltip.show(e.clientX, e.clientY, html);
          },
          () => {
            clear(hoverG);
            tooltip.hide();
          }
        );
      } else {
        // m(n, 100k)
        const mNat = arch.id === "mlp_null" && mlpCurves ? mlpCurves.m_nat : arch.m_nat_100k;
        const mPerm = arch.id === "mlp_null" && mlpCurves ? mlpCurves.m_perm : arch.m_perm_100k;
        if (!mNat || !mPerm) return;

        const ns = Object.keys(mNat).map(Number).sort((a, b) => a - b);
        insetTitleReadout.textContent = arch.id === "mlp_null" ? "m(n, 100k) \u00b7 MLP repA/B" : `m(n, 100k) \u00b7 log\u2082 n*_nat = ${fmtNum(arch.log2_nstar_nat, 2)}`;

        const xScale = scaleLog([ns[0] * 0.85, ns[ns.length - 1] * 1.18], [m.l, m.l + pw], 2);
        const yScale = scaleLinear([0, 1.04], [m.t + ph, m.t]);

        drawAxes(svg, {
          w,
          h,
          m,
          xScale,
          yScale,
          xTicks: ns,
          yTicks: [0, 0.5, 1.0],
          xFormat: (v) => String(v),
          yFormat: (v) => fmtNum(v, 1),
          xLabel: "Dataset size n (100k steps)",
          yLabel: "m",
          hLines: [{ y: 0.5, color: T.muted, width: 1, dash: "3,3", opacity: 0.65 }],
        });

        const natPts = ns.map((nv) => [xScale(nv), yScale(mNat[String(nv)])]);
        const permPts = ns.map((nv) => [xScale(nv), yScale(mPerm[String(nv)])]);
        drawGuidePath(svg, natPts, T.nat, null, 2.2, 0.96);
        drawGuidePath(svg, permPts, T.perm, "4,3", 2.2, 0.96);
        for (const nv of ns) {
          svg.appendChild(drawMarker(svg, xScale(nv), yScale(mNat[String(nv)]), "circle", 3.2, T.nat, T.surface));
          svg.appendChild(drawMarker(svg, xScale(nv), yScale(mPerm[String(nv)]), "square", 3.1, T.perm, T.surface));
        }

        addTrapRect(
          svg,
          m,
          pw,
          ph,
          (e) => {
            const rect = svg.getBoundingClientRect();
            const mx = ((e.clientX - rect.left) / rect.width) * w;
            let bestN = ns[0];
            let bestDist = Infinity;
            for (const nv of ns) {
              const dist = Math.abs(xScale(nv) - mx);
              if (dist < bestDist) {
                bestDist = dist;
                bestN = nv;
              }
            }
            clear(hoverG);
            const hx = xScale(bestN);
            const vNat = mNat[String(bestN)];
            const vPerm = mPerm[String(bestN)];
            hoverG.appendChild(svgEl("line", { x1: hx, x2: hx, y1: m.t, y2: m.t + ph, stroke: T.muted, "stroke-width": 1, "stroke-dasharray": "3,3" }));
            hoverG.appendChild(svgEl("circle", { cx: hx, cy: yScale(vNat), r: 5.5, fill: "none", stroke: T.nat, "stroke-width": 1.8 }));
            hoverG.appendChild(svgEl("circle", { cx: hx, cy: yScale(vPerm), r: 5.5, fill: "none", stroke: T.perm, "stroke-width": 1.8 }));
            let html = `<div class="viz-tooltip__title">${esc(arch.label)} \u00b7 n = ${bestN}</div>`;
            html += tipRow("Natural m(100k)", fmtNum(vNat, 3), T.nat);
            html += tipRow("Permuted m(100k)", fmtNum(vPerm, 3), T.perm);
            html += tipRow("Gap (Nat \u2212 Perm)", fmtSigned(vNat - vPerm, 3));
            tooltip.show(e.clientX, e.clientY, html);
          },
          () => {
            clear(hoverG);
            tooltip.hide();
          }
        );
      }

      clear(archInsetPlot);
      archInsetPlot.appendChild(svg);
    }

    // ---- Draw Tab 2 Left: CIFAR-10 Trajectories (2k -> 300k) ----
    function drawCifarTraj() {
      const w = cifarTrajPlot.clientWidth;
      if (!w || !cifar.trajs) return;
      const T = K.theme();
      const h = 258;
      const m = { l: 54, r: 60, t: 20, b: 38 };
      const pw = w - m.l - m.r;
      const ph = h - m.t - m.b;

      const isLog = state.cifarXScale === "log";
      const xScale = isLog
        ? scaleLog([1800, 330000], [m.l, m.l + pw], 10)
        : scaleLinear([0, 310000], [m.l, m.l + pw]);
      const yScale = scaleLinear([0, 1.05], [m.t + ph, m.t]);

      const svg = makeSvg(w, h);
      drawAxes(svg, {
        w,
        h,
        m,
        xScale,
        yScale,
        xTicks: isLog ? [2000, 5000, 10000, 20000, 50000, 100000, 200000, 300000] : [0, 50000, 100000, 150000, 200000, 250000, 300000],
        yTicks: [0, 0.25, 0.5, 0.75, 1.0],
        xFormat: (v) => (v === 0 ? "0" : fmtSteps(v)),
        yFormat: (v) => fmtNum(v, 2),
        xLabel: `Training step (${isLog ? "log" : "linear"} scale)`,
        yLabel: "Memorisation fraction m",
        hLines: [
          { y: 0.5, color: T.muted, width: 1, dash: "3,3", opacity: 0.7, label: "onset (0.5)", align: "right" },
        ],
        vLines: [
          { x: 100000, color: T.muted, width: 1.1, dash: "4,3", opacity: 0.75, label: "100k \u2192 300k", dx: 6, dy: ph - 12 },
        ],
      });

      const nKeys = ["128", "256", "512"];
      const styleMap = {
        "128": { width: 1.8, dashPerm: "5,3", shape: "circle", baseOpacity: 0.78 },
        "256": { width: 2.5, dashPerm: "6,3", shape: "square", baseOpacity: 1.0 },
        "512": { width: 1.6, dashPerm: "2,2", shape: "triangle", baseOpacity: 0.58 },
      };

      const endLabels = [];
      for (const nk of nKeys) {
        const pair = cifar.trajs[nk];
        if (!pair) continue;
        const st = styleMap[nk];
        const focused = state.cifarFocusN === "all" || state.cifarFocusN === nk;
        const op = state.cifarFocusN === "all" ? st.baseOpacity : focused ? 1.0 : 0.16;

        const natPts = pair.nat.map(([s, v]) => [xScale(s), yScale(v)]);
        const permPts = pair.perm.map(([s, v]) => [xScale(s), yScale(v)]);

        if (state.cifarFocusN === nk) {
          drawAreaBetween(svg, natPts, permPts, T.perm, 0.12);
        }

        drawGuidePath(svg, natPts, T.nat, null, focused ? st.width : 1.4, op);
        drawGuidePath(svg, permPts, T.perm, st.dashPerm, focused ? st.width : 1.4, op);

        if (focused) {
          for (const [s, v] of pair.nat) {
            svg.appendChild(drawMarker(svg, xScale(s), yScale(v), st.shape, nk === "256" ? 3.2 : 2.6, T.nat, T.surface));
          }
          for (const [s, v] of pair.perm) {
            svg.appendChild(drawMarker(svg, xScale(s), yScale(v), st.shape, nk === "256" ? 3.2 : 2.6, T.perm, T.surface));
          }
        }

        const lastNat = pair.nat[pair.nat.length - 1];
        const lastPerm = pair.perm[pair.perm.length - 1];
        if (focused) {
          endLabels.push({ y: yScale(lastNat[1]) + 3, color: T.nat, op, text: `n=${nk}` });
          endLabels.push({ y: yScale(lastPerm[1]) + 3, color: T.perm, op, text: `n=${nk}` });
        }
      }
      // Deconflict overlapping 300k right-edge labels cleanly
      endLabels.sort((a, b) => a.y - b.y);
      for (let i = 1; i < endLabels.length; i++) {
        if (endLabels[i].y - endLabels[i - 1].y < 12) {
          endLabels[i].y = endLabels[i - 1].y + 12;
        }
      }
      for (const lbl of endLabels) {
        svg.appendChild(svgEl("text", {
          x: xScale(300000) + 6,
          y: lbl.y,
          fill: lbl.color,
          "font-family": T.mono,
          "font-size": 10,
          "font-weight": 600,
          opacity: lbl.op,
          text: lbl.text,
        }));
      }

      // Hover crosshair across timestamps
      const allSteps = (cifar.trajs["256"] && cifar.trajs["256"].nat.map((p) => p[0])) || [];
      const hoverG = addHoverLayer(svg);
      addTrapRect(
        svg,
        m,
        pw,
        ph,
        (e) => {
          const rect = svg.getBoundingClientRect();
          const mx = ((e.clientX - rect.left) / rect.width) * w;
          let bestStep = allSteps[0];
          let bestDist = Infinity;
          for (const s of allSteps) {
            const dist = Math.abs(xScale(s) - mx);
            if (dist < bestDist) {
              bestDist = dist;
              bestStep = s;
            }
          }
          clear(hoverG);
          const hx = xScale(bestStep);
          hoverG.appendChild(svgEl("line", { x1: hx, x2: hx, y1: m.t, y2: m.t + ph, stroke: T.muted, "stroke-width": 1, "stroke-dasharray": "3,3" }));

          let html = `<div class="viz-tooltip__title">CIFAR-10 \u00b7 T = ${fmtSteps(bestStep)}</div>`;
          const activeKeys = state.cifarFocusN === "all" ? nKeys : [state.cifarFocusN];
          for (const nk of activeKeys) {
            const pair = cifar.trajs[nk];
            const nr = pair.nat.find((p) => p[0] === bestStep);
            const pr = pair.perm.find((p) => p[0] === bestStep);
            if (nr && pr) {
              hoverG.appendChild(svgEl("circle", { cx: hx, cy: yScale(nr[1]), r: 5, fill: "none", stroke: T.nat, "stroke-width": 1.8 }));
              hoverG.appendChild(svgEl("circle", { cx: hx, cy: yScale(pr[1]), r: 5, fill: "none", stroke: T.perm, "stroke-width": 1.8 }));
              html += tipRow(`n=${nk} Nat / Perm`, `${fmtNum(nr[1], 3)} / ${fmtNum(pr[1], 3)} (gap ${fmtNum(nr[1] - pr[1], 3)})`, nk === "256" ? T.perm : T.nat);
            }
          }
          tooltip.show(e.clientX, e.clientY, html);
        },
        () => {
          clear(hoverG);
          tooltip.hide();
        }
      );

      clear(cifarTrajPlot);
      cifarTrajPlot.appendChild(svg);

      clear(cifarTrajLegend);
      cifarTrajLegend.appendChild(buildLegend([
        { label: "Natural CIFAR-10", color: T.nat },
        { label: "Permuted CIFAR-10 (n=128/256/512)", color: T.perm, dashed: true },
      ]));

      const p128 = (cifar.per_n_300k && cifar.per_n_300k["128"]) || {};
      const p256 = (cifar.per_n_300k && cifar.per_n_300k["256"]) || {};
      renderStatStrip(cifarTrajStats, [
        { label: "n=128 @ 300k", value: `${fmtNum(p128.m_nat_300k, 2)} vs ${fmtNum(p128.m_perm_300k, 3)}`, color: T.nat },
        { label: "n=128 gap", value: fmtNum(p128.gap, 3) },
        { label: "n=256 @ 300k", value: `${fmtNum(p256.m_nat_300k, 3)} vs ${fmtNum(p256.m_perm_300k, 3)}`, color: T.perm },
        { label: "n=256 plateau gap", value: fmtNum(p256.gap, 3), color: T.perm },
      ]);
    }

    // ---- Draw Tab 2 Right Top: CIFAR-10 m(n) at 100k inset ----
    function drawCifarMn() {
      const w = cifarMnPlot.clientWidth;
      if (!w || !cifar.m_nat_100k) return;
      const T = K.theme();
      const h = 166;
      const m = { l: 44, r: 18, t: 16, b: 34 };
      const pw = w - m.l - m.r;
      const ph = h - m.t - m.b;

      const ns = Object.keys(cifar.m_nat_100k).map(Number).sort((a, b) => a - b);
      const xScale = scaleLog([105, 2450], [m.l, m.l + pw], 2);
      const yScale = scaleLinear([0, 1.05], [m.t + ph, m.t]);

      const svg = makeSvg(w, h);
      const nStarNat = Math.pow(2, cifar.log2_nstar_nat_100k);
      const nStarPerm = Math.pow(2, cifar.log2_nstar_perm_100k);

      drawAxes(svg, {
        w,
        h,
        m,
        xScale,
        yScale,
        xTicks: ns,
        yTicks: [0, 0.5, 1.0],
        xFormat: (v) => String(v),
        yFormat: (v) => fmtNum(v, 1),
        xLabel: "Dataset size n (100k steps, log\u2082 scale)",
        yLabel: "m(100k)",
        hLines: [
          { y: 0.5, color: T.muted, width: 1, dash: "3,3", opacity: 0.65 },
        ],
        vLines: [
          { x: nStarNat, color: T.nat, width: 1.2, dash: "3,3", opacity: 0.85 },
          { x: nStarPerm, color: T.perm, width: 1.2, dash: "3,3", opacity: 0.85 },
        ],
      });

      const natPts = ns.map((nv) => [xScale(nv), yScale(cifar.m_nat_100k[String(nv)])]);
      const permPts = ns.map((nv) => [xScale(nv), yScale(cifar.m_perm_100k[String(nv)])]);
      drawGuidePath(svg, natPts, T.nat, null, 2.2, 0.96);
      drawGuidePath(svg, permPts, T.perm, "5,3", 2.2, 0.96);

      for (const nv of ns) {
        svg.appendChild(drawMarker(svg, xScale(nv), yScale(cifar.m_nat_100k[String(nv)]), "circle", 3.3, T.nat, T.surface));
        svg.appendChild(drawMarker(svg, xScale(nv), yScale(cifar.m_perm_100k[String(nv)]), "square", 3.2, T.perm, T.surface));
      }

      // Annotate Delta = -0.48 between n*_perm and n*_nat at y = 0.5
      const xP = xScale(nStarPerm);
      const xN = xScale(nStarNat);
      const yMid = yScale(0.5);
      svg.appendChild(svgEl("line", {
        x1: xN,
        x2: xP,
        y1: yMid,
        y2: yMid,
        stroke: T.text,
        "stroke-width": 1.6,
      }));
      svg.appendChild(svgEl("text", {
        x: xN + 6,
        y: yMid - 8,
        fill: T.text,
        "font-family": T.mono,
        "font-size": 10.5,
        "font-weight": 600,
        text: `\u0394 = ${fmtSigned(cifar.delta_100k, 2)} (\u03c1=${fmtNum(cifar.rho_mean, 2)})`,
      }));

      const hoverG = addHoverLayer(svg);
      addTrapRect(
        svg,
        m,
        pw,
        ph,
        (e) => {
          const rect = svg.getBoundingClientRect();
          const mx = ((e.clientX - rect.left) / rect.width) * w;
          let bestN = ns[0];
          let bestDist = Infinity;
          for (const nv of ns) {
            const dist = Math.abs(xScale(nv) - mx);
            if (dist < bestDist) {
              bestDist = dist;
              bestN = nv;
            }
          }
          clear(hoverG);
          const hx = xScale(bestN);
          const vn = cifar.m_nat_100k[String(bestN)];
          const vp = cifar.m_perm_100k[String(bestN)];
          hoverG.appendChild(svgEl("line", { x1: hx, x2: hx, y1: m.t, y2: m.t + ph, stroke: T.muted, "stroke-width": 1, "stroke-dasharray": "3,3" }));
          hoverG.appendChild(svgEl("circle", { cx: hx, cy: yScale(vn), r: 5.5, fill: "none", stroke: T.nat, "stroke-width": 1.8 }));
          hoverG.appendChild(svgEl("circle", { cx: hx, cy: yScale(vp), r: 5.5, fill: "none", stroke: T.perm, "stroke-width": 1.8 }));
          let html = `<div class="viz-tooltip__title">CIFAR-10 @ 100k \u00b7 n = ${bestN}</div>`;
          html += tipRow("Natural m(100k)", fmtNum(vn, 3), T.nat);
          html += tipRow("Permuted m(100k)", fmtNum(vp, 3), T.perm);
          html += tipRow("log\u2082 n* (Nat / Perm)", `${fmtNum(cifar.log2_nstar_nat_100k, 2)} / ${fmtNum(cifar.log2_nstar_perm_100k, 2)}`);
          tooltip.show(e.clientX, e.clientY, html);
        },
        () => {
          clear(hoverG);
          tooltip.hide();
        }
      );

      clear(cifarMnPlot);
      cifarMnPlot.appendChild(svg);

      clear(cifarMnLegend);
      cifarMnLegend.appendChild(buildLegend([
        { label: `Natural (log\u2082 n* = ${fmtNum(cifar.log2_nstar_nat_100k, 2)})`, color: T.nat },
        { label: `Permuted (log\u2082 n* = ${fmtNum(cifar.log2_nstar_perm_100k, 2)})`, color: T.perm, dashed: true },
      ]));
    }

    syncTopTabs();
    selectArch(state.selectedArchId);
    drawWidth();

    mountResponsive(widthPlot, drawWidth);
    mountResponsive(dialPlot, drawDial);
    mountResponsive(archInsetPlot, drawArchInset);
    mountResponsive(cifarTrajPlot, drawCifarTraj);
    mountResponsive(cifarMnPlot, drawCifarMn);
  }

  window.DiffMemWidgets = window.DiffMemWidgets || {};
  window.DiffMemWidgets.mountDeltaRho = mountDeltaRho;
  window.DiffMemWidgets.mountScaleArch = mountScaleArch;
})();
