/* Interactive Widget: Pure-Locality Network-Free Square-Patch LS Model */
(function () {
  "use strict";

  const L_LEVELS = [1, 3, 5, 7, 9, 13, 55];
  const L_LABELS = {
    1: "1×1",
    3: "3×3",
    5: "5×5",
    7: "7×7",
    9: "9×9",
    13: "13×13",
    55: "Global (55)",
  };
  const GRID_N = 28;
  const CENTER_RC = 14;
  const PURE_NOISE_T = 958;

  function esc(str) {
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function findCrossing(rows, key) {
    if (!rows.length) return { found: false, n: 2, log2n: 1 };
    if (rows[0][key] < 0.5) {
      return { found: false, n: rows[0].n, log2n: Math.log2(rows[0].n) };
    }
    for (let i = 0; i < rows.length - 1; i++) {
      const a = rows[i];
      const b = rows[i + 1];
      const ya = a[key];
      const yb = b[key];
      if (ya >= 0.5 && yb <= 0.5 && ya !== yb) {
        const frac = (0.5 - ya) / (yb - ya);
        const log2n = Math.log2(a.n) + frac * (Math.log2(b.n) - Math.log2(a.n));
        return { found: true, n: Math.pow(2, log2n), log2n };
      }
    }
    const last = rows[rows.length - 1];
    return { found: false, n: last.n, log2n: Math.log2(last.n) };
  }

  function formatNStar(cross) {
    if (!cross.found) return `< ${cross.n}`;
    return cross.n >= 100 ? Math.round(cross.n).toLocaleString() : cross.n.toFixed(1);
  }

  function getUNetNStar(data, thresh) {
    const rep = data && data.mnist && data.mnist.unet && data.mnist.unet.rep0;
    const series = rep ? (thresh === "13" ? rep.delta_1_3 : rep.delta_1_2) : null;
    if (Array.isArray(series) && series.length) {
      const last = series[series.length - 1];
      if (last && typeof last.log2_nstar_nat === "number") {
        return Math.pow(2, last.log2_nstar_nat);
      }
    }
    return thresh === "13" ? 2740 : 3110;
  }

  function getRegime(t, L) {
    if (t >= PURE_NOISE_T) {
      return {
        label: "Pure noise (L_ref → 1)",
        badge: "Pure noise · L→1 fallback",
        isPureNoise: true,
        note: "At t ≥ 958, SNR is near zero so patches of any size L carry no signal; L_ref defaults back to 1.",
      };
    }
    if (L === 1) {
      return {
        label: "Low noise: pixel-local",
        badge: "Low noise: pixel-local",
        isPureNoise: false,
        note: "Fine pixel detail dominates; a 1×1 receptive field suffices for natural and permuted images alike.",
      };
    }
    if (L <= 13) {
      const px = L * L;
      return {
        label: "Mid noise: local patch",
        badge: `Mid noise: ${L}×${L} local`,
        isPureNoise: false,
        note: `Natural ${L}×${L} square pools correlated strokes; under π, those ${px} pixels scatter across the grid.`,
      };
    }
    return {
      label: "High noise: global",
      badge: "High noise: global",
      isPureNoise: false,
      note: "L = 55 covers the full 28×28 image (784 px) from any center to integrate global signal over heavy noise.",
    };
  }

  function computePatchSets(L, perm784) {
    const half = (L - 1) >> 1;
    const natSet = new Set();
    const permSet = new Set();
    const rMin = Math.max(0, CENTER_RC - half);
    const rMax = Math.min(GRID_N - 1, CENTER_RC + half);
    const cMin = Math.max(0, CENTER_RC - half);
    const cMax = Math.min(GRID_N - 1, CENTER_RC + half);

    for (let r = rMin; r <= rMax; r++) {
      for (let c = cMin; c <= cMax; c++) {
        const idx = r * GRID_N + c;
        natSet.add(idx);
        const mapped = perm784 && perm784.length === GRID_N * GRID_N ? perm784[idx] : idx;
        permSet.add(mapped);
      }
    }
    const centerNat = CENTER_RC * GRID_N + CENTER_RC;
    const centerPerm = perm784 && perm784.length === GRID_N * GRID_N ? perm784[centerNat] : centerNat;
    return {
      natSet,
      permSet,
      centerNat,
      centerPerm,
      bounds: { rMin, rMax, cMin, cMax },
      count: natSet.size,
    };
  }

  function drawPatchCanvas(canvas, activeSet, centerIdx, fillColor, boundsOrNull, T, isMuted) {
    const size = 84;
    const cell = size / GRID_N; // exact 3px per cell
    const dpr = Math.max(1, Math.min(3, window.devicePixelRatio || 1));
    const targetW = Math.round(size * dpr);
    const targetH = Math.round(size * dpr);
    if (canvas.width !== targetW || canvas.height !== targetH) {
      canvas.width = targetW;
      canvas.height = targetH;
      canvas.style.width = `${size}px`;
      canvas.style.height = `${size}px`;
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.imageSmoothingEnabled = false;

    ctx.fillStyle = T.cardAlt;
    ctx.fillRect(0, 0, size, size);

    ctx.fillStyle = isMuted ? T.muted : fillColor;
    ctx.globalAlpha = isMuted ? 0.55 : 0.88;
    for (const idx of activeSet) {
      const r = Math.floor(idx / GRID_N);
      const c = idx % GRID_N;
      ctx.fillRect(c * cell, r * cell, cell, cell);
    }
    ctx.globalAlpha = 1;

    // Subtle pixel lattice so 28x28 cells remain distinct even at L = 55
    ctx.strokeStyle = T.surface;
    ctx.globalAlpha = 0.28;
    ctx.lineWidth = 1 / dpr;
    ctx.beginPath();
    for (let i = 1; i < GRID_N; i++) {
      const p = i * cell;
      ctx.moveTo(p, 0);
      ctx.lineTo(p, size);
      ctx.moveTo(0, p);
      ctx.lineTo(size, p);
    }
    ctx.stroke();
    ctx.globalAlpha = 1;

    // Crisp outer patch outline on natural view
    if (boundsOrNull && activeSet.size < GRID_N * GRID_N) {
      const { rMin, rMax, cMin, cMax } = boundsOrNull;
      ctx.strokeStyle = T.text;
      ctx.globalAlpha = 0.65;
      ctx.lineWidth = 1;
      ctx.strokeRect(
        cMin * cell + 0.5,
        rMin * cell + 0.5,
        (cMax - cMin + 1) * cell - 1,
        (rMax - rMin + 1) * cell - 1
      );
      ctx.globalAlpha = 1;
    }

    // Highlight center pixel (14,14) and its mapped image under pi
    const cr = Math.floor(centerIdx / GRID_N);
    const cc = centerIdx % GRID_N;
    ctx.fillStyle = T.gold;
    ctx.fillRect(cc * cell, cr * cell, cell, cell);
    ctx.strokeStyle = T.surface;
    ctx.lineWidth = 0.75;
    ctx.strokeRect(cc * cell + 0.35, cr * cell + 0.35, cell - 0.7, cell - 0.7);
  }

  function arrowPolygon(tipX, tipY, dirX, size) {
    // Horizontal arrowhead pointing in sign(dirX)
    const s = dirX >= 0 ? 1 : -1;
    const h = size * 0.62;
    return `${tipX},${tipY} ${tipX - s * size},${tipY - h} ${tipX - s * size},${tipY + h}`;
  }

  function mountLS(container, data) {
    const K = window.DiffMemKit;
    if (!K) throw new Error("DiffMemKit must load before widget_ls_metric.js");
    const ls = data && data.ls;
    if (!ls || !Array.isArray(ls.rows) || !Array.isArray(ls.L_schedule)) {
      throw new Error("Missing data.ls.rows or data.ls.L_schedule");
    }

    const rows = ls.rows;
    const schedule = ls.L_schedule;
    const perm784 = data.hero && Array.isArray(data.hero.perm784) ? data.hero.perm784 : null;

    // Default hovered timestep: t = 449 (L = 7)
    let defaultIdx = schedule.findIndex((pair) => pair[0] === 449);
    if (defaultIdx < 0) defaultIdx = Math.floor(schedule.length * 0.45);

    const state = {
      thresh: "13",
      showWiener: true,
      activeSchedIdx: defaultIdx,
    };

    // ---------- 1. Build static DOM scaffold ONCE ----------
    K.clear(container);

    const refBadge = K.el("span", {
      class: "viz-badge",
      text: `L(t) calibrated at n_ref = ${(ls.L_ref_n || 10000).toLocaleString()}`,
    });

    const header = K.el("div", { class: "viz-header" }, [
      K.el("div", {}, [
        K.el("h3", {
          class: "viz-title",
          text: "What Pure Locality Predicts: The Network-Free Square-Patch LS Model",
        }),
        K.el("p", {
          class: "viz-subtitle",
          text: "Comparing the fixed-scale square-patch local-score (LS) model against the permutation-covariant data-statistics null (Δ = 0).",
        }),
      ]),
      refBadge,
    ]);

    const threshSeg = K.makeSeg(
      [
        { value: "13", label: "d₁/d₂ < 1/3 (primary)" },
        { value: "12", label: "d₁/d₂ < 1/2 (secondary)" },
      ],
      state.thresh,
      (val) => {
        state.thresh = val;
        redrawLeft();
      }
    );

    const wienerInput = K.el("input", { type: "checkbox" });
    wienerInput.checked = state.showWiener;
    wienerInput.addEventListener("change", () => {
      state.showWiener = wienerInput.checked;
      redrawLeft();
    });

    const wienerCheck = K.el("label", { class: "viz-check" }, [
      wienerInput,
      K.el("span", { text: "Show Wiener-mask / Data-statistics prediction (Δ = 0)" }),
    ]);

    const controls = K.el("div", { class: "viz-controls" }, [
      K.el("div", { class: "viz-control-group" }, [
        K.el("span", { class: "viz-control-label", text: "Threshold" }),
        threshSeg,
      ]),
      wienerCheck,
    ]);

    // Left Panel DOM
    const leftBadge = K.el("span", { class: "viz-badge viz-badge--perm", text: "" });
    const leftPlotWrap = K.el("div");
    const leftLegend = K.el("div", { class: "viz-legend" });
    const leftPanel = K.el("div", { class: "viz-panel" }, [
      K.el("div", { class: "viz-panel-title" }, [
        K.el("span", { text: "Memorised fraction vs. n (LS Model)" }),
        leftBadge,
      ]),
      leftPlotWrap,
      leftLegend,
    ]);

    // Right Panel DOM: Step chart + Interactive 28x28 Patch Footprint Inset
    const rightBadge = K.el("span", { class: "viz-badge viz-badge--nat", text: "" });
    const rightPlotWrap = K.el("div");

    const natCanvas = K.el("canvas", {
      width: 84,
      height: 84,
      "aria-label": "28 by 28 natural square patch footprint",
      style: "display:block;border-radius:4px;border:1px solid var(--border);",
    });
    const permCanvas = K.el("canvas", {
      width: 84,
      height: 84,
      "aria-label": "28 by 28 permuted patch pixel footprint",
      style: "display:block;border-radius:4px;border:1px solid var(--border);",
    });

    const insetTitleT = K.el("span", {
      style: "font-family:var(--mono);font-weight:700;font-variant-numeric:tabular-nums;",
    });
    const insetRegimePill = K.el("span", { class: "viz-badge" });
    const insetSizeStat = K.el("div", {
      style: "font-family:var(--mono);font-size:12.5px;font-weight:600;color:var(--text);font-variant-numeric:tabular-nums;",
    });
    const insetDesc = K.el("div", {
      style: "font-size:12px;color:var(--muted);line-height:1.38;margin-top:3px;",
    });

    // Quick preset regime buttons inside the inset for touch/click convenience
    const presetTargets = [
      { t: 61, label: "t=61 (1×1)" },
      { t: 449, label: "t=449 (7×7)" },
      { t: 816, label: "t=816 (Global)" },
      { t: 979, label: "t=979 (Noise)" },
    ];
    const presetWrap = K.el("div", {
      style: "display:flex;flex-wrap:wrap;gap:4px;margin-top:6px;",
    });
    const presetBtns = [];

    const insetBox = K.el(
      "div",
      {
        style:
          "margin-top:10px;background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:10px 12px;",
      },
      [
        K.el(
          "div",
          {
            style:
              "display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center;gap:6px;margin-bottom:8px;",
          },
          [
            K.el("div", { style: "font-size:12px;font-weight:600;color:var(--text);" }, [
              "28×28 Patch Footprint at ",
              insetTitleT,
            ]),
            insetRegimePill,
          ]
        ),
        K.el(
          "div",
          {
            style: "display:flex;flex-wrap:wrap;align-items:center;gap:12px;",
          },
          [
            K.el("div", { style: "display:flex;gap:8px;align-items:flex-start;flex-shrink:0;" }, [
              K.el("div", { style: "text-align:center;" }, [
                natCanvas,
                K.el("div", {
                  text: "Natural (L×L)",
                  style: "font-size:11px;font-weight:600;color:var(--viz-nat);margin-top:3px;",
                }),
              ]),
              K.el("div", { style: "text-align:center;" }, [
                permCanvas,
                K.el("div", {
                  text: "Permuted (π)",
                  style: "font-size:11px;font-weight:600;color:var(--viz-perm);margin-top:3px;",
                }),
              ]),
            ]),
            K.el("div", { style: "flex:1 1 165px;min-width:155px;" }, [
              insetSizeStat,
              insetDesc,
              presetWrap,
            ]),
          ]
        ),
      ]
    );

    const rightPanel = K.el("div", { class: "viz-panel" }, [
      K.el("div", { class: "viz-panel-title" }, [
        K.el("span", { text: "Calibrated Patch Size L(t) vs. Noise Level t" }),
        rightBadge,
      ]),
      rightPlotWrap,
      insetBox,
    ]);

    const grid = K.el("div", { class: "viz-grid-2" }, [leftPanel, rightPanel]);

    const figcaption = K.el("div", { class: "viz-figcaption" }, [
      K.el("b", { text: "Why LS predicts Δ_LS ≤ −3 while data-statistics predict Δ = 0: " }),
      "A fixed-scale square-patch local score (LS) relies on spatially neighboring pixels sharing strong correlations; under pixel permutation π, each L×L grid patch gathers ",
      K.el("b", { text: "L² scattered, uncorrelated pixels" }),
      " that disagree with adjacent patches, collapsing memorisation into un-copied mosaics (Δ_LS ≤ −3 octaves, at least an 8× drop). By contrast, a global Wiener-mask / data-covariance estimator is ",
      K.el("b", { text: "permutation-covariant" }),
      " and therefore predicts zero shift (Δ_data = 0), matching the Natural curve on both arms.",
    ]);

    container.appendChild(header);
    container.appendChild(controls);
    container.appendChild(grid);
    container.appendChild(figcaption);

    // References to live-updatable elements in Right SVG
    let rightLiveHandles = null;

    function syncPresetButtons(curT) {
      for (const item of presetBtns) {
        const isCurrent = item.t === curT;
        item.btn.style.borderColor = isCurrent ? "var(--accent)" : "var(--border)";
        item.btn.style.color = isCurrent ? "var(--accent)" : "var(--muted)";
        item.btn.style.fontWeight = isCurrent ? "600" : "500";
      }
    }

    for (const p of presetTargets) {
      const idx = schedule.findIndex((pair) => pair[0] === p.t);
      if (idx < 0) continue;
      const btn = K.el("button", {
        type: "button",
        text: p.label,
        style:
          "appearance:none;border:1px solid var(--border);background:var(--viz-card-alt);border-radius:5px;padding:2px 6px;font:500 10.5px/1.3 var(--mono);cursor:pointer;",
        onClick: () => {
          state.activeSchedIdx = idx;
          updateRightInteractive(false);
        },
      });
      presetBtns.push({ t: p.t, idx, btn });
      presetWrap.appendChild(btn);
    }

    // ---------- 2. Left Panel SVG Renderer ----------
    function renderLegendItem(label, color, dashed, shape) {
      const sw = K.svgEl("svg", {
        width: 22,
        height: 10,
        viewBox: "0 0 22 10",
        style: "flex-shrink:0;display:block;",
      });
      sw.appendChild(
        K.svgEl("line", {
          x1: 1,
          x2: 21,
          y1: 5,
          y2: 5,
          stroke: color,
          "stroke-width": 2.2,
          "stroke-dasharray": dashed ? "4,3" : null,
        })
      );
      if (shape === "circle") {
        sw.appendChild(K.svgEl("circle", { cx: 11, cy: 5, r: 3, fill: color }));
      } else if (shape === "square") {
        sw.appendChild(
          K.svgEl("rect", { x: 8.2, y: 2.2, width: 5.6, height: 5.6, rx: 1, fill: color })
        );
      }
      return K.el("span", { class: "viz-legend__item" }, [
        sw,
        K.el("span", { text: label }),
      ]);
    }

    function redrawLeft() {
      const T = K.theme();
      const natKey = state.thresh === "13" ? "nat_m13" : "nat_m12";
      const permKey = state.thresh === "13" ? "perm_m13" : "perm_m12";
      const tauStr = state.thresh === "13" ? "1/3" : "1/2";

      const natCross = findCrossing(rows, natKey);
      const permCross = findCrossing(rows, permKey);
      const unetNstar = getUNetNStar(data, state.thresh);
      const ratioToUnet = Math.round(unetNstar / natCross.n);

      let deltaText;
      if (permCross.found) {
        const dOct = permCross.log2n - natCross.log2n;
        const mult = Math.round(natCross.n / permCross.n);
        deltaText = `Δ_LS = ${dOct.toFixed(1)} oct (${mult}× drop)`;
      } else {
        const boundOct = permCross.log2n - natCross.log2n; // <= -3.06
        deltaText = `Δ_LS ≤ ${boundOct.toFixed(1)} oct (≥8× drop)`;
      }

      leftBadge.textContent = `n*_nat ≈ ${formatNStar(natCross)} · ${deltaText}`;

      // Rebuild legend cleanly
      K.clear(leftLegend);
      leftLegend.appendChild(renderLegendItem("Natural LS", T.nat, false, "circle"));
      leftLegend.appendChild(renderLegendItem("Permuted LS", T.perm, true, "square"));
      if (state.showWiener) {
        leftLegend.appendChild(
          renderLegendItem("Wiener / Data null (Δ = 0)", T.gold, true, null)
        );
      }

      const w = Math.max(260, leftPlotWrap.clientWidth || 420);
      const h = 330;
      const m = { l: 46, r: 16, t: 18, b: 40 };
      const pw = w - m.l - m.r;
      const ph = h - m.t - m.b;

      const xScale = K.scaleLog([2, 2048], [m.l, m.l + pw], 2);
      const yScale = K.scaleLinear([0, 1], [m.t + ph, m.t]);

      K.clear(leftPlotWrap);
      const svg = K.makeSvg(w, h);

      const allTicks = rows.map((r) => r.n);
      const xTicks = pw < 310 ? [2, 8, 32, 128, 512, 2048] : allTicks;

      K.drawAxes(svg, {
        w,
        h,
        m,
        xScale,
        yScale,
        xTicks,
        yTicks: [0, 0.25, 0.5, 0.75, 1.0],
        xFormat: (v) => String(v),
        yFormat: (v) => v.toFixed(2),
        xLabel: "Training set size n (log₂ scale)",
        yLabel: `Memorised fraction m (d₁/d₂ < ${tauStr})`,
        hLines: [
          {
            y: 0.5,
            color: T.muted,
            dash: "3,3",
            width: 1.2,
            opacity: 0.75,
            label: "m = 0.5",
            align: "right",
          },
        ],
      });

      const natPts = rows.map((r) => [xScale(r.n), yScale(r[natKey])]);
      const permPts = rows.map((r) => [xScale(r.n), yScale(r[permKey])]);

      // Subtle shaded gap between Natural and Permuted curves
      const areaPts = natPts.concat([...permPts].reverse());
      svg.appendChild(
        K.svgEl("polygon", {
          points: areaPts.map((p) => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(" "),
          fill: T.band,
          opacity: 0.55,
        })
      );

      // Curves
      svg.appendChild(
        K.svgEl("path", {
          d: K.linePath(permPts),
          fill: "none",
          stroke: T.perm,
          "stroke-width": 2.3,
          "stroke-dasharray": "6,4",
        })
      );

      svg.appendChild(
        K.svgEl("path", {
          d: K.linePath(natPts),
          fill: "none",
          stroke: T.nat,
          "stroke-width": state.showWiener ? 3.2 : 2.5,
        })
      );

      if (state.showWiener) {
        svg.appendChild(
          K.svgEl("path", {
            d: K.linePath(natPts),
            fill: "none",
            stroke: T.gold,
            "stroke-width": 2.3,
            "stroke-dasharray": "5,5",
            "stroke-linecap": "round",
          })
        );
      }

      // Data markers
      for (let i = 0; i < rows.length; i++) {
        svg.appendChild(
          K.drawMarker(svg, permPts[i][0], permPts[i][1], "square", 3.6, T.perm, T.surface)
        );
        svg.appendChild(
          K.drawMarker(svg, natPts[i][0], natPts[i][1], "circle", 4.0, T.nat, T.surface)
        );
      }

      // Crossing & horizontal shift annotation at m = 0.5
      const yHalf = yScale(0.5);
      const xNatStar = xScale(natCross.n);
      const xPermStar = permCross.found ? xScale(permCross.n) : xScale(2);

      // Vertical drop dashed guide from n*_nat
      svg.appendChild(
        K.svgEl("line", {
          x1: xNatStar,
          x2: xNatStar,
          y1: yHalf,
          y2: yScale(0),
          stroke: T.nat,
          "stroke-width": 1.2,
          "stroke-dasharray": "3,3",
          opacity: 0.75,
        })
      );
      svg.appendChild(
        K.svgEl("circle", {
          cx: xNatStar,
          cy: yHalf,
          r: 4.2,
          fill: T.surface,
          stroke: T.nat,
          "stroke-width": 2,
        })
      );

      if (permCross.found) {
        svg.appendChild(
          K.svgEl("line", {
            x1: xPermStar,
            x2: xPermStar,
            y1: yHalf,
            y2: yScale(0),
            stroke: T.perm,
            "stroke-width": 1.2,
            "stroke-dasharray": "3,3",
            opacity: 0.75,
          })
        );
        svg.appendChild(
          K.svgEl("circle", {
            cx: xPermStar,
            cy: yHalf,
            r: 4.2,
            fill: T.surface,
            stroke: T.perm,
            "stroke-width": 2,
          })
        );
      }

      // Horizontal shift arrow from xNatStar leftward to xPermStar
      svg.appendChild(
        K.svgEl("line", {
          x1: xNatStar - 5,
          x2: xPermStar + 6,
          y1: yHalf,
          y2: yHalf,
          stroke: T.perm,
          "stroke-width": 2,
        })
      );
      svg.appendChild(
        K.svgEl("polygon", {
          points: arrowPolygon(xPermStar + 2, yHalf, -1, 7),
          fill: T.perm,
        })
      );

      // Crossing value labels just above m = 0.5
      const crossFontSize = pw < 290 ? 9.5 : 10.5;
      svg.appendChild(
        K.svgEl("text", {
          x: xNatStar + 8,
          y: yHalf - 9,
          "text-anchor": "start",
          fill: T.nat,
          "font-family": T.mono,
          "font-size": crossFontSize,
          "font-weight": 700,
          style: `paint-order:stroke;stroke:${T.cardAlt};stroke-width:3.5px;stroke-linejoin:round;`,
          text: `n*_nat ≈ ${formatNStar(natCross)}`,
        })
      );

      const permLabelText = permCross.found
        ? `n*_perm ≈ ${formatNStar(permCross)}`
        : pw < 290
        ? "perm < 2"
        : "n*_perm < 2";
      svg.appendChild(
        K.svgEl("text", {
          x: xPermStar + 4,
          y: yHalf - 8,
          "text-anchor": "start",
          fill: T.perm,
          "font-family": T.mono,
          "font-size": crossFontSize,
          "font-weight": 700,
          text: permLabelText,
        })
      );

      // Pill label centered under the horizontal shift arrow
      const pillCx = Math.max(m.l + 58, (xPermStar + xNatStar) / 2);
      const pillText = permCross.found
        ? `Δ_LS = ${(permCross.log2n - natCross.log2n).toFixed(1)} oct (${Math.round(natCross.n / permCross.n)}×)`
        : "Δ_LS ≤ −3 oct (≥8× drop)";
      const pillW = Math.min(pw * 0.56, pillText.length * 6.4 + 14);
      svg.appendChild(
        K.svgEl("rect", {
          x: pillCx - pillW / 2,
          y: yHalf + 8,
          width: pillW,
          height: 19,
          rx: 5,
          fill: T.surface,
          stroke: T.perm,
          "stroke-width": 1,
          opacity: 0.95,
        })
      );
      svg.appendChild(
        K.svgEl("text", {
          x: pillCx,
          y: yHalf + 21,
          "text-anchor": "middle",
          fill: T.perm,
          "font-family": T.mono,
          "font-size": 10,
          "font-weight": 700,
          text: pillText,
        })
      );

      // Direct curve labels (right-aligned near n = 256..512 so they never clip)
      const anchor256 = rows[7]; // n = 256
      const ax256 = xScale(anchor256.n);
      const ay256 = yScale(anchor256[natKey]);
      const labelRightX = m.l + pw - 6;
      const labelTopY = ay256 - (state.showWiener ? 30 : 20);
      svg.appendChild(
        K.svgEl("text", {
          x: labelRightX,
          y: labelTopY,
          "text-anchor": "end",
          fill: T.nat,
          "font-family": T.sans,
          "font-size": 10.8,
          "font-weight": 700,
          text: "Natural LS",
        })
      );
      if (state.showWiener) {
        svg.appendChild(
          K.svgEl("text", {
            x: labelRightX,
            y: labelTopY + 13,
            "text-anchor": "end",
            fill: T.gold,
            "font-family": T.sans,
            "font-size": 10.2,
            "font-weight": 700,
            text: "= Wiener null (Δ = 0)",
          })
        );
      }
      svg.appendChild(
        K.svgEl("line", {
          x1: ax256 + 4,
          x2: Math.max(ax256 + 12, labelRightX - 78),
          y1: ay256 - 4,
          y2: labelTopY + (state.showWiener ? 8 : 3),
          stroke: T.nat,
          "stroke-width": 1,
          opacity: 0.65,
        })
      );

      svg.appendChild(
        K.svgEl("text", {
          x: xScale(32),
          y: yScale(0) - 10,
          "text-anchor": "start",
          fill: T.perm,
          "font-family": T.sans,
          "font-size": 10.8,
          "font-weight": 700,
          text: "Permuted LS",
        })
      );

      // Upper-right annotation comparing LS n* vs trained U-Net (n* ≈ 2,740)
      const boxW = Math.min(206, Math.max(152, pw * 0.49));
      const boxH = 54;
      const boxX = m.l + pw - boxW - 2;
      const boxY = m.t + 4;
      const unetDisplay = state.thresh === "13" ? "2,740" : Math.round(unetNstar).toLocaleString();
      const compact = boxW < 175;

      svg.appendChild(
        K.svgEl("rect", {
          x: boxX,
          y: boxY,
          width: boxW,
          height: boxH,
          rx: 6,
          fill: T.surface,
          stroke: T.border,
          "stroke-width": 1,
          opacity: 0.95,
        })
      );
      svg.appendChild(
        K.svgEl("text", {
          x: boxX + 8,
          y: boxY + 16,
          fill: T.text,
          "font-family": T.sans,
          "font-size": compact ? 9.8 : 10.6,
          "font-weight": 700,
          text: `Trained U-Net: n* ≈ ${unetDisplay} →`,
        })
      );
      svg.appendChild(
        K.svgEl("text", {
          x: boxX + 8,
          y: boxY + 31,
          fill: T.muted,
          "font-family": T.sans,
          "font-size": compact ? 9.3 : 10,
          text: `LS n* ≈ ${formatNStar(natCross)} is ~${ratioToUnet}× lower:`,
        })
      );
      svg.appendChild(
        K.svgEl("text", {
          x: boxX + 8,
          y: boxY + 45,
          fill: T.muted,
          "font-family": T.sans,
          "font-size": compact ? 9.3 : 10,
          text: compact ? "fixed patches lack multi-scale features" : "fixed patches lack learned features",
        })
      );

      // Hover crosshair & highlighted points
      const hoverLine = K.svgEl("line", {
        x1: 0,
        x2: 0,
        y1: m.t,
        y2: m.t + ph,
        stroke: T.axis,
        "stroke-width": 1.2,
        "stroke-dasharray": "2,2",
        visibility: "hidden",
      });
      const hoverNat = K.svgEl("circle", {
        cx: 0,
        cy: 0,
        r: 6.5,
        fill: T.nat,
        stroke: T.surface,
        "stroke-width": 2,
        visibility: "hidden",
      });
      const hoverPerm = K.svgEl("rect", {
        x: 0,
        y: 0,
        width: 11,
        height: 11,
        rx: 2,
        fill: T.perm,
        stroke: T.surface,
        "stroke-width": 2,
        visibility: "hidden",
      });
      svg.appendChild(hoverLine);
      svg.appendChild(hoverNat);
      svg.appendChild(hoverPerm);

      const hitRect = K.svgEl("rect", {
        x: m.l,
        y: m.t,
        width: pw,
        height: ph,
        fill: "transparent",
        style: "cursor:crosshair;",
      });

      function onMove(clientX, clientY) {
        const rect = svg.getBoundingClientRect();
        const svgX = ((clientX - rect.left) / (rect.width || 1)) * w;
        const logN = Math.log2(Math.max(2, Math.min(2048, xScale.invert(svgX))));
        const idx = Math.max(0, Math.min(rows.length - 1, Math.round(logN - 1)));
        const r = rows[idx];
        const px = natPts[idx][0];
        const pyNat = natPts[idx][1];
        const pyPerm = permPts[idx][1];

        hoverLine.setAttribute("x1", px);
        hoverLine.setAttribute("x2", px);
        hoverLine.setAttribute("visibility", "visible");

        hoverNat.setAttribute("cx", px);
        hoverNat.setAttribute("cy", pyNat);
        hoverNat.setAttribute("visibility", "visible");

        hoverPerm.setAttribute("x", px - 5.5);
        hoverPerm.setAttribute("y", pyPerm - 5.5);
        hoverPerm.setAttribute("visibility", "visible");

        const wienerRow = state.showWiener
          ? K.tipRow("Wiener / Data null (Δ = 0)", r[natKey].toFixed(3), T.gold)
          : "";

        const html =
          `<div class="viz-tooltip__title">n = ${esc(r.n.toLocaleString())} (log₂ n = ${Math.log2(r.n)})</div>` +
          K.tipRow(`Natural LS m (τ=${tauStr})`, r[natKey].toFixed(3), T.nat) +
          K.tipRow(`Permuted LS m (τ=${tauStr})`, r[permKey].toFixed(3), T.perm) +
          wienerRow +
          `<div style="margin-top:5px;padding-top:4px;border-top:1px solid rgba(255,255,255,0.14);font-size:11.5px;">` +
          K.tipRow("Median d₁/d₂ (Nat vs Perm)", `${r.nat_med_r.toFixed(3)} vs ${r.perm_med_r.toFixed(3)}`) +
          K.tipRow("Median d₁ (Nat vs Perm)", `${r.nat_d1.toFixed(2)} vs ${r.perm_d1.toFixed(2)}`) +
          `</div>`;

        K.tooltip.show(clientX, clientY, html);
      }

      hitRect.addEventListener("mousemove", (e) => onMove(e.clientX, e.clientY));
      hitRect.addEventListener("touchstart", (e) => {
        if (e.touches && e.touches[0]) {
          onMove(e.touches[0].clientX, e.touches[0].clientY);
        }
      }, { passive: true });
      hitRect.addEventListener("mouseleave", () => {
        hoverLine.setAttribute("visibility", "hidden");
        hoverNat.setAttribute("visibility", "hidden");
        hoverPerm.setAttribute("visibility", "hidden");
        K.tooltip.hide();
      });

      svg.appendChild(hitRect);
      leftPlotWrap.appendChild(svg);
    }

    // ---------- 3. Right Panel: Step chart + 28x28 Inset ----------
    function updateRightInteractive(showTip, clientX, clientY) {
      const T = K.theme();
      const idx = Math.max(0, Math.min(schedule.length - 1, state.activeSchedIdx));
      const [t, L] = schedule[idx];
      const regime = getRegime(t, L);
      const patch = computePatchSets(L, perm784);
      const pct = ((patch.count / (GRID_N * GRID_N)) * 100).toFixed(1);

      // Update panel header badge & inset readout
      rightBadge.textContent = `t = ${t} · ${L === 55 ? "Global (28×28)" : `${L}×${L}`}`;
      insetTitleT.textContent = `t = ${t}`;
      insetRegimePill.textContent = regime.badge;
      insetRegimePill.className = regime.isPureNoise
        ? "viz-badge viz-badge--warn"
        : L === 55
        ? "viz-badge viz-badge--perm"
        : "viz-badge viz-badge--nat";

      const lLabel = L === 55 ? "Global (L = 55 → full 28×28)" : `L = ${L}×${L}`;
      insetSizeStat.textContent = `${lLabel} · ${patch.count} / 784 px (${pct}%)`;
      insetDesc.textContent = regime.note;

      syncPresetButtons(t);

      // Render crisp DPR-scaled 28x28 canvases
      drawPatchCanvas(natCanvas, patch.natSet, patch.centerNat, T.nat, patch.bounds, T, regime.isPureNoise);
      drawPatchCanvas(permCanvas, patch.permSet, patch.centerPerm, T.perm, null, T, regime.isPureNoise);

      // Update SVG crosshair & active point in place
      if (rightLiveHandles) {
        const { xScale, yScale, hoverLine, hoverHoriz, hoverDot } = rightLiveHandles;
        const rank = Math.max(0, L_LEVELS.indexOf(L));
        const px = xScale(t);
        const py = yScale(rank);
        const dotColor = regime.isPureNoise ? T.muted : T.nat;

        hoverLine.setAttribute("x1", px);
        hoverLine.setAttribute("x2", px);
        hoverHoriz.setAttribute("y1", py);
        hoverHoriz.setAttribute("y2", py);
        hoverDot.setAttribute("cx", px);
        hoverDot.setAttribute("cy", py);
        hoverDot.setAttribute("fill", dotColor);
      }

      if (showTip && clientX !== undefined && clientY !== undefined) {
        const extraNote = regime.isPureNoise
          ? `<div style="margin-top:5px;padding-top:4px;border-top:1px solid rgba(255,255,255,0.14);font-size:11.5px;opacity:0.9;">Pure-noise regime (t ≥ 958): all patches are uninformative, so L_ref resets to 1.</div>`
          : "";
        const html =
          `<div class="viz-tooltip__title">Timestep t = ${t} / 999</div>` +
          K.tipRow("Calibrated patch L(t)", L === 55 ? "Global (55 → 28×28)" : `${L}×${L}`, regime.isPureNoise ? T.muted : T.nat) +
          K.tipRow("Footprint size", `${patch.count} / 784 px (${pct}%)`) +
          K.tipRow("Regime", regime.label) +
          extraNote;
        K.tooltip.show(clientX, clientY, html);
      }
    }

    function redrawRight() {
      const T = K.theme();
      const w = Math.max(260, rightPlotWrap.clientWidth || 420);
      const h = 212;
      const m = { l: 80, r: 14, t: 16, b: 36 };
      const pw = w - m.l - m.r;
      const ph = h - m.t - m.b;

      const xScale = K.scaleLinear([0, 999], [m.l, m.l + pw]);
      const yScale = K.scaleLinear([0, L_LEVELS.length - 1], [m.t + ph, m.t]);

      K.clear(rightPlotWrap);
      const svg = K.makeSvg(w, h);

      // Subtle regime shading: global zone [754, 958] and pure-noise fallback [958, 999]
      const xGlobalStart = xScale(754);
      const xNoiseStart = xScale(PURE_NOISE_T);
      const xEnd = xScale(999);

      svg.appendChild(
        K.svgEl("rect", {
          x: xGlobalStart,
          y: m.t,
          width: Math.max(0, xNoiseStart - xGlobalStart),
          height: ph,
          fill: T.natSoft,
          opacity: 0.45,
        })
      );
      svg.appendChild(
        K.svgEl("rect", {
          x: xNoiseStart,
          y: m.t,
          width: Math.max(0, xEnd - xNoiseStart),
          height: ph,
          fill: T.band,
        })
      );

      K.drawAxes(svg, {
        w,
        h,
        m,
        xScale,
        yScale,
        xTicks: pw < 300 ? [0, 250, 500, 750, 999] : [0, 200, 400, 600, 800, 999],
        yTicks: [0, 1, 2, 3, 4, 5, 6],
        xFormat: (v) => String(v),
        yFormat: (idx) => L_LABELS[L_LEVELS[idx]] || "",
        xLabel: "Diffusion timestep t (low noise → high noise)",
      });

      // Subtle axis-break tick marks between 13×13 (idx 5) and Global (55) (idx 6)
      const yBreak = (yScale(5) + yScale(6)) / 2;
      svg.appendChild(
        K.svgEl("rect", {
          x: m.l - 3,
          y: yBreak - 3,
          width: 6,
          height: 6,
          fill: T.cardAlt,
        })
      );
      svg.appendChild(
        K.svgEl("line", {
          x1: m.l - 4,
          x2: m.l + 4,
          y1: yBreak + 1,
          y2: yBreak - 4,
          stroke: T.axis,
          "stroke-width": 1.2,
        })
      );
      svg.appendChild(
        K.svgEl("line", {
          x1: m.l - 4,
          x2: m.l + 4,
          y1: yBreak + 4,
          y2: yBreak - 1,
          stroke: T.axis,
          "stroke-width": 1.2,
        })
      );

      // Build step-after path cleanly split into active [0..958) and pure-noise fallback [958..999]
      const splitIdx = schedule.findIndex((p) => p[0] >= PURE_NOISE_T);
      const activePts = splitIdx > 0 ? schedule.slice(0, splitIdx) : schedule;
      const noisePts = splitIdx > 0 ? schedule.slice(splitIdx) : [];

      function buildStepPath(pairs, extendEndT) {
        const cmds = [];
        for (let i = 0; i < pairs.length; i++) {
          const [ti, Li] = pairs[i];
          const xi = xScale(ti);
          const yi = yScale(Math.max(0, L_LEVELS.indexOf(Li)));
          if (i === 0) {
            cmds.push(`M${xi.toFixed(2)},${yi.toFixed(2)}`);
          } else {
            const prevY = yScale(Math.max(0, L_LEVELS.indexOf(pairs[i - 1][1])));
            cmds.push(`L${xi.toFixed(2)},${prevY.toFixed(2)}`);
            cmds.push(`L${xi.toFixed(2)},${yi.toFixed(2)}`);
          }
        }
        if (extendEndT !== undefined && pairs.length) {
          const lastL = pairs[pairs.length - 1][1];
          const lastY = yScale(Math.max(0, L_LEVELS.indexOf(lastL)));
          cmds.push(`L${xScale(extendEndT).toFixed(2)},${lastY.toFixed(2)}`);
        }
        return cmds.join(" ");
      }

      const activeEndT = noisePts.length ? noisePts[0][0] : 999;
      const activeStepD = buildStepPath(activePts, activeEndT);
      const yBase = yScale(0);
      const areaD =
        `${activeStepD} L${xScale(activeEndT).toFixed(2)},${yBase.toFixed(2)} ` +
        `L${xScale(activePts[0][0]).toFixed(2)},${yBase.toFixed(2)} Z`;

      svg.appendChild(
        K.svgEl("path", {
          d: areaD,
          fill: T.natSoft,
        })
      );

      svg.appendChild(
        K.svgEl("path", {
          d: activeStepD,
          fill: "none",
          stroke: T.nat,
          "stroke-width": 2.4,
        })
      );

      if (noisePts.length) {
        const lastActiveY = yScale(Math.max(0, L_LEVELS.indexOf(activePts[activePts.length - 1][1])));
        const firstNoiseY = yScale(Math.max(0, L_LEVELS.indexOf(noisePts[0][1])));
        // Dashed drop at t = 958 and muted dashed floor [958, 999]
        svg.appendChild(
          K.svgEl("line", {
            x1: xNoiseStart,
            x2: xNoiseStart,
            y1: lastActiveY,
            y2: firstNoiseY,
            stroke: T.muted,
            "stroke-width": 1.6,
            "stroke-dasharray": "3,3",
          })
        );
        svg.appendChild(
          K.svgEl("path", {
            d: buildStepPath(noisePts, 999),
            fill: "none",
            stroke: T.muted,
            "stroke-width": 2.1,
            "stroke-dasharray": "4,3",
          })
        );

        // Annotation explaining pure-noise reset at t >= 958 (placed in clear mid-right whitespace)
        const noiseNoteY = yScale(3.6);
        svg.appendChild(
          K.svgEl("text", {
            x: xNoiseStart - 10,
            y: noiseNoteY,
            "text-anchor": "end",
            fill: T.muted,
            "font-family": T.sans,
            "font-size": 10,
            "font-weight": 600,
            style: `paint-order:stroke;stroke:${T.cardAlt};stroke-width:3.5px;stroke-linejoin:round;`,
            text: "t ≥ 958: pure noise",
          })
        );
        svg.appendChild(
          K.svgEl("text", {
            x: xNoiseStart - 10,
            y: noiseNoteY + 12,
            "text-anchor": "end",
            fill: T.muted,
            "font-family": T.sans,
            "font-size": 10,
            style: `paint-order:stroke;stroke:${T.cardAlt};stroke-width:3.5px;stroke-linejoin:round;`,
            text: "→ L_ref resets to 1",
          })
        );
        svg.appendChild(
          K.svgEl("line", {
            x1: xNoiseStart - 6,
            x2: xNoiseStart - 1,
            y1: noiseNoteY + 8,
            y2: yScale(0.35),
            stroke: T.muted,
            "stroke-width": 1,
            opacity: 0.75,
          })
        );
      }

      // Subtle regime label inside global band (top area, clear of drop)
      if (xNoiseStart - xGlobalStart > 38) {
        svg.appendChild(
          K.svgEl("text", {
            x: (xGlobalStart + xNoiseStart) / 2,
            y: yScale(6) + 15,
            "text-anchor": "middle",
            fill: T.nat,
            "font-family": T.sans,
            "font-size": 10,
            "font-weight": 600,
            text: "Global",
          })
        );
      }

      // Small sample points along step schedule
      for (const [ti, Li] of schedule) {
        const rank = Math.max(0, L_LEVELS.indexOf(Li));
        svg.appendChild(
          K.svgEl("circle", {
            cx: xScale(ti),
            cy: yScale(rank),
            r: 2.1,
            fill: ti >= PURE_NOISE_T ? T.muted : T.nat,
            opacity: 0.7,
          })
        );
      }

      // Live crosshair + active point marker
      const hoverLine = K.svgEl("line", {
        x1: 0,
        x2: 0,
        y1: m.t,
        y2: m.t + ph,
        stroke: T.axis,
        "stroke-width": 1.2,
        "stroke-dasharray": "2,2",
      });
      const hoverHoriz = K.svgEl("line", {
        x1: m.l,
        x2: m.l + pw,
        y1: 0,
        y2: 0,
        stroke: T.axis,
        "stroke-width": 1,
        "stroke-dasharray": "2,2",
        opacity: 0.7,
      });
      const hoverDot = K.svgEl("circle", {
        cx: 0,
        cy: 0,
        r: 5.5,
        fill: T.nat,
        stroke: T.surface,
        "stroke-width": 2,
      });

      svg.appendChild(hoverLine);
      svg.appendChild(hoverHoriz);
      svg.appendChild(hoverDot);

      rightLiveHandles = { xScale, yScale, hoverLine, hoverHoriz, hoverDot };

      const hitRect = K.svgEl("rect", {
        x: m.l,
        y: m.t,
        width: pw,
        height: ph,
        fill: "transparent",
        style: "cursor:crosshair;",
      });

      function snapSchedule(clientX) {
        const rect = svg.getBoundingClientRect();
        const svgX = ((clientX - rect.left) / (rect.width || 1)) * w;
        const targetT = Math.max(0, Math.min(999, xScale.invert(svgX)));
        let bestIdx = 0;
        let bestDist = Infinity;
        for (let i = 0; i < schedule.length; i++) {
          const dist = Math.abs(schedule[i][0] - targetT);
          if (dist < bestDist) {
            bestDist = dist;
            bestIdx = i;
          }
        }
        return bestIdx;
      }

      hitRect.addEventListener("mousemove", (e) => {
        state.activeSchedIdx = snapSchedule(e.clientX);
        updateRightInteractive(true, e.clientX, e.clientY);
      });
      hitRect.addEventListener("click", (e) => {
        state.activeSchedIdx = snapSchedule(e.clientX);
        updateRightInteractive(true, e.clientX, e.clientY);
      });
      hitRect.addEventListener(
        "touchstart",
        (e) => {
          if (e.touches && e.touches[0]) {
            const touch = e.touches[0];
            state.activeSchedIdx = snapSchedule(touch.clientX);
            updateRightInteractive(true, touch.clientX, touch.clientY);
          }
        },
        { passive: true }
      );
      hitRect.addEventListener("mouseleave", () => {
        K.tooltip.hide();
      });

      svg.appendChild(hitRect);
      rightPlotWrap.appendChild(svg);

      updateRightInteractive(false);
    }

    function redrawAll() {
      redrawLeft();
      redrawRight();
    }

    redrawAll();
    K.mountResponsive(container, redrawAll);
  }

  window.DiffMemWidgets = window.DiffMemWidgets || {};
  window.DiffMemWidgets.mountLS = mountLS;
})();
