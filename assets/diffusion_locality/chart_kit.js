/* Shared responsive SVG/Canvas charting utilities, tooltip, and citation popover */
(function () {
  "use strict";

  const SVG_NS = "http://www.w3.org/2000/svg";

  function theme() {
    const cs = getComputedStyle(document.documentElement);
    const g = (k, fb) => (cs.getPropertyValue(k) || "").trim() || fb;
    return {
      bg: g("--bg", "#faf7f0"),
      surface: g("--surface", "#ffffff"),
      cardAlt: g("--viz-card-alt", "#f6f2e8"),
      text: g("--text", "#1f2328"),
      muted: g("--muted", "#5f6670"),
      border: g("--border", "#e6dfd1"),
      accent: g("--accent", "#c2551f"),
      link: g("--link", "#22706b"),
      nat: g("--viz-nat", "#22706b"),
      natSoft: g("--viz-nat-soft", "rgba(34,112,107,0.14)"),
      perm: g("--viz-perm", "#c2551f"),
      permSoft: g("--viz-perm-soft", "rgba(194,85,31,0.15)"),
      mn: g("--viz-mn", "#1a8d63"),
      fm: g("--viz-fm", "#5b4ac7"),
      gold: g("--viz-gold", "#b87e14"),
      grid: g("--viz-grid", "rgba(31,35,40,0.08)"),
      axis: g("--viz-axis", "rgba(31,35,40,0.24)"),
      band: g("--viz-band", "rgba(95,102,112,0.10)"),
      sans: g("--sans", "Inter, system-ui, sans-serif"),
      mono: g("--mono", "ui-monospace, monospace"),
    };
  }

  function el(tag, attrs, children) {
    const node = document.createElement(tag);
    if (attrs) {
      for (const [k, v] of Object.entries(attrs)) {
        if (v === undefined || v === null) continue;
        if (k === "class") node.className = v;
        else if (k === "text") node.textContent = v;
        else if (k === "html") node.innerHTML = v;
        else if (k.startsWith("on") && typeof v === "function") node.addEventListener(k.slice(2).toLowerCase(), v);
        else node.setAttribute(k, String(v));
      }
    }
    if (children !== undefined) {
      const arr = Array.isArray(children) ? children : [children];
      for (const ch of arr) {
        if (ch === null || ch === undefined) continue;
        node.appendChild(typeof ch === "string" ? document.createTextNode(ch) : ch);
      }
    }
    return node;
  }

  function svgEl(tag, attrs, children) {
    const node = document.createElementNS(SVG_NS, tag);
    if (attrs) {
      for (const [k, v] of Object.entries(attrs)) {
        if (v === undefined || v === null) continue;
        if (k === "text") node.textContent = v;
        else node.setAttribute(k, String(v));
      }
    }
    if (children !== undefined) {
      const arr = Array.isArray(children) ? children : [children];
      for (const ch of arr) {
        if (ch === null || ch === undefined) continue;
        node.appendChild(typeof ch === "string" ? document.createTextNode(ch) : ch);
      }
    }
    return node;
  }

  function clear(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
  }

  function scaleLinear(domain, range) {
    const [d0, d1] = domain;
    const [r0, r1] = range;
    const span = d1 - d0 || 1;
    const fn = (v) => r0 + ((v - d0) / span) * (r1 - r0);
    fn.domain = domain;
    fn.range = range;
    fn.invert = (px) => d0 + ((px - r0) / ((r1 - r0) || 1)) * span;
    fn.ticks = (n = 5) => {
      const stepRaw = span / Math.max(1, n);
      const pow = Math.pow(10, Math.floor(Math.log10(Math.abs(stepRaw))));
      const err = stepRaw / pow;
      const mult = err >= 5 ? 5 : err >= 2 ? 2 : 1;
      const step = mult * pow;
      const start = Math.ceil(d0 / step) * step;
      const out = [];
      for (let v = start; v <= d1 + 1e-9 * Math.abs(step); v += step) {
        out.push(Number(v.toFixed(10)));
      }
      return out;
    };
    return fn;
  }

  function scaleLog(domain, range, base = 2) {
    const [d0, d1] = domain;
    const [r0, r1] = range;
    const lg = (x) => (base === 2 ? Math.log2(x) : Math.log10(x));
    const l0 = lg(d0);
    const l1 = lg(d1);
    const span = l1 - l0 || 1;
    const fn = (v) => r0 + ((lg(v) - l0) / span) * (r1 - r0);
    fn.domain = domain;
    fn.range = range;
    fn.invert = (px) => Math.pow(base, l0 + ((px - r0) / ((r1 - r0) || 1)) * span);
    return fn;
  }

  function makeSvg(w, h, cls = "") {
    return svgEl("svg", {
      viewBox: `0 0 ${w} ${h}`,
      width: "100%",
      height: h,
      class: cls,
      style: "display:block;overflow:visible;user-select:none;",
    });
  }

  function drawAxes(svg, opts) {
    const T = theme();
    const {
      w, h, m, xScale, yScale,
      xTicks = [], yTicks = [],
      xFormat = (v) => String(v),
      yFormat = (v) => String(v),
      xLabel = "", yLabel = "",
      hBands = [], hLines = [], vLines = [],
    } = opts;
    const pw = w - m.l - m.r;
    const ph = h - m.t - m.b;
    const g = svgEl("g");

    for (const b of hBands) {
      const yTop = yScale(b.y1);
      const yBot = yScale(b.y0);
      g.appendChild(svgEl("rect", {
        x: m.l, y: Math.min(yTop, yBot),
        width: pw, height: Math.abs(yBot - yTop),
        fill: b.fill || T.band,
      }));
      if (b.label) {
        g.appendChild(svgEl("text", {
          x: m.l + pw - 6, y: Math.min(yTop, yBot) + 12,
          "text-anchor": "end", fill: T.muted,
          "font-family": T.sans, "font-size": 10.5,
          text: b.label,
        }));
      }
    }

    for (const v of xTicks) {
      const x = xScale(v);
      g.appendChild(svgEl("line", {
        x1: x, x2: x, y1: m.t, y2: m.t + ph,
        stroke: T.grid, "stroke-width": 1,
      }));
      g.appendChild(svgEl("text", {
        x, y: m.t + ph + 16,
        "text-anchor": "middle", fill: T.muted,
        "font-family": T.mono, "font-size": 10.5,
        text: xFormat(v),
      }));
    }

    for (const v of yTicks) {
      const y = yScale(v);
      g.appendChild(svgEl("line", {
        x1: m.l, x2: m.l + pw, y1: y, y2: y,
        stroke: T.grid, "stroke-width": 1,
      }));
      g.appendChild(svgEl("text", {
        x: m.l - 7, y: y + 3.5,
        "text-anchor": "end", fill: T.muted,
        "font-family": T.mono, "font-size": 10.5,
        text: yFormat(v),
      }));
    }

    for (const hl of hLines) {
      const y = yScale(hl.y);
      g.appendChild(svgEl("line", {
        x1: m.l, x2: m.l + pw, y1: y, y2: y,
        stroke: hl.color || T.muted,
        "stroke-width": hl.width || 1.1,
        "stroke-dasharray": hl.dash || "4,3",
        opacity: hl.opacity ?? 0.8,
      }));
      if (hl.label) {
        g.appendChild(svgEl("text", {
          x: hl.align === "left" ? m.l + 6 : m.l + pw - 4,
          y: y - 5,
          "text-anchor": hl.align === "left" ? "start" : "end",
          fill: hl.color || T.muted,
          "font-family": T.sans, "font-size": 10.5, "font-weight": 600,
          text: hl.label,
        }));
      }
    }

    for (const vl of vLines) {
      const x = xScale(vl.x);
      g.appendChild(svgEl("line", {
        x1: x, x2: x, y1: m.t, y2: m.t + ph,
        stroke: vl.color || T.muted,
        "stroke-width": vl.width || 1.2,
        "stroke-dasharray": vl.dash || "4,3",
        opacity: vl.opacity ?? 0.85,
      }));
      if (vl.label) {
        g.appendChild(svgEl("text", {
          x: x + (vl.dx || 5), y: m.t + (vl.dy || 13),
          "text-anchor": vl.anchor || "start",
          fill: vl.color || T.muted,
          "font-family": T.sans, "font-size": 10.5, "font-weight": 600,
          text: vl.label,
        }));
      }
    }

    // Crisp L-frame axes
    g.appendChild(svgEl("line", { x1: m.l, x2: m.l + pw, y1: m.t + ph, y2: m.t + ph, stroke: T.axis, "stroke-width": 1.2 }));
    g.appendChild(svgEl("line", { x1: m.l, x2: m.l, y1: m.t, y2: m.t + ph, stroke: T.axis, "stroke-width": 1.2 }));

    if (xLabel) {
      g.appendChild(svgEl("text", {
        x: m.l + pw / 2, y: h - 5,
        "text-anchor": "middle", fill: T.muted,
        "font-family": T.sans, "font-size": 11.5, "font-weight": 500,
        text: xLabel,
      }));
    }
    if (yLabel) {
      g.appendChild(svgEl("text", {
        x: 13, y: m.t + ph / 2,
        transform: `rotate(-90 13 ${m.t + ph / 2})`,
        "text-anchor": "middle", fill: T.muted,
        "font-family": T.sans, "font-size": 11.5, "font-weight": 500,
        text: yLabel,
      }));
    }
    svg.appendChild(g);
    return g;
  }

  function linePath(pts) {
    if (!pts.length) return "";
    return pts.map((p, i) => `${i === 0 ? "M" : "L"}${p[0].toFixed(2)},${p[1].toFixed(2)}`).join(" ");
  }

  function drawMarker(svg, cx, cy, shape, r, fill, stroke = "#ffffff") {
    if (shape === "square") {
      return svgEl("rect", {
        x: cx - r, y: cy - r, width: r * 2, height: r * 2, rx: 1,
        fill, stroke, "stroke-width": 1.2,
      });
    }
    if (shape === "triangle") {
      const d = r * 1.25;
      return svgEl("polygon", {
        points: `${cx},${cy - d} ${cx - d},${cy + d * 0.85} ${cx + d},${cy + d * 0.85}`,
        fill, stroke, "stroke-width": 1.2,
      });
    }
    if (shape === "diamond") {
      const d = r * 1.25;
      return svgEl("polygon", {
        points: `${cx},${cy - d} ${cx + d},${cy} ${cx},${cy + d} ${cx - d},${cy}`,
        fill, stroke, "stroke-width": 1.2,
      });
    }
    return svgEl("circle", {
      cx, cy, r, fill, stroke, "stroke-width": 1.2,
    });
  }

  // Singleton Tooltip
  let tipNode = null;
  function ensureTip() {
    if (!tipNode) {
      tipNode = el("div", { class: "viz-tooltip", role: "tooltip" });
      document.body.appendChild(tipNode);
    }
    return tipNode;
  }

  const tooltip = {
    show(clientX, clientY, html) {
      const node = ensureTip();
      node.innerHTML = html;
      node.classList.add("is-visible");
      const pad = 12;
      const rect = node.getBoundingClientRect();
      let left = clientX + 14;
      let top = clientY - rect.height - 12;
      if (left + rect.width > window.innerWidth - pad) {
        left = Math.max(pad, clientX - rect.width - 14);
      }
      if (top < pad) {
        top = Math.min(window.innerHeight - rect.height - pad, clientY + 18);
      }
      node.style.left = `${Math.round(left)}px`;
      node.style.top = `${Math.round(top)}px`;
    },
    hide() {
      if (tipNode) tipNode.classList.remove("is-visible");
    },
  };

  function tipRow(label, value, color) {
    const dot = color ? `<span class="viz-tooltip__dot" style="background:${color}"></span>` : "";
    return `<div class="viz-tooltip__row"><span>${dot}${label}</span><strong>${value}</strong></div>`;
  }

  function makeSeg(options, value, onChange) {
    const wrap = el("div", { class: "viz-seg", role: "group" });
    const btns = [];
    for (const opt of options) {
      const b = el("button", {
        type: "button",
        text: opt.label,
        "aria-pressed": opt.value === value ? "true" : "false",
        onClick: () => {
          for (const other of btns) other.setAttribute("aria-pressed", "false");
          b.setAttribute("aria-pressed", "true");
          onChange(opt.value);
        },
      });
      btns.push(b);
      wrap.appendChild(b);
    }
    return wrap;
  }

  function mountResponsive(target, onRedraw) {
    const ro = new ResizeObserver(() => onRedraw());
    ro.observe(target);
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    if (mq.addEventListener) mq.addEventListener("change", onRedraw);
    return () => {
      ro.disconnect();
      if (mq.removeEventListener) mq.removeEventListener("change", onRedraw);
    };
  }

  function initCitations() {
    const pop = el("div", { class: "cite-popover" });
    document.body.appendChild(pop);
    let hideTimer = null;
    const links = document.querySelectorAll(".prose sup.cite a[href^='#ref-']");
    links.forEach((a) => {
      const show = () => {
        clearTimeout(hideTimer);
        const id = a.getAttribute("href").slice(1);
        const li = document.getElementById(id);
        if (!li) return;
        pop.innerHTML = li.innerHTML;
        pop.classList.add("is-visible");
        const ar = a.getBoundingClientRect();
        const pr = pop.getBoundingClientRect();
        const pad = 12;
        let left = Math.max(pad, Math.min(window.innerWidth - pr.width - pad, ar.left - pr.width / 2 + ar.width / 2));
        let top = ar.top - pr.height - 10;
        if (top < pad) top = ar.bottom + 10;
        pop.style.left = `${Math.round(left)}px`;
        pop.style.top = `${Math.round(top)}px`;
      };
      const hide = () => {
        hideTimer = setTimeout(() => pop.classList.remove("is-visible"), 80);
      };
      a.addEventListener("mouseenter", show);
      a.addEventListener("focus", show);
      a.addEventListener("mouseleave", hide);
      a.addEventListener("blur", hide);
    });
  }

  window.DiffMemKit = {
    theme,
    el,
    svgEl,
    clear,
    scaleLinear,
    scaleLog,
    makeSvg,
    drawAxes,
    linePath,
    drawMarker,
    tooltip,
    tipRow,
    makeSeg,
    mountResponsive,
    initCitations,
    fmtSteps: (s) => (s >= 1000 ? `${Math.round(s / 1000)}k` : String(s)),
  };
})();
