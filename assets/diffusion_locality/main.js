/* Bootstrap all interactive widgets and inline citation popovers */
(function () {
  "use strict";

  const BASE = window.POST_ASSET_BASE || "/assets/diffusion_locality/";

  function setStatus(id, msg) {
    const node = document.getElementById(id);
    if (node && !node.dataset.mounted) {
      node.innerHTML = `<div class="viz-loading">${msg}</div>`;
    }
  }

  async function boot() {
    const ids = [
      "viz-hero",
      "viz-ls",
      "viz-capacity",
      "viz-speed",
      "viz-delta-rho",
      "viz-scale-arch",
    ];
    ids.forEach((id) => setStatus(id, "Loading experimental data…"));

    if (window.DiffMemKit && typeof window.DiffMemKit.initCitations === "function") {
      window.DiffMemKit.initCitations();
    }

    let data;
    try {
      const resp = await fetch(`${BASE}data.json`, { cache: "no-cache" });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      data = await resp.json();
    } catch (err) {
      ids.forEach((id) =>
        setStatus(id, `Could not load experimental data (${err.message}).`)
      );
      return;
    }

    const W = window.DiffMemWidgets || {};
    const mounts = [
      ["viz-hero", W.mountHero],
      ["viz-ls", W.mountLS],
      ["viz-capacity", W.mountCapacity],
      ["viz-speed", W.mountSpeed],
      ["viz-delta-rho", W.mountDeltaRho],
      ["viz-scale-arch", W.mountScaleArch],
    ];

    for (const [id, fn] of mounts) {
      const node = document.getElementById(id);
      if (!node || typeof fn !== "function") continue;
      try {
        node.innerHTML = "";
        node.dataset.mounted = "1";
        fn(node, data);
      } catch (e) {
        console.error(`Failed to mount #${id}:`, e);
        node.innerHTML = `<div class="viz-loading">Visualization error in #${id}: ${e.message}</div>`;
      }
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
