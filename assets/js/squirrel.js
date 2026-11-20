/*
 * Pixel-art squirrel that runs across the hero, stops on top of the name
 * for a few seconds, then runs off to the right. Loops with a pause.
 * Honors prefers-reduced-motion (it just sits still).
 */
(function () {
  "use strict";

  // . empty  o fur  d outline/dark fur  t tail highlight  l belly  k eye  n nose
  var SPRITES = {
    run1: [
      "...ddddd..............",
      ".ddtttttdd............",
      "dtttoooootd...........",
      "dttoooooootd..........",
      "dtood..doootd.........",
      ".dd.....dootd....d.d..",
      "........dootd...dodod.",
      "........doottd.dooood.",
      "........dootttddoookod",
      ".........dooooooooooon",
      "........doooooooooood.",
      "........dolllloooood..",
      ".......dod.dllldodd...",
      "......dod.......dod...",
      "......dd.........ddd..",
    ],
    run2: [ // legs tucked; drawn one pixel higher for a bounce
      "...ddddd..............",
      ".ddtttttdd............",
      "dtttoooootd...........",
      "dttoooooootd..........",
      "dtood..doootd.........",
      ".dd.....dootd....d.d..",
      "........dootd...dodod.",
      "........doottd.dooood.",
      "........dootttddoookod",
      ".........dooooooooooon",
      "........doooooooooood.",
      "........dolllloooood..",
      "........dlllddoodd....",
      ".......ddoodood.......",
      "........dd..dd........",
    ],
    sit1: [
      "..ddddd...............",
      ".dtttttdd.............",
      "dttoooootd............",
      "dtood.dootd....d.d....",
      ".dd...dootd...dodod...",
      "......dootd..dooood...",
      "......dootd..doookod..",
      "......dootd..doooooon.",
      "......dootd.dooooood..",
      "......doottddoolllod..",
      ".......dottdoolllldd..",
      ".......dottdollllld...",
      "........ddoollllloo...",
      ".........dooooddood...",
      "..........dddd..ddd...",
    ],
    sit2: [ // tail flick + ear twitch
      "...ddddd..............",
      "..dtttttdd............",
      ".dttoooootd...........",
      ".dtod.dootd....d......",
      "..d...dootd...dod.d...",
      "......dootd..dooood...",
      "......dootd..doookod..",
      "......dootd..doooooon.",
      "......dootd.dooooood..",
      "......doottddoolllod..",
      ".......dottdoolllldd..",
      ".......dottdollllld...",
      "........ddoollllloo...",
      ".........dooooddood...",
      "..........dddd..ddd...",
    ],
  };

  var LIGHT = { o: "#c2551f", d: "#7a3212", t: "#e48a4e", l: "#f4d8b0", k: "#141414", n: "#3a1a0a" };
  var DARK  = { o: "#e07a45", d: "#9a4a20", t: "#f2a774", l: "#f6e0c0", k: "#141414", n: "#3a1a0a" };

  var SCALE = 3;          // CSS pixels per sprite pixel
  var SPEED = 140;        // CSS px per second while running
  var STOP_FOR = 3200;    // ms sitting on the name
  var REST_FOR = 6000;    // ms offscreen before the next lap

  var canvas = document.getElementById("squirrel");
  if (!canvas) return;
  var ctx = canvas.getContext("2d");
  var name = document.querySelector(".hero__name");
  var reduce = window.matchMedia("(prefers-reduced-motion: reduce)");
  var darkQ = window.matchMedia("(prefers-color-scheme: dark)");

  var W = SPRITES.run1[0].length, H = SPRITES.run1.length;
  var spriteW = W * SCALE, spriteH = H * SCALE;
  var cache = {};

  function bake(key) {
    var pal = darkQ.matches ? DARK : LIGHT;
    var id = key + (darkQ.matches ? "-d" : "-l");
    if (cache[id]) return cache[id];
    var c = document.createElement("canvas");
    c.width = W; c.height = H;
    var g = c.getContext("2d");
    SPRITES[key].forEach(function (row, y) {
      for (var x = 0; x < row.length; x++) {
        var col = pal[row[x]];
        if (col) { g.fillStyle = col; g.fillRect(x, y, 1, 1); }
      }
    });
    return (cache[id] = c);
  }

  var dpr = 1, cssW = 0, stopX = 0;
  function resize() {
    dpr = window.devicePixelRatio || 1;
    cssW = canvas.clientWidth;
    canvas.width = Math.round(cssW * dpr);
    canvas.height = Math.round((spriteH + SCALE) * dpr);
    canvas.style.height = spriteH + SCALE + "px";
    // Stop roughly above the middle of the name ("Biertimpel"'s first letters).
    var nameW = name ? name.getBoundingClientRect().width : cssW / 2;
    stopX = Math.max(0, Math.min(cssW - spriteW, nameW * 0.55 - spriteW / 2));
  }

  function draw(key, x) {
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.imageSmoothingEnabled = false;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.drawImage(bake(key), Math.round(x), key === "run2" ? 0 : SCALE, spriteW, spriteH);
  }

  // Phases: run in -> sit -> run out -> rest
  var phase = "in", phaseStart = 0, x = -spriteW, last = 0;

  function tick(now) {
    if (!last) { last = now; phaseStart = now; }
    var dt = Math.min(50, now - last) / 1000;
    last = now;
    var t = now - phaseStart;
    var runFrame = Math.floor(now / 110) % 2 ? "run1" : "run2";

    if (phase === "in") {
      x += SPEED * dt;
      if (x >= stopX) { x = stopX; phase = "sit"; phaseStart = now; }
      draw(runFrame, x);
    } else if (phase === "sit") {
      // Flick the tail twice while sitting.
      var flick = (t > 900 && t < 1150) || (t > 2100 && t < 2350);
      draw(flick ? "sit2" : "sit1", x);
      if (t > STOP_FOR) { phase = "out"; phaseStart = now; }
    } else if (phase === "out") {
      x += SPEED * 1.3 * dt;
      draw(runFrame, x);
      if (x > cssW + 4) { phase = "rest"; phaseStart = now; ctx.clearRect(0, 0, canvas.width, canvas.height); }
    } else if (phase === "rest") {
      if (t > REST_FOR) { phase = "in"; phaseStart = now; x = -spriteW; }
    }
    requestAnimationFrame(tick);
  }

  function start() {
    resize();
    if (reduce.matches) { draw("sit1", stopX); return; }
    requestAnimationFrame(tick);
  }

  window.addEventListener("resize", function () {
    resize();
    if (reduce.matches) draw("sit1", stopX);
  });
  darkQ.addEventListener && darkQ.addEventListener("change", function () { if (reduce.matches) draw("sit1", stopX); });

  // Wait for web fonts so the name width (and stop position) is final.
  (document.fonts && document.fonts.ready ? document.fonts.ready : Promise.resolve()).then(start);

  // Expose for the sprite preview page during development.
  window.__squirrel = { SPRITES: SPRITES, LIGHT: LIGHT };
})();
