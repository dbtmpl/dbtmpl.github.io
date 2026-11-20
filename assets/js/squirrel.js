/*
 * Pixel-art squirrel that wanders back and forth above the name, stopping
 * now and then to sit and flick its tail. It never leaves its strip.
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

  // Darker, earthier browns. Dark mode is a touch lighter so it reads on the dark background.
  var LIGHT = { o: "#8f3d17", d: "#4a1e0a", t: "#b0592b", l: "#dcbb92", k: "#111111", n: "#2a1206" };
  var DARK  = { o: "#b9602f", d: "#6e3414", t: "#d27f4c", l: "#e6cba5", k: "#111111", n: "#2a1206" };

  var SCALE = 2;              // CSS pixels per sprite pixel
  var SPEED = 90;             // CSS px per second while running
  var RUN_MS = [1500, 4000];  // run this long (random) before the next stop
  var SIT_MS = [1800, 3800];  // sit this long (random)

  var canvas = document.getElementById("squirrel");
  if (!canvas) return;
  var ctx = canvas.getContext("2d");
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

  function rand(r) { return r[0] + Math.random() * (r[1] - r[0]); }

  var dpr = 1, maxX = 0;
  function resize() {
    dpr = window.devicePixelRatio || 1;
    var cssW = canvas.clientWidth;
    canvas.width = Math.round(cssW * dpr);
    canvas.height = Math.round((spriteH + SCALE) * dpr);
    canvas.style.height = spriteH + SCALE + "px";
    maxX = Math.max(0, cssW - spriteW);
    x = Math.min(x, maxX);
  }

  // dir: 1 = facing right, -1 = facing left (sprite is mirrored)
  function draw(key, x, dir) {
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.imageSmoothingEnabled = false;
    var y = key === "run2" ? 0 : SCALE;
    if (dir < 0) {
      ctx.setTransform(-dpr, 0, 0, dpr, (Math.round(x) + spriteW) * dpr, 0);
      ctx.drawImage(bake(key), 0, y, spriteW, spriteH);
    } else {
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.drawImage(bake(key), Math.round(x), y, spriteW, spriteH);
    }
  }

  // Wanders back and forth inside the canvas: run, sit, run; turns at the edges.
  var x = 0, dir = 1, phase = "run", phaseEnd = 0, sitStart = 0, last = 0;

  function tick(now) {
    if (!last) { last = now; phaseEnd = now + rand(RUN_MS); }
    var dt = Math.min(50, now - last) / 1000;
    last = now;

    if (phase === "run") {
      x += dir * SPEED * dt;
      if (x >= maxX) { x = maxX; dir = -1; }
      if (x <= 0) { x = 0; dir = 1; }
      draw(Math.floor(now / 120) % 2 ? "run1" : "run2", x, dir);
      if (now > phaseEnd) { phase = "sit"; sitStart = now; phaseEnd = now + rand(SIT_MS); }
    } else {
      var t = now - sitStart;
      var flick = (t > 700 && t < 950) || (t > 1700 && t < 1950);
      draw(flick ? "sit2" : "sit1", x, dir);
      if (now > phaseEnd) {
        phase = "run";
        phaseEnd = now + rand(RUN_MS);
        if (Math.random() < 0.3) dir = -dir;  // sometimes turn around after a rest
      }
    }
    requestAnimationFrame(tick);
  }

  function start() {
    resize();
    if (reduce.matches) { x = Math.min(maxX, 150); draw("sit1", x, 1); return; }
    requestAnimationFrame(tick);
  }

  window.addEventListener("resize", function () {
    resize();
    if (reduce.matches) draw("sit1", x, 1);
  });
  darkQ.addEventListener && darkQ.addEventListener("change", function () { if (reduce.matches) draw("sit1", x, 1); });

  // Wait for web fonts so the layout is final.
  (document.fonts && document.fonts.ready ? document.fonts.ready : Promise.resolve()).then(start);

  // Expose for the sprite preview page during development.
  window.__squirrel = { SPRITES: SPRITES, LIGHT: LIGHT };
})();
