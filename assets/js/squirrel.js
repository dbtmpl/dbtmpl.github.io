/*
 * Pixel-art squirrel that wanders back and forth along the top of the name.
 * It always stops at either end of the name; sometimes it loses its balance
 * at the edge, catches itself, and scurries back the other way. It also sits
 * down now and then mid-way.
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
  var RUN_MS = [2500, 6000];  // run this long (random) before a stop mid-way
  var SIT_MS = [1800, 3800];  // sit this long (random)
  var WOBBLE_CHANCE = 0.35;   // chance to lose balance when reaching an edge
  var WOBBLE_MS = 1600;
  var FOOT = 18.5;            // sprite column of the front feet (pivot for the wobble)
  var HEADROOM = 10 * SCALE;  // room above the sprite for tilting and the "!"

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

  function rand(r) { return r[0] + Math.random() * (r[1] - r[0]); }

  // Width of the rendered name text (the h1 box itself spans the whole column).
  function nameWidth() {
    if (!name) return canvas.clientWidth;
    var r = document.createRange();
    r.selectNodeContents(name);
    return r.getBoundingClientRect().width;
  }

  var dpr = 1, maxX = 0;
  function resize() {
    dpr = window.devicePixelRatio || 1;
    var cssW = canvas.clientWidth;
    canvas.width = Math.round(cssW * dpr);
    canvas.height = Math.round((HEADROOM + spriteH + SCALE) * dpr);
    canvas.style.height = HEADROOM + spriteH + SCALE + "px";
    // Right edge: front feet line up with the end of the name.
    var edge = Math.min(cssW, nameWidth());
    maxX = Math.max(0, Math.min(cssW - spriteW, edge - FOOT * SCALE));
    x = Math.max(0, Math.min(x, maxX));
  }

  // dir: 1 = facing right, -1 = facing left (mirrored). tilt: degrees, nose-down positive.
  function draw(key, x, dir, tilt, alarm) {
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.imageSmoothingEnabled = false;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    x = Math.round(x);
    var y = HEADROOM + (key === "run2" ? 0 : SCALE);
    var footX = dir > 0 ? x + FOOT * SCALE : x + (W - FOOT) * SCALE;
    var footY = HEADROOM + SCALE + spriteH;

    if (alarm) {  // pixel "!" above the head
      var ax = dir > 0 ? x + 19 * SCALE : x + 2 * SCALE;
      ctx.fillStyle = darkQ.matches ? "#e8814d" : "#c2551f";
      ctx.fillRect(ax, 0, SCALE, 3 * SCALE);
      ctx.fillRect(ax, 4 * SCALE, SCALE, SCALE);
    }
    if (tilt) {
      ctx.translate(footX, footY);
      ctx.rotate(dir * tilt * Math.PI / 180);
      ctx.translate(-footX, -footY);
    }
    if (dir < 0) {  // mirror around the sprite's centre
      ctx.translate(2 * x + spriteW, 0);
      ctx.scale(-1, 1);
    }
    ctx.drawImage(bake(key), x, y, spriteW, spriteH);
  }

  // Phases: run -> (edge) sit | wobble -> run back; occasional sits mid-way.
  var x = 0, dir = 1, phase = "run", phaseStart = 0, phaseEnd = 0, boostUntil = 0, last = 0;

  function go(p, now, ms) { phase = p; phaseStart = now; phaseEnd = now + ms; }

  function tick(now) {
    if (!last) { last = now; go("run", now, rand(RUN_MS)); }
    var dt = Math.min(50, now - last) / 1000;
    last = now;
    var t = now - phaseStart;

    if (phase === "run") {
      var speed = SPEED * (now < boostUntil ? 1.7 : 1);
      x += dir * speed * dt;
      var atEdge = (dir > 0 && x >= maxX) || (dir < 0 && x <= 0);
      if (atEdge) {
        x = dir > 0 ? maxX : 0;
        if (Math.random() < WOBBLE_CHANCE) go("wobble", now, WOBBLE_MS);
        else go("sit", now, rand(SIT_MS));
        draw("sit1", x, dir);
      } else {
        draw(Math.floor(now / 120) % 2 ? "run1" : "run2", x, dir);
        if (now > phaseEnd) go("sit", now, rand(SIT_MS));
      }
    } else if (phase === "sit") {
      var flick = (t > 700 && t < 950) || (t > 1700 && t < 1950);
      draw(flick ? "sit2" : "sit1", x, dir);
      if (now > phaseEnd) {
        var edgeHere = x <= 0 || x >= maxX;
        if (edgeHere || Math.random() < 0.25) dir = -dir;
        go("run", now, rand(RUN_MS));
      }
    } else if (phase === "wobble") {
      // Tips forward over the edge, windmills back and forth, then catches itself.
      var tilt = 26 * Math.sin(2 * Math.PI * t / 460) * Math.exp(-t / 650);
      var legs = t < 1100 ? (Math.floor(t / 70) % 2 ? "run1" : "run2") : "sit1";
      draw(legs, x, dir, tilt, t > 80 && t < 1000);
      if (now > phaseEnd) {
        dir = -dir;
        boostUntil = now + 1200;  // scurries away from the edge
        go("run", now, rand(RUN_MS));
      }
    }
    requestAnimationFrame(tick);
  }

  function start() {
    resize();
    if (reduce.matches) { x = maxX; draw("sit1", x, 1); return; }
    requestAnimationFrame(tick);
  }

  window.addEventListener("resize", function () {
    resize();
    if (reduce.matches) { x = maxX; draw("sit1", x, 1); }
  });
  darkQ.addEventListener && darkQ.addEventListener("change", function () { if (reduce.matches) draw("sit1", x, 1); });

  // Wait for web fonts so the layout is final.
  (document.fonts && document.fonts.ready ? document.fonts.ready : Promise.resolve()).then(start);

  // Expose for the sprite preview page during development.
  window.__squirrel = { SPRITES: SPRITES, LIGHT: LIGHT };
})();
