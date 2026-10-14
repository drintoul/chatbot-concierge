/* chatbot-concierge embeddable widget.
 *
 *   <script src="https://HOST/static/widget.js" data-url="https://tenant-site.com/" async></script>
 *
 * data-url      tenant corpus URL (required — what the concierge answers about)
 * data-api      concierge base URL (default: origin this script was loaded from)
 * data-name     title-bar label (default "concierge")
 * data-greeting opening bot message (default "Hi! How can we help?")
 * data-color    accent color, hex e.g. "#e11d48" (default "#34d399")
 *
 * Everything lives in a shadow root so host-page CSS can't bleed in.
 */
(function () {
  var script = document.currentScript;
  var api = (script.getAttribute("data-api") || new URL(script.src).origin).replace(/\/+$/, "");

  var opts = {
    url: script.getAttribute("data-url") || "",
    name: script.getAttribute("data-name") || "concierge",
    greeting: script.getAttribute("data-greeting") || "Hi! How can we help?",
    color: /^#[0-9a-fA-F]{6}$/.test(script.getAttribute("data-color") || "")
      ? script.getAttribute("data-color")
      : "#34d399",
  };

  // Foreground color readable on top of the accent.
  function fgFor(hex) {
    var c = parseInt(hex.slice(1), 16);
    var l = 0.299 * (c >> 16 & 255) + 0.587 * (c >> 8 & 255) + 0.114 * (c & 255);
    return l > 150 ? "#06281e" : "#ffffff";
  }

  var host = document.createElement("div");
  var root = host.attachShadow({ mode: "open" });
  root.innerHTML =
    "<style>" +
    ":host { all: initial; font-family: ui-sans-serif, system-ui, sans-serif; }" +
    "#launch { position: fixed; right: 26px; bottom: 26px; width: 62px; height: 62px;" +
    "  border: 0; border-radius: 50%; cursor: pointer; z-index: 2147483000;" +
    "  background: linear-gradient(135deg, var(--cc), color-mix(in srgb, var(--cc), #000 22%));" +
    "  box-shadow: 0 8px 30px rgba(0,0,0,.5); display: flex; align-items: center; justify-content: center;" +
    "  transition: transform .15s; }" +
    "#launch:hover { transform: scale(1.08); }" +
    "#launch svg { fill: var(--cc-fg); }" +
    "#pop { position: fixed; right: 26px; bottom: 100px; z-index: 2147483000;" +
    "  width: 390px; max-width: calc(100vw - 40px); height: 540px; max-height: 72vh;" +
    "  display: none; flex-direction: column; overflow: hidden;" +
    "  background: #0a0f0d; border: 2px solid #2a3833; border-radius: 20px;" +
    "  box-shadow: 0 18px 60px rgba(0,0,0,.6); font-size: 15px; color: #e9f0ec; }" +
    "#pop.open { display: flex; }" +
    "#head { display: flex; align-items: center; gap: 10px; padding: 14px 18px;" +
    "  background: #1a2420; border-bottom: 2px solid #2a3833; font-weight: 800; }" +
    "#head .host { color: #8ba398; font-weight: 600; font-size: 13px; }" +
    "#head button { background: none; border: 0; color: #8ba398; font-size: 24px;" +
    "  line-height: 1; cursor: pointer; padding: 0 2px; }" +
    "#head button:hover { color: #e9f0ec; }" +
    "#head button:first-of-type { margin-left: auto; }" +
    "#head .clear { font-size: 19px; }" +
    "#chat { display: flex; flex-direction: column; gap: 14px; flex: 1; min-height: 0;" +
    "  overflow-y: auto; padding: 16px; }" +
    "#bar { display: flex; gap: 10px; padding: 12px; background: #131a17; border-top: 2px solid #2a3833; }" +
    "#bar input { flex: 1; background: #1a2420; border: 2px solid #2a3833; border-radius: 12px;" +
    "  color: #e9f0ec; padding: 10px 14px; font: inherit; outline: none; }" +
    "#bar input:focus { border-color: var(--cc); }" +
    "#bar button { padding: 14px 22px; border: 0; border-radius: 14px; font: inherit;" +
    "  font-weight: 800; cursor: pointer; color: var(--cc-fg);" +
    "  background: linear-gradient(135deg, var(--cc), color-mix(in srgb, var(--cc), #000 22%)); }" +
    ".msg { max-width: 82%; padding: 14px 20px; border-radius: 20px; white-space: pre-wrap; }" +
    ".msg.user { align-self: flex-end; color: var(--cc-fg); font-weight: 600;" +
    "  background: linear-gradient(135deg, var(--cc), color-mix(in srgb, var(--cc), #000 22%));" +
    "  border-bottom-right-radius: 6px; }" +
    ".msg.bot { align-self: flex-start; background: #1a2420; border: 2px solid #2a3833;" +
    "  border-bottom-left-radius: 6px; }" +
    ".srcs { margin-top: 10px; padding-top: 10px; border-top: 2px solid #2a3833; font-size: 13px; }" +
    ".srcs a { color: var(--cc); word-break: break-all; }" +
    ".typing { display: inline-flex; gap: 6px; padding: 6px 2px; }" +
    ".typing i { width: 8px; height: 8px; border-radius: 50%; background: #8ba398;" +
    "  animation: ccw-blink 1.2s infinite; }" +
    ".typing i:nth-child(2) { animation-delay: .2s; }" +
    ".typing i:nth-child(3) { animation-delay: .4s; }" +
    "@keyframes ccw-blink { 0%,60%,100% { opacity: .25; } 30% { opacity: 1; } }" +
    "</style>" +
    '<button id="launch" title="Chat with us"><svg viewBox="0 0 24 24" width="27" height="27"><path d="M20 2H4a2 2 0 0 0-2 2v18l4-4h14a2 2 0 0 0 2-2V4a2 2 0 0 0-2-2zm-4 9H8V9h8zm-6 4v-2h8v2z"/></svg></button>' +
    '<div id="pop">' +
    '  <div id="head"><img src="' + api + '/static/favicon.svg" width="22" height="22" alt="">' +
    '    <span class="name"></span><span class="host"></span>' +
    '    <button class="clear" title="New conversation">&#8635;</button>' +
    '    <button class="close" title="Close">&times;</button></div>' +
    '  <div id="chat"><div class="msg bot greet"></div></div>' +
    '  <form id="bar"><input id="text" placeholder="Ask us anything…" autocomplete="off">' +
    '    <button type="submit">Send</button></form>' +
    "</div>";

  document.body.appendChild(host);

  var pop = root.getElementById("pop"),
    chat = root.getElementById("chat"),
    text = root.getElementById("text");

  function applyTheme() {
    host.style.setProperty("--cc", opts.color);
    host.style.setProperty("--cc-fg", fgFor(opts.color));
    root.querySelector(".name").textContent = opts.name;
    // Re-seed the greeting only while it is still the sole message.
    var g = chat.querySelector(".greet");
    if (chat.childElementCount === 1 && g) g.textContent = opts.greeting;
  }

  function hostLabel() {
    try { root.querySelector(".host").textContent = "· " + new URL(opts.url).hostname; }
    catch (e) { /* ignore */ }
  }
  applyTheme();
  hostLabel();

  function bubble(cls, html) {
    var m = document.createElement("div");
    m.className = "msg " + cls;
    m.innerHTML = html;
    chat.appendChild(m);
    chat.scrollTop = chat.scrollHeight;
    return m;
  }

  root.getElementById("launch").onclick = function () {
    pop.classList.toggle("open");
    if (pop.classList.contains("open")) text.focus();
  };
  root.querySelector(".close").onclick = function () { pop.classList.remove("open"); };
  root.querySelector(".clear").onclick = function () {
    chat.innerHTML = "";
    var m = document.createElement("div");
    m.className = "msg bot greet";
    m.textContent = opts.greeting;
    chat.appendChild(m);
  };

  root.getElementById("bar").addEventListener("submit", function (e) {
    e.preventDefault();
    var q = text.value.trim();
    if (!q) return;
    text.value = "";
    bubble("user", q.replace(/</g, "&lt;"));
    var typing = bubble("bot", '<span class="typing"><i></i><i></i><i></i></span>');
    fetch(api + "/query", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ url: opts.url, question: q }),
    })
      .then(function (r) { return r.json(); })
      .then(function (start) {
        var runId = start.run_id;
        (function poll() {
          fetch(api + "/runs/" + encodeURIComponent(runId))
            .then(function (r) { return r.json(); })
            .then(function (st) {
              if (st.status === "running") { setTimeout(poll, 900); return; }
              typing.innerHTML = (st.answer || st.error || "…")
                  .replace(/</g, "&lt;").replace(/\n/g, "<br>") +
                (st.sources && st.sources.length
                  ? '<div class="srcs">source: <a href="' + st.sources[0] + '" target="_blank">' + st.sources[0] + "</a></div>"
                  : "");
            })
            .catch(function () { setTimeout(poll, 1500); });
        })();
      })
      .catch(function (err) { typing.textContent = "Something went wrong: " + err; });
  });

  window.ConciergeWidget = {
    setUrl: function (u) { opts.url = u; hostLabel(); },
    setVisible: function (v) { host.style.display = v ? "" : "none"; },
    setTheme: function (o) {
      if (o.name != null) opts.name = o.name || "concierge";
      if (o.greeting != null) opts.greeting = o.greeting || "Hi! How can we help?";
      if (o.color != null && /^#[0-9a-fA-F]{6}$/.test(o.color)) opts.color = o.color;
      applyTheme();
    },
  };
})();
