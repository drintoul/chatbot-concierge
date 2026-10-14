/* chatbot-concierge embeddable widget.
 *
 *   <script src="https://HOST/static/widget.js" data-url="https://tenant-site.com/" async></script>
 *
 * data-url   tenant corpus URL (required — what the concierge answers about)
 * data-api   concierge base URL (default: origin this script was loaded from)
 *
 * Everything lives in a shadow root so host-page CSS can't bleed in.
 */
(function () {
  var script = document.currentScript;
  var api = (script.getAttribute("data-api") || new URL(script.src).origin).replace(/\/+$/, "");
  var url = script.getAttribute("data-url") || "";

  var host = document.createElement("div");
  var root = host.attachShadow({ mode: "open" });
  root.innerHTML =
    "<style>" +
    ":host { all: initial; font-family: ui-sans-serif, system-ui, sans-serif; }" +
    "#launch { position: fixed; right: 26px; bottom: 26px; width: 62px; height: 62px;" +
    "  border: 0; border-radius: 50%; cursor: pointer; z-index: 2147483000;" +
    "  background: linear-gradient(135deg, #34d399, #2dd4bf);" +
    "  box-shadow: 0 8px 30px rgba(0,0,0,.5); display: flex; align-items: center; justify-content: center;" +
    "  transition: transform .15s; }" +
    "#launch:hover { transform: scale(1.08); }" +
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
    "#bar input:focus { border-color: #34d399; }" +
    "#bar button { padding: 14px 22px; border: 0; border-radius: 14px; font: inherit;" +
    "  font-weight: 800; cursor: pointer; color: #06281e;" +
    "  background: linear-gradient(135deg, #34d399, #2dd4bf); }" +
    ".msg { max-width: 82%; padding: 14px 20px; border-radius: 20px; white-space: pre-wrap; }" +
    ".msg.user { align-self: flex-end; background: linear-gradient(135deg, #34d399, #2dd4bf);" +
    "  color: #06281e; font-weight: 600; border-bottom-right-radius: 6px; }" +
    ".msg.bot { align-self: flex-start; background: #1a2420; border: 2px solid #2a3833;" +
    "  border-bottom-left-radius: 6px; }" +
    ".srcs { margin-top: 10px; padding-top: 10px; border-top: 2px solid #2a3833; font-size: 13px; }" +
    ".srcs a { color: #34d399; word-break: break-all; }" +
    ".typing { display: inline-flex; gap: 6px; padding: 6px 2px; }" +
    ".typing i { width: 8px; height: 8px; border-radius: 50%; background: #8ba398;" +
    "  animation: ccw-blink 1.2s infinite; }" +
    ".typing i:nth-child(2) { animation-delay: .2s; }" +
    ".typing i:nth-child(3) { animation-delay: .4s; }" +
    "@keyframes ccw-blink { 0%,60%,100% { opacity: .25; } 30% { opacity: 1; } }" +
    "</style>" +
    '<button id="launch" title="Chat with us"><svg viewBox="0 0 24 24" width="27" height="27" fill="#06281e"><path d="M20 2H4a2 2 0 0 0-2 2v18l4-4h14a2 2 0 0 0 2-2V4a2 2 0 0 0-2-2zm-4 9H8V9h8zm-6 4v-2h8v2z"/></svg></button>' +
    '<div id="pop">' +
    '  <div id="head"><img src="' + api + '/static/favicon.svg" width="22" height="22" alt="">' +
    '    <span>concierge</span><span class="host"></span>' +
    '    <button class="clear" title="New conversation">&#8635;</button>' +
    '    <button class="close" title="Close">&times;</button></div>' +
    '  <div id="chat"><div class="msg bot">Hi! How can we help?</div></div>' +
    '  <form id="bar"><input id="text" placeholder="Ask us anything…" autocomplete="off">' +
    '    <button type="submit">Send</button></form>' +
    "</div>";

  document.body.appendChild(host);

  var pop = root.getElementById("pop"),
    chat = root.getElementById("chat"),
    text = root.getElementById("text");

  function hostLabel() {
    try { root.querySelector(".host").textContent = "· " + new URL(url).hostname; }
    catch (e) { /* ignore */ }
  }
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
    bubble("bot", "Hi! How can we help?");
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
      body: JSON.stringify({ url: url, question: q }),
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
    setUrl: function (u) { url = u; hostLabel(); },
    setVisible: function (v) { host.style.display = v ? "" : "none"; },
  };
})();
