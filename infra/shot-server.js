// Minimal screenshot endpoint — runs inside the firecrawl playwright-service
// image (which already ships Chromium + the playwright lib), mounted over the
// stock server so we get screenshots without pulling another browser image.
//
// POST /screenshot {"url": "https://..."} -> image/png
// GET  /health                           -> 200

const http = require("http");
const { chromium } = require("playwright");

const PORT = parseInt(process.env.PORT || "3000", 10);
let browser = null;
let busy = Promise.resolve();

async function shot(url) {
  if (!browser) {
    browser = await chromium.launch({
      headless: true,
      args: ["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage"],
    });
  }
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 800 } });
  try {
    const page = await ctx.newPage();
    await page.goto(url, { waitUntil: "load", timeout: 20000 });
    await page.waitForTimeout(1200);
    return await page.screenshot({ type: "png" });
  } finally {
    await ctx.close();
  }
}

http
  .createServer((req, res) => {
    if (req.method === "GET" && req.url === "/health") {
      res.writeHead(200).end("ok");
      return;
    }
    if (req.method !== "POST" || req.url !== "/screenshot") {
      res.writeHead(404).end();
      return;
    }
    let body = "";
    req.on("data", (c) => (body += c));
    req.on("end", () => {
      // serialize captures — one browser, so don't overlap pages
      busy = busy.then(async () => {
        try {
          const { url } = JSON.parse(body || "{}");
          if (!/^https?:\/\//.test(url || "")) throw new Error("invalid url");
          const png = await shot(url);
          res.writeHead(200, { "content-type": "image/png" });
          res.end(png);
        } catch (e) {
          res.writeHead(500, { "content-type": "application/json" });
          res.end(JSON.stringify({ error: String(e && e.message || e) }));
        }
      });
    });
  })
  .listen(PORT, () => console.log(`shot-server on :${PORT}`));
