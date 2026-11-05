#!/usr/bin/env python3
"""Ask each indexed collection a battery of realistic visitor questions.

Questions are written the way real visitors type: lowercase, missing
apostrophes, swapped letters, filler words. Mix of answerable and
out-of-corpus questions to exercise both grounding and handoff.

Usage:
    python3 scripts/ask_questions.py                 # all sites
    python3 scripts/ask_questions.py --only highwayman
    python3 scripts/ask_questions.py --workers 4 --out /tmp/qa
"""

import argparse
import concurrent.futures
import json
import sys
import time
import urllib.request
from datetime import datetime, timezone

API = "http://localhost:8000"
POLL_EVERY = 4
POLL_TIMEOUT = 300

# site-key -> questions. The key is matched against the site's URL
# substring, so tokens are never hardcoded here.
QUESTIONS = {
    "farmhousebrewing": [
        "wut time do u guys open",
        "do yous have food or is it just beer",
        "can i bring my dog",
        "were are you located at",
        "do yall do flights or like samplers",
        "is there parking there",
        "can kids come in or is it 19 plus only",
        "do you sell cans or growlers to go",
        "whats on tap right now",
        "do you have any gluten free beer",
        "can i book a table for saturday nite",
        "do you host private events or partys",
        "wat time does the kitchen close",
        "got any non alcoholic drinks",
        "is there a happy hour",
        "how much is a pint",
        "do yous do brewery tours",
        "can i buy a gift card for my dad",
        "are you open on holidays like boxing day",
        "do you have wifi for customers",
        "any vegan options on the menu",
    ],
    "oldyalebrewing": [
        "when r u open",
        "where is the brewery exactly",
        "do you have a taproom or do you just sell beer",
        "can i get a tour of the brewery",
        "whats ur most popular beer",
        "do you serve food at the taproom",
        "is there a patio",
        "can dogs come in",
        "do you sell merch like hats and stuff",
        "can i order beer online",
        "do you deliver",
        "whats the alcohol percent of your ipa",
        "do yous have seasonal or limited beers",
        "is there parking at the brewery",
        "are kids allowed in",
        "do you do trivia nights or events",
        "can i book the space for a private event",
        "do you fill growlers",
        "how do i apply for a job there",
        "do you have gift cards",
        "where can i buy your beer in stores",
    ],
    "westerncanoekayak": [
        "what time are you open",
        "are you open sundays",
        "where are you guys located",
        "do you rent kayaks",
        "how much is it to rent a canoe",
        "do you sell used canoes",
        "whats your return policy",
        "do you offer kayak lessons",
        "can i demo a kayak before i buy it",
        "do yous have paddleboards too",
        "do you sell life jackets",
        "whats a good kayak for a beginer",
        "do you do repairs on canoes",
        "is there parking at the shop",
        "can i order online and pick up in store",
        "do you price match other stores",
        "what brands do you carry",
        "do you have dry bags and waterproof stuff",
        "when do kayaks usually go on sale",
        "whats ur phone number",
        "can i trade in my old canoe",
        "do you sell roof racks for my car",
    ],
    "fraserriverlodge": [
        "how much is a room per night",
        "what time is check in and check out",
        "do you allow pets",
        "is breakfast included",
        "do the rooms have wifi",
        "can i book a fishing trip",
        "whats included in the fishing packages",
        "do you have cabins or just rooms",
        "can i have my wedding there",
        "how many people can the venue hold",
        "is there a hot tub",
        "do you have a resturant on site",
        "whats your cancelation policy",
        "are you open in the winter",
        "do you have somewhere to moor my boat",
        "how far are you from vancouver",
        "do you offer sturgeon fishing trips",
        "is there a spa",
        "can kids stay there",
        "do you have ev charging",
        "what should i bring for a fishing trip",
    ],
    "fieldhousebrewing": [
        "what time do you open",
        "do you have two locations or just one",
        "were is the chilliwack location",
        "do you serve food",
        "hows the pizza",
        "can i bring my dog",
        "do you have a patio",
        "are kids allowed",
        "do yous do trivia nights",
        "can i buy beer to go",
        "got any non beer options like cider",
        "whats your sour beer selection like",
        "can i rent the space for a party",
        "do you sell merch",
        "is there parking at the abbotsford spot",
        "do you take reservations",
        "do you have live music",
        "what beers do you have in cans",
        "do you deliver",
        "can i get a keg for a party",
        "do you have gift cards",
    ],
    "pizzagarden": [
        "what are your locations",
        "do yous deliver to surrey",
        "whats the gluten free pizza like",
        "do you have vegan cheese",
        "is there a happy hour",
        "how much is a medium pizza",
        "can i order online for pickup",
        "what time does the abbotsford one close",
        "do yous do catering for office partys",
        "got any deals or coupons right now",
        "is there a kids menu",
        "whats the pg points thing",
        "do you take cash or card only",
        "can i get a pizza kit to make at home",
        "wich location is closest to langley",
        "are you open on christmas",
        "do you have halal options",
        "can i book a table for 10 people",
        "whats in the pizza garden special",
        "do yous still do the lunch special",
    ],
    "kingkabob": [
        "what time do you close",
        "were are you located",
        "is everything halal",
        "do you do catering for weddings",
        "can i make a reservation for friday",
        "do you have a lunch special",
        "whats on the menu besides kabobs",
        "do you deliver or is it pickup only",
        "got any vegetarian dishes",
        "how much is the mixed grill",
        "do you sell gift cards",
        "is there parking",
        "can i order online",
        "do yous do family platters",
        "any gluten free options",
        "whats the difference between ur kabobs",
        "do you have baklava or desserts",
        "can i rent the place for a private party",
        "are you open mondays",
        "do you have a kids menu",
    ],
    "highwaymanpubonline": [
        "what time do you close tonight",
        "whats the daily special today",
        "do you have wing specials",
        "can i watch the game there",
        "do you have pool tables",
        "is there a kids menu",
        "can minors come in with there parents",
        "do you take reservations",
        "whats on the drink menu",
        "do you have a happy hour",
        "whens trivia night",
        "do you do wing wednesday",
        "can i order take out",
        "any gluten free options",
        "is there parking",
        "do you have a patio and can dogs come",
        "do you host birthday partys",
        "any live music this weekend",
        "do you deliver thru skip the dishes",
        "whats the breakfast like",
        "do you have vegetarian options",
    ],
}


def post(path: str, body: dict) -> dict:
    req = urllib.request.Request(
        API + path,
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
    )
    return json.load(urllib.request.urlopen(req))


def get(path: str) -> dict:
    return json.load(urllib.request.urlopen(API + path))


def ask(url: str, token: str, question: str) -> dict:
    started = time.time()
    try:
        rid = post("/query", {"url": url, "question": question, "token": token})["run_id"]
    except Exception as e:
        return {"question": question, "answer": f"REQUEST FAILED: {e}", "source": None, "secs": 0}
    while True:
        run = get(f"/runs/{rid}")
        if run["status"] != "running":
            ranked = run.get("ranked") or []
            return {
                "question": question,
                "answer": run.get("answer") or "(no answer)",
                "grounded": run.get("grounded"),
                "source": ranked[0]["source"] if ranked else None,
                "status": run["status"],
                "secs": round(time.time() - started, 1),
            }
        if time.time() - started > POLL_TIMEOUT:
            return {"question": question, "answer": "TIMEOUT", "source": None, "secs": POLL_TIMEOUT}
        time.sleep(POLL_EVERY)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="only run sites whose URL contains this")
    ap.add_argument("--limit", type=int, help="cap questions per site (smoke test)")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default=f"qa-report-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}")
    args = ap.parse_args()

    sites = get("/sites")["sites"]
    jobs = []
    order = {}  # question -> original position, per site
    for key, qs in QUESTIONS.items():
        site = next((s for s in sites if key in s["url"]), None)
        if not site:
            print(f"!! no indexed site matching '{key}' — skipping", file=sys.stderr)
            continue
        if args.only and args.only not in site["url"]:
            continue
        for i, q in enumerate(qs[: args.limit] if args.limit else qs):
            order[(site["url"], q)] = i
            jobs.append((site["url"], site["token"], q))

    print(f"{len(jobs)} questions, workers={args.workers}")
    results = []
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(ask, u, t, q): (u, q) for u, t, q in jobs}
        for fut in concurrent.futures.as_completed(futs):
            url, q = futs[fut]
            r = fut.result()
            r["site"] = url
            results.append(r)
            done += 1
            print(f"[{done}/{len(jobs)}] {url.split('//')[1]:34} {q[:50]!r} -> {r['answer'][:70]!r}")

    results.sort(key=lambda r: (r["site"], order[(r["site"], r["question"])]))

    md, js = args.out + ".md", args.out + ".json"
    with open(js, "w") as f:
        json.dump(results, f, indent=2)
    with open(md, "w") as f:
        f.write(f"# Q&A report — {datetime.now():%Y-%m-%d %H:%M}\n\n")
        cur = None
        for r in results:
            if r["site"] != cur:
                cur = r["site"]
                f.write(f"\n## {cur}\n\n")
            src = f" · [src]({r['source']})" if r.get("source") else ""
            f.write(f"**Q: {r['question']}**\n\n{r['answer']}{src} · {r.get('secs')}s\n\n")
    print(f"\nwrote {md} and {js}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
