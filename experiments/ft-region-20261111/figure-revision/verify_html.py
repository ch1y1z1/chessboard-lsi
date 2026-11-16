"""Verify the standalone overview in an explicitly selected local Chromium."""
import argparse
import json
from pathlib import Path
from playwright.sync_api import sync_playwright


ap = argparse.ArgumentParser()
ap.add_argument("--output", type=Path, required=True)
ap.add_argument("--chromium-path", required=True)
a = ap.parse_args()
out = a.output.resolve()
results = {}
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, executable_path=a.chromium_path)
    for name, w, h in [("desktop", 1280, 950), ("mobile", 390, 844)]:
        page = browser.new_page(viewport={"width": w, "height": h})
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto((out/"ft-region-overview-revised-20261116.html").as_uri(),
                  wait_until="networkidle")
        page.evaluate("document.fonts.ready")
        page.screenshot(path=str(out/f"html-{name}.png"), full_page=True)
        result = page.evaluate("""() => {
          const p=[...document.querySelectorAll('.am-panel')].map(x=>{
            let r=x.getBoundingClientRect();return {top:r.top,bottom:r.bottom}});
          return {width:innerWidth,scrollWidth:document.documentElement.scrollWidth,
            images:[...document.images].map(x=>({loaded:x.complete&&x.naturalWidth>0})),
            panels:p};
        }""")
        result["page_errors"] = errors
        results[name] = result
        assert result["width"] == result["scrollWidth"] and not errors
        assert len(result["panels"]) == 5
        assert len(result["images"]) == 4 and all(i["loaded"] for i in result["images"])
        assert all(result["panels"][i]["bottom"] <= result["panels"][i+1]["top"]
                   for i in range(4))
        page.close()
    browser.close()
(out/"html-verification.json").write_text(json.dumps(results, indent=2))
print("Five panels and four embedded images; desktop/mobile layout checks passed.")
