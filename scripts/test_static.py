"""Drive the static (in-browser) LeafLens build to verify predictions are correct
and capture screenshots. Requires the static server on http://127.0.0.1:8090.
"""

import os
from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "docs", "screenshots")
DATA = os.path.join(HERE, "..", "ml", "data")
URL = "http://127.0.0.1:8090/"
os.makedirs(OUT, exist_ok=True)

CASES = [
    ("Rice___Blast", "Rice", "Blast"),
    ("Tomato___Late_blight", "Tomato", "Late blight"),
    ("Potato___Early_blight", "Potato", "Early blight"),
    ("Corn___Common_rust", "Corn", "Common rust"),
]


def run():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport={"width": 412, "height": 900}, device_scale_factor=2)
        page = ctx.new_page()
        page.goto(URL, wait_until="networkidle")
        # wait until the model is loaded (diagnose button becomes enabled once a file
        # is chosen; first upload a file so the button reflects modelReady)
        first_img = sorted(__import__("glob").glob(os.path.join(DATA, CASES[0][0], "*.jpg")))[0]
        page.set_input_files('input[type="file"]', os.path.abspath(first_img))
        page.wait_for_selector("button.btn:not([disabled])", timeout=90000)

        results = []
        # first case screenshots
        page.screenshot(path=os.path.join(OUT, "01-upload.png"))
        page.click("button.btn")
        page.wait_for_selector(".diagnosis", timeout=60000)
        page.wait_for_timeout(800)
        disease = page.locator(".disease-name").inner_text()
        conf = page.locator(".conflabel").inner_text()
        results.append((CASES[0][0], disease, conf))
        page.screenshot(path=os.path.join(OUT, "02-result-bn.png"), full_page=True)
        page.get_by_role("button", name="EN", exact=True).click()
        page.wait_for_timeout(600)
        page.screenshot(path=os.path.join(OUT, "03-result-en.png"), full_page=True)

        # more cases to verify accuracy (no screenshots)
        for label, crop, dis in CASES[1:]:
            page.get_by_role("button", name="বাংলা", exact=True).click()
            page.wait_for_timeout(200)
            # reset
            page.get_by_role("button").filter(has_text="").first  # noop
            page.click("button.btn.secondary")  # "Diagnose another"
            img = sorted(__import__("glob").glob(os.path.join(DATA, label, "*.jpg")))[0]
            page.set_input_files('input[type="file"]', os.path.abspath(img))
            page.wait_for_selector("button.btn:not([disabled])", timeout=30000)
            page.click("button.btn")
            page.wait_for_selector(".diagnosis", timeout=60000)
            page.wait_for_timeout(500)
            d = page.locator(".disease-name").inner_text()
            c = page.locator(".conflabel").inner_text()
            results.append((label, d, c))

        # history screenshot (bangla)
        page.get_by_role("button", name="বাংলা", exact=True).click()
        page.wait_for_timeout(300)
        page.locator(".card").last.scroll_into_view_if_needed()
        page.wait_for_timeout(300)
        page.screenshot(path=os.path.join(OUT, "04-history.png"))

        browser.close()

        print("=== predictions ===")
        ok = 0
        for label, disease, conf in results:
            expect = label.split("___")[1].replace("_", " ")
            good = disease.strip().lower() == expect.strip().lower()
            ok += good
            print(f"{label:28s} -> UI shows '{disease}' ({conf}) {'OK' if good else 'MISMATCH expected '+expect}")
        print(f"\n{ok}/{len(results)} correct")


if __name__ == "__main__":
    run()
