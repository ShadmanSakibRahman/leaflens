"""Verify the LIVE GitHub Pages site works end-to-end and capture screenshots."""

import os
import glob
from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "docs", "screenshots")
DATA = os.path.join(HERE, "..", "ml", "data")
URL = os.environ.get("LIVE_URL", "https://shadmansakibrahman.github.io/leaflens/")
os.makedirs(OUT, exist_ok=True)


def img_for(label):
    return os.path.abspath(sorted(glob.glob(os.path.join(DATA, label, "*.jpg")))[0])


def run():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport={"width": 412, "height": 900}, device_scale_factor=2)
        page = ctx.new_page()
        print("loading", URL)
        page.goto(URL, wait_until="networkidle", timeout=90000)

        page.set_input_files('input[type="file"]', img_for("Rice___Blast"))
        page.wait_for_selector("button.btn:not([disabled])", timeout=120000)  # model load
        page.screenshot(path=os.path.join(OUT, "01-upload.png"))

        page.click("button.btn")
        page.wait_for_selector(".diagnosis", timeout=90000)
        page.wait_for_timeout(1000)
        disease = page.locator(".disease-name").inner_text()
        conf = page.locator(".conflabel").inner_text()
        print(f"LIVE prediction: {disease} ({conf})")
        page.screenshot(path=os.path.join(OUT, "02-result-bn.png"), full_page=True)

        page.get_by_role("button", name="EN", exact=True).click()
        page.wait_for_timeout(700)
        page.screenshot(path=os.path.join(OUT, "03-result-en.png"), full_page=True)

        # a couple more to populate history
        page.get_by_role("button", name="বাংলা", exact=True).click()
        for label in ["Potato___Early_blight", "Corn___Common_rust", "Tomato___Late_blight"]:
            page.click("button.btn.secondary")
            page.set_input_files('input[type="file"]', img_for(label))
            page.wait_for_selector("button.btn:not([disabled])", timeout=30000)
            page.click("button.btn")
            page.wait_for_selector(".diagnosis", timeout=60000)
            page.wait_for_timeout(500)

        page.locator(".card").last.scroll_into_view_if_needed()
        page.wait_for_timeout(400)
        page.screenshot(path=os.path.join(OUT, "04-history.png"))
        browser.close()
        print("screenshots saved; live site verified")


if __name__ == "__main__":
    run()
