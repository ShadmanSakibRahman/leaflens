"""Capture app screenshots for the README/report by driving the local app with a
headless mobile-sized browser. Requires backend on :8000 and frontend on :3000.
"""

import os
from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "docs", "screenshots")
SAMPLE = os.path.join(HERE, "..", "ml", "data", "Rice___Blast", "0003.jpg")

os.makedirs(OUT, exist_ok=True)


def run():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(
            viewport={"width": 412, "height": 900},
            device_scale_factor=2,
        )
        page = ctx.new_page()
        page.goto("http://localhost:3000", wait_until="networkidle")
        page.wait_for_timeout(1500)  # let the health ping resolve

        # 1. Home / upload view
        page.screenshot(path=os.path.join(OUT, "01-upload.png"))
        print("saved 01-upload.png")

        # Upload a leaf image and diagnose
        page.set_input_files('input[type="file"]', os.path.abspath(SAMPLE))
        page.wait_for_timeout(500)
        page.click(".btn")  # Diagnose
        page.wait_for_selector(".diagnosis", timeout=40000)
        page.wait_for_timeout(1200)

        # 2. Result in Bangla (default language)
        page.screenshot(path=os.path.join(OUT, "02-result-bn.png"), full_page=True)
        print("saved 02-result-bn.png")

        # 3. Toggle to English and re-screenshot the result
        page.get_by_role("button", name="EN", exact=True).click()
        page.wait_for_timeout(700)
        page.screenshot(path=os.path.join(OUT, "03-result-en.png"), full_page=True)
        print("saved 03-result-en.png")

        # 4. History section (last card on the page)
        page.locator(".card").last.scroll_into_view_if_needed()
        page.wait_for_timeout(400)
        page.screenshot(path=os.path.join(OUT, "04-history.png"))
        print("saved 04-history.png")

        browser.close()


if __name__ == "__main__":
    run()
