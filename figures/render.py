import base64
from playwright.sync_api import sync_playwright
from pathlib import Path

def render(html_path, output_dir):
    output = output_dir / (html_path.stem + ".png")
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        # Load the HTML content
        with open(html_path, "r") as file:
            html_content = file.read()

        page.set_content(html_content)

        page.wait_for_timeout(1000)

        element = page.locator("#img_A")
        data = element.get_attribute("src")

        data = data.split('base64,')[1]
        with open(output, "wb") as img_file:
            img_file.write(base64.b64decode(data))

        browser.close()

def render_all_html_files(directory):
    directory_path = Path(directory)
    output_dir = directory_path / "png"
    output_dir.mkdir(exist_ok=True)

    for file in directory_path.glob('*.html'):
        print(file)
        render(html_path=file, output_dir=output_dir)

# Call the function with the desired directory
render_all_html_files(".")
