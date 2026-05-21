"""
Browser Tool — provides ARIA with visual agency and web browsing capabilities.

Uses Playwright for stealthy, headless browsing and html2text for markdown extraction.
Supports navigation, scraping, clicking, typing, and 1080p screenshots.
"""

import json
import ipaddress
import socket
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse
import re

try:
    from playwright.async_api import async_playwright, Page, Browser, BrowserContext
    import html2text
except ImportError:
    pass

try:
    from playwright_stealth import stealth_async
except ImportError:
    stealth_async = None

from silex.tools.base import BaseTool
from silex.utils.config import browser_actions_enabled, WORKSPACE_DIR
from silex.utils.logger import setup_logger

log = setup_logger("silex.tools.browser")
BROWSER_OUTPUT_DIR = WORKSPACE_DIR / "browser"
ALLOWED_SCHEMES = {"http", "https"}


def _validate_public_url(url: str) -> None:
    """Validate that a URL points to a public internet address.

    Known limitation — DNS rebinding (TOCTOU):
      This function resolves the hostname and checks if the IP is private.
      However, the actual HTTP request happens *after* this validation.
      An attacker could set up a DNS record that resolves to a public IP
      during validation, then changes to 127.0.0.1 by the time Playwright
      makes the request (a DNS rebinding attack).

      The proper fix is to pin the resolved IP and pass it to Playwright via
      ``--host-resolver-rules``. This is tracked as a known security backlog
      item. Risk is low for local-only deployments.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ALLOWED_SCHEMES or not parsed.hostname:
        raise ValueError("Only http and https URLs with a hostname are allowed.")

    try:
        addresses = socket.getaddrinfo(parsed.hostname, None)
    except socket.gaierror:
        raise ValueError("Could not resolve URL hostname.") from None

    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast:
            raise ValueError("Refusing to browse private or local network addresses.")


def _resolve_screenshot_path(filepath: str) -> Path:
    BROWSER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    candidate = Path(filepath)
    if not candidate.is_absolute():
        candidate = BROWSER_OUTPUT_DIR / candidate.name
    resolved = candidate.resolve()
    try:
        resolved.relative_to(BROWSER_OUTPUT_DIR)
    except ValueError:
        raise ValueError("Screenshot path must stay inside workspace/browser.")
    return resolved

class BrowserTool(BaseTool):
    """
    A tool that allows ARIA to interact with the web.
    """
    name = "browser"
    risk_level = "network"
    description = "Browse the web, scrape content as markdown, and take screenshots."
    schema = {
        "action": "str: 'navigate', 'scrape', 'screenshot', 'click', 'type'",
        "url": "str, optional: for navigate",
        "selector": "str, optional: for click/type",
        "text": "str, optional: for type",
        "filepath": "str, optional: for screenshot (default: aria_vision_capture.png)"
    }

    def __init__(self):
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None

    async def _ensure_started(self):
        """Lazy initialization of the browser."""
        if self.page is not None:
            return

        log.info("Starting headless Chromium instance...")
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=True)
        self.context = await self.browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
        )
        self.page = await self.context.new_page()
        
        if stealth_async:
            await stealth_async(self.page)

    async def execute(self, **kwargs) -> str:
        """
        Execute a browser action.
        """
        if not browser_actions_enabled():
            return "Error: Browser automation is disabled by ARIA_ENABLE_BROWSER_ACTIONS."

        action = kwargs.get("action")
        await self._ensure_started()

        try:
            if action == "navigate":
                url = kwargs.get("url")
                if not url:
                    return "Error: Missing 'url' for navigate action."
                _validate_public_url(url)
                await self.page.goto(url, wait_until="domcontentloaded", timeout=30000)
                return await self._observation("navigate", f"Successfully navigated to {url}")

            elif action == "scrape":
                html_content = await self.page.content()
                h = html2text.HTML2Text()
                h.ignore_links = False
                h.ignore_images = True
                h.body_width = 0
                markdown = h.handle(html_content)
                # Strip known prompt injection keywords
                markdown = re.sub(r'(?i)(system instruction|critical instruction|ignore previous|you are now|system override|forget all)', '[REDACTED]', markdown)
                return await self._observation("scrape", markdown[:8000])

            elif action == "screenshot":
                filepath = kwargs.get("filepath", "aria_vision_capture.png")
                safe_path = _resolve_screenshot_path(filepath)
                await self.page.screenshot(path=str(safe_path), full_page=False)
                return await self._observation("screenshot", f"Screenshot saved to {safe_path}", screenshot_path=str(safe_path))

            elif action == "click":
                selector = kwargs.get("selector")
                if not selector:
                    return "Error: Missing 'selector' for click action."
                await self.page.click(selector, timeout=5000)
                return await self._observation("click", f"Clicked element: {selector}")

            elif action == "type":
                selector = kwargs.get("selector")
                text = kwargs.get("text")
                if not selector or text is None:
                    return "Error: Missing 'selector' or 'text' for type action."
                await self.page.fill(selector, text, timeout=5000)
                return await self._observation("type", f"Typed text into: {selector}")

            else:
                return f"Error: Unknown browser action '{action}'."

        except Exception as e:
            log.error(f"Browser action '{action}' failed: {str(e)}")
            return f"Error executing browser action: {str(e)}"

    async def close(self):
        """Cleanup."""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
        self.page = None
        self.context = None
        self.browser = None
        self.playwright = None

    async def _page_state(self) -> dict:
        if not self.page:
            return {}
        title = await self.page.title()
        try:
            visible_text = await self.page.locator("body").inner_text(timeout=2000)
        except Exception:
            visible_text = ""
        return {
            "url": self.page.url,
            "title": title,
            "visible_text_preview": visible_text[:1000],
        }

    async def _observation(self, action: str, result: str, **extra) -> str:
        payload = {
            "action": action,
            "result": result,
            "page": await self._page_state(),
            **extra,
        }
        return "BROWSER_OBSERVATION:\n" + json.dumps(payload, indent=2)
