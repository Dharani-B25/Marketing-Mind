"""Lightweight, public-page SEO checks for the MarketMind dashboard."""

from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse
import time

import requests


class _PageParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.title = ""
        self.in_title = False
        self.meta = {}
        self.h1 = []
        self.headings = []
        self.images = []
        self.links = []
        self.canonical = ""
        self.viewport = False
        self._heading = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "title":
            self.in_title = True
        elif tag == "meta":
            name = (attrs.get("name") or attrs.get("property") or "").lower()
            if name:
                self.meta[name] = attrs.get("content", "").strip()
            if name == "viewport":
                self.viewport = True
        elif tag == "link" and "canonical" in (attrs.get("rel") or "").lower():
            self.canonical = attrs.get("href", "").strip()
        elif tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._heading = [tag, ""]
        elif tag == "img":
            self.images.append(attrs.get("alt"))
        elif tag == "a" and attrs.get("href"):
            self.links.append(attrs["href"].strip())

    def handle_endtag(self, tag):
        if tag == "title":
            self.in_title = False
        if self._heading and tag == self._heading[0]:
            level, text = self._heading
            if level == "h1":
                self.h1.append(text.strip())
            self.headings.append((level, text.strip()))
            self._heading = None

    def handle_data(self, data):
        if self.in_title:
            self.title += data
        if self._heading:
            self._heading[1] += data


def _normalize_url(url):
    url = url.strip()
    if not url:
        raise ValueError("Enter a website URL.")
    if "://" not in url:
        url = "https://" + url
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Use a valid http:// or https:// website URL.")
    return url


def audit_page(url):
    url = _normalize_url(url)
    started = time.perf_counter()
    response = requests.get(
        url,
        timeout=12,
        headers={"User-Agent": "MarketMindAI-SEO-Audit/1.0"},
        allow_redirects=True,
    )
    elapsed = time.perf_counter() - started
    response.raise_for_status()
    content_type = response.headers.get("content-type", "").lower()
    if "html" not in content_type:
        raise ValueError("The URL did not return an HTML page.")

    parser = _PageParser()
    parser.feed(response.text[:2_000_000])
    final_url = response.url
    page_host = urlparse(final_url).netloc.lower()
    internal, external = [], []
    for href in parser.links:
        absolute = urljoin(final_url, href)
        parsed_link = urlparse(absolute)
        if parsed_link.scheme in {"http", "https"}:
            (internal if parsed_link.netloc.lower() == page_host else external).append(absolute)

    robots_url = urljoin(f"{urlparse(final_url).scheme}://{page_host}/", "robots.txt")
    sitemap_url = urljoin(f"{urlparse(final_url).scheme}://{page_host}/", "sitemap.xml")
    robots_ok = _url_available(robots_url)
    sitemap_ok = _url_available(sitemap_url)
    title = " ".join(parser.title.split())
    description = parser.meta.get("description", "")
    alt_count = sum(bool((alt or "").strip()) for alt in parser.images)
    checks = {
        "HTTPS enabled": urlparse(final_url).scheme == "https",
        "Page title present (and 30–60 chars)": bool(title) and 30 <= len(title) <= 60,
        "Meta description present (and 70–160 chars)": bool(description) and 70 <= len(description) <= 160,
        "Exactly one H1 heading": len(parser.h1) == 1,
        "Mobile viewport configured": parser.viewport,
        "Canonical URL present": bool(parser.canonical),
        "Images have alt text": not parser.images or alt_count == len(parser.images),
        "HTML page under 500 KB": len(response.content) <= 500 * 1024,
        "Robots.txt reachable": robots_ok,
        "Sitemap.xml reachable": sitemap_ok,
        "Response under 2 seconds": elapsed < 2,
    }
    return {
        "url": final_url,
        "status": response.status_code,
        "title": title,
        "description": description,
        "h1": parser.h1,
        "headings": parser.headings,
        "images": len(parser.images),
        "images_with_alt": alt_count,
        "internal_links": len(internal),
        "external_links": len(external),
        "response_seconds": elapsed,
        "page_kb": len(response.content) / 1024,
        "checks": checks,
    }


def _url_available(url):
    try:
        result = requests.get(url, timeout=5, headers={"User-Agent": "MarketMindAI-SEO-Audit/1.0"})
        return result.status_code < 400
    except requests.RequestException:
        return False


def recommendations(audit):
    checks = audit["checks"]
    items = []
    advice = {
        "HTTPS enabled": "Move the site to HTTPS and redirect HTTP traffic securely.",
        "Page title present (and 30–60 chars)": "Write a unique, descriptive title around 30–60 characters for this page.",
        "Meta description present (and 70–160 chars)": "Add a useful 70–160 character meta description that explains the page and encourages a click.",
        "Exactly one H1 heading": "Use one clear H1 that matches the page's main search intent.",
        "Mobile viewport configured": "Add a responsive viewport setting and review the page on mobile.",
        "Canonical URL present": "Add a canonical URL to help search engines identify the preferred page.",
        "Images have alt text": "Add concise, descriptive alt text to meaningful images.",
        "HTML page under 500 KB": "Reduce unnecessary HTML and inline code; keep large assets in separate optimized files.",
        "Robots.txt reachable": "Publish and review a robots.txt file so crawl rules are explicit.",
        "Sitemap.xml reachable": "Publish an XML sitemap and submit it through Google Search Console.",
        "Response under 2 seconds": "Investigate server response time and reduce heavy assets or slow requests.",
    }
    for name, passed in checks.items():
        if not passed:
            items.append({"priority": "High" if name in {"HTTPS enabled", "Exactly one H1 heading", "Mobile viewport configured"} else "Medium", "issue": name, "action": advice[name]})
    if audit["internal_links"] < 3:
        items.append({"priority": "Medium", "issue": "Few internal links detected", "action": "Link this page to relevant service, product, and supporting content pages."})
    if not items:
        items.append({"priority": "Maintain", "issue": "Core checks passed", "action": "Keep monitoring rankings, conversions, and page experience; this audit does not measure search visibility."})
    return items
