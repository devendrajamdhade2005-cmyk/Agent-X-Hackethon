"""Research connectors: arXiv, Semantic Scholar, OpenAlex."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import httpx

from . import simulation
from .base import RawItem, SourceConnector, SourceError, SourceQuery

_ATOM = {"a": "http://www.w3.org/2005/Atom"}


class ArxivConnector(SourceConnector):
    """arXiv Atom API — free, no key, the backbone of the research feed."""

    name = "arxiv"
    source_type = "research"
    label = "arXiv"
    requires_key = False
    rate_limit_per_min = 20  # arXiv asks for ~1 request / 3s
    timeout_seconds = 14.0
    docs_url = "https://info.arxiv.org/help/api/index.html"

    async def fetch(self, client: httpx.AsyncClient, q: SourceQuery) -> list[RawItem]:
        terms = q.keywords[:4] or [q.query]
        search = " OR ".join(f'all:"{t}"' for t in terms if t)
        resp = await self._get(
            client,
            "https://export.arxiv.org/api/query",
            params={
                "search_query": search,
                "start": 0,
                "max_results": q.limit,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            },
        )
        try:
            root = ET.fromstring(resp.text)
        except ET.ParseError as exc:
            raise SourceError(f"malformed Atom payload: {exc}", retryable=False) from exc

        items: list[RawItem] = []
        for entry in root.findall("a:entry", _ATOM):
            title = (entry.findtext("a:title", default="", namespaces=_ATOM) or "").strip()
            if not title:
                continue
            link = entry.findtext("a:id", default="", namespaces=_ATOM) or ""
            authors = [
                (a.findtext("a:name", default="", namespaces=_ATOM) or "").strip()
                for a in entry.findall("a:author", _ATOM)
            ]
            cats = [
                c.get("term", "")
                for c in entry.findall("a:category", _ATOM)
                if c.get("term")
            ]
            items.append(
                RawItem(
                    source_type=self.source_type,
                    source_name=self.name,
                    title=title,
                    url=link,
                    raw_text=entry.findtext("a:summary", default="", namespaces=_ATOM) or "",
                    author=", ".join(a for a in authors if a)[:300],
                    published_at=self._parse_date(
                        entry.findtext("a:published", default="", namespaces=_ATOM)
                    ),
                    external_id=link.rsplit("/", 1)[-1],
                    credibility="high",
                    meta={
                        "categories": cats,
                        "venue": "arXiv preprint",
                        "updated": entry.findtext("a:updated", default="", namespaces=_ATOM),
                    },
                )
            )
        return items

    def simulate(self, q: SourceQuery) -> list[RawItem]:
        return simulation.simulate(self.name, self.source_type, q)


class SemanticScholarConnector(SourceConnector):
    """Semantic Scholar Graph API — adds citation counts and venue metadata."""

    name = "semantic_scholar"
    source_type = "research"
    label = "Semantic Scholar"
    # Keyless access shares one heavily-throttled pool and returns 429 far more
    # often than data, so without a key we go straight to simulated items rather
    # than burn three retries and 3s of the run budget on a predictable failure.
    requires_key = True
    rate_limit_per_min = 20
    timeout_seconds = 15.0
    docs_url = "https://api.semanticscholar.org/api-docs/graph"

    _FIELDS = "title,abstract,authors,year,publicationDate,url,citationCount,venue,externalIds,openAccessPdf"

    async def fetch(self, client: httpx.AsyncClient, q: SourceQuery) -> list[RawItem]:
        headers = {"x-api-key": self.api_key} if self.api_key else {}
        resp = await self._get(
            client,
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={
                "query": q.query or " ".join(q.keywords[:3]),
                "limit": min(q.limit, 20),
                "fields": self._FIELDS,
            },
            headers=headers,
        )
        payload = resp.json()
        items: list[RawItem] = []
        for paper in payload.get("data", []) or []:
            title = (paper.get("title") or "").strip()
            if not title:
                continue
            authors = [a.get("name", "") for a in paper.get("authors") or []]
            ext = paper.get("externalIds") or {}
            url = paper.get("url") or (
                f"https://doi.org/{ext['DOI']}" if ext.get("DOI") else ""
            )
            items.append(
                RawItem(
                    source_type=self.source_type,
                    source_name=self.name,
                    title=title,
                    url=url,
                    raw_text=paper.get("abstract") or "",
                    author=", ".join(a for a in authors if a)[:300],
                    published_at=self._parse_date(paper.get("publicationDate"))
                    or self._parse_date(f"{paper.get('year')}-01-01" if paper.get("year") else None),
                    external_id=str(paper.get("paperId") or ext.get("DOI") or ""),
                    credibility="high",
                    meta={
                        "citation_count": paper.get("citationCount") or 0,
                        "venue": paper.get("venue") or "",
                        "doi": ext.get("DOI", ""),
                        "open_access_pdf": (paper.get("openAccessPdf") or {}).get("url", ""),
                    },
                )
            )
        return items

    def simulate(self, q: SourceQuery) -> list[RawItem]:
        return simulation.simulate(self.name, self.source_type, q)


class OpenAlexConnector(SourceConnector):
    """OpenAlex — free, no key, good institutional and concept metadata."""

    name = "openalex"
    source_type = "research"
    label = "OpenAlex"
    requires_key = False
    rate_limit_per_min = 60
    timeout_seconds = 15.0
    docs_url = "https://docs.openalex.org/"

    async def fetch(self, client: httpx.AsyncClient, q: SourceQuery) -> list[RawItem]:
        resp = await self._get(
            client,
            "https://api.openalex.org/works",
            params={
                "search": q.query or " ".join(q.keywords[:3]),
                "per-page": min(q.limit, 25),
                "sort": "publication_date:desc",
                "filter": f"from_publication_date:{q.since.date().isoformat()}",
                "mailto": "insightpulse@example.com",
            },
        )
        payload = resp.json()
        items: list[RawItem] = []
        for work in payload.get("results", []) or []:
            title = (work.get("title") or work.get("display_name") or "").strip()
            if not title:
                continue
            authorships = work.get("authorships") or []
            authors = [
                (a.get("author") or {}).get("display_name", "") for a in authorships[:6]
            ]
            institutions = [
                inst.get("display_name", "")
                for a in authorships[:6]
                for inst in (a.get("institutions") or [])
            ]
            abstract = work.get("abstract") or _invert_abstract(
                work.get("abstract_inverted_index") or {}
            )
            primary = work.get("primary_location") or {}
            source_meta = primary.get("source") or {}
            items.append(
                RawItem(
                    source_type=self.source_type,
                    source_name=self.name,
                    title=title,
                    url=work.get("doi") or primary.get("landing_page_url") or work.get("id") or "",
                    raw_text=abstract,
                    author=", ".join(a for a in authors if a)[:300],
                    published_at=self._parse_date(work.get("publication_date")),
                    external_id=str(work.get("id") or ""),
                    credibility="high",
                    meta={
                        "citation_count": work.get("cited_by_count") or 0,
                        "venue": source_meta.get("display_name") or "",
                        "institutions": sorted({i for i in institutions if i})[:5],
                        "concepts": [
                            c.get("display_name", "") for c in (work.get("concepts") or [])[:5]
                        ],
                        "type": work.get("type") or "",
                    },
                )
            )
        return items

    def simulate(self, q: SourceQuery) -> list[RawItem]:
        return simulation.simulate(self.name, self.source_type, q)


def _invert_abstract(index: dict[str, list[int]]) -> str:
    """OpenAlex ships abstracts as an inverted index; rebuild the prose."""
    if not index:
        return ""
    positions: list[tuple[int, str]] = []
    for word, spots in index.items():
        for spot in spots or []:
            positions.append((spot, word))
    positions.sort()
    return " ".join(word for _, word in positions)[:6000]
