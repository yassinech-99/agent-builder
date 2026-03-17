"""Registry of prebuilt LangChain tools with per-tool config requirements.

Each tool entry declares:
  - name / description for UI display
  - category for grouping
  - required_config: list of dicts {key, label, type, hint}
  - factory(config) -> tool | list[tool]
  - is_toolkit: bool — when True the factory returns a list of tools
"""

from __future__ import annotations

import math
import os
from typing import Any, Callable

from langchain_core.tools import tool


# ---------------------------------------------------------------------------
# Built-in tools (zero external deps)
# ---------------------------------------------------------------------------

@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression. Supports basic arithmetic, power, sqrt, etc.
    Example: '2 + 3 * 4', 'sqrt(16)', '2**10'."""
    allowed = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
    allowed.update({"abs": abs, "round": round, "min": min, "max": max})
    try:
        result = eval(expression, {"__builtins__": {}}, allowed)  # noqa: S307
        return str(result)
    except Exception as exc:
        return f"Error evaluating expression: {exc}"


@tool
def current_datetime() -> str:
    """Return the current date and time in ISO-8601 format."""
    from datetime import datetime, timezone
    return datetime.now(tz=timezone.utc).isoformat()


@tool
def json_explorer(payload: str) -> str:
    """Parse and pretty-print a JSON string. Useful for inspecting API responses."""
    import json as _json
    try:
        data = _json.loads(payload)
        return _json.dumps(data, indent=2, ensure_ascii=False)[:4000]
    except Exception as exc:
        return f"JSON parse error: {exc}"


@tool
def sleep_tool(seconds: int) -> str:
    """Sleep for the specified number of seconds (max 30). Use to throttle or wait."""
    import time
    seconds = min(max(seconds, 1), 30)
    time.sleep(seconds)
    return f"Slept for {seconds} seconds."


@tool
def human_input(question: str) -> str:
    """Ask the human user a question and wait for their response. Only use when you truly need human confirmation."""
    return f"[HUMAN INPUT REQUESTED]: {question} — Please respond to continue."


# ---------------------------------------------------------------------------
# Factory helpers — lazy imports keep catalog load fast
# ---------------------------------------------------------------------------

# ── Search ────────────────────────────────────────────────────────────────

def _make_tavily(cfg: dict[str, str]):
    from langchain_tavily import TavilySearch
    os.environ.setdefault("TAVILY_API_KEY", cfg.get("TAVILY_API_KEY", ""))
    return TavilySearch(max_results=5)


def _make_duckduckgo(_cfg: dict[str, str]):
    from langchain_community.tools import DuckDuckGoSearchRun
    return DuckDuckGoSearchRun()


def _make_brave(cfg: dict[str, str]):
    from langchain_community.tools import BraveSearch
    return BraveSearch.from_api_key(
        api_key=cfg.get("BRAVE_SEARCH_API_KEY", ""),
        search_kwargs={"count": 5},
    )


def _make_google_serper(cfg: dict[str, str]):
    from langchain_community.utilities import GoogleSerperAPIWrapper
    from langchain_core.tools import Tool
    os.environ.setdefault("SERPER_API_KEY", cfg.get("SERPER_API_KEY", ""))
    wrapper = GoogleSerperAPIWrapper()
    return Tool(name="google_serper", func=wrapper.run,
                description="Search Google via Serper API.")


def _make_google_search(cfg: dict[str, str]):
    from langchain_community.utilities import GoogleSearchAPIWrapper
    from langchain_core.tools import Tool
    os.environ.setdefault("GOOGLE_API_KEY", cfg.get("GOOGLE_API_KEY", ""))
    os.environ.setdefault("GOOGLE_CSE_ID", cfg.get("GOOGLE_CSE_ID", ""))
    wrapper = GoogleSearchAPIWrapper()
    return Tool(name="google_search", func=wrapper.run,
                description="Search Google. Returns top results.")


def _make_serpapi(cfg: dict[str, str]):
    from langchain_community.utilities import SerpAPIWrapper
    from langchain_core.tools import Tool
    os.environ.setdefault("SERPAPI_API_KEY", cfg.get("SERPAPI_API_KEY", ""))
    wrapper = SerpAPIWrapper()
    return Tool(name="serpapi", func=wrapper.run,
                description="Search Google via SerpAPI. Returns rich snippets.")


def _make_youtube_search(_cfg: dict[str, str]):
    from langchain_community.tools import YouTubeSearchTool
    return YouTubeSearchTool()


# ── Knowledge & Research ──────────────────────────────────────────────────

def _make_wikipedia(_cfg: dict[str, str]):
    from langchain_community.tools import WikipediaQueryRun
    from langchain_community.utilities import WikipediaAPIWrapper
    return WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=3))


def _make_arxiv(_cfg: dict[str, str]):
    from langchain_community.tools import ArxivQueryRun
    from langchain_community.utilities import ArxivAPIWrapper
    return ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=3))


def _make_pubmed(_cfg: dict[str, str]):
    from langchain_community.tools import PubmedQueryRun
    return PubmedQueryRun()


def _make_stackexchange(_cfg: dict[str, str]):
    from langchain_community.utilities import StackExchangeAPIWrapper
    from langchain_core.tools import Tool
    wrapper = StackExchangeAPIWrapper()
    return Tool(name="stack_exchange", func=wrapper.run,
                description="Search Stack Overflow / Stack Exchange for programming Q&A.")


def _make_semanticscholar(_cfg: dict[str, str]):
    from langchain_community.tools.semanticscholar.tool import SemanticScholarQueryRun
    return SemanticScholarQueryRun()


# ── Computation ───────────────────────────────────────────────────────────

def _make_wolfram(cfg: dict[str, str]):
    from langchain_community.utilities import WolframAlphaAPIWrapper
    from langchain_core.tools import Tool
    os.environ.setdefault("WOLFRAM_ALPHA_APPID", cfg.get("WOLFRAM_ALPHA_APPID", ""))
    wrapper = WolframAlphaAPIWrapper()
    return Tool(name="wolfram_alpha", func=wrapper.run,
                description="Query Wolfram Alpha for computational and factual answers.")


# ── Code & System ─────────────────────────────────────────────────────────

def _make_python_repl(_cfg: dict[str, str]):
    from langchain_core.tools import Tool
    from langchain_community.utilities import PythonREPL
    repl = PythonREPL()
    return Tool(name="python_repl", func=repl.run,
                description="Execute Python code. Use for calculations and data processing.")


def _make_shell(_cfg: dict[str, str]):
    from langchain_community.tools import ShellTool
    return ShellTool()


def _make_e2b_sandbox(cfg: dict[str, str]):
    from langchain_community.tools.e2b_data_analysis.tool import E2BDataAnalysisTool
    return E2BDataAnalysisTool(api_key=cfg.get("E2B_API_KEY", ""))


# ── Web / HTTP ────────────────────────────────────────────────────────────

def _make_requests_get(_cfg: dict[str, str]):
    from langchain_community.tools import RequestsGetTool
    from langchain_community.utilities import TextRequestsWrapper
    return RequestsGetTool(requests_wrapper=TextRequestsWrapper())


def _make_requests_post(_cfg: dict[str, str]):
    from langchain_community.tools import RequestsPostTool
    from langchain_community.utilities import TextRequestsWrapper
    return RequestsPostTool(requests_wrapper=TextRequestsWrapper())


def _make_requests_delete(_cfg: dict[str, str]):
    from langchain_community.tools import RequestsDeleteTool
    from langchain_community.utilities import TextRequestsWrapper
    return RequestsDeleteTool(requests_wrapper=TextRequestsWrapper())


def _make_requests_put(_cfg: dict[str, str]):
    from langchain_community.tools import RequestsPutTool
    from langchain_community.utilities import TextRequestsWrapper
    return RequestsPutTool(requests_wrapper=TextRequestsWrapper())


def _make_requests_patch(_cfg: dict[str, str]):
    from langchain_community.tools import RequestsPatchTool
    from langchain_community.utilities import TextRequestsWrapper
    return RequestsPatchTool(requests_wrapper=TextRequestsWrapper())


# ── Email & Communication ─────────────────────────────────────────────────

def _make_gmail_toolkit(cfg: dict[str, str]):
    from langchain_community.agent_toolkits import GmailToolkit
    os.environ.setdefault("GMAIL_CREDENTIALS_FILE", cfg.get("GMAIL_CREDENTIALS_FILE", "credentials.json"))
    toolkit = GmailToolkit()
    return toolkit.get_tools()


def _make_o365_toolkit(cfg: dict[str, str]):
    from langchain_community.agent_toolkits import O365Toolkit
    os.environ.setdefault("O365_CLIENT_ID", cfg.get("O365_CLIENT_ID", ""))
    os.environ.setdefault("O365_CLIENT_SECRET", cfg.get("O365_CLIENT_SECRET", ""))
    os.environ.setdefault("O365_TENANT_ID", cfg.get("O365_TENANT_ID", ""))
    toolkit = O365Toolkit()
    return toolkit.get_tools()


# ── File & Data ───────────────────────────────────────────────────────────

def _make_file_management(_cfg: dict[str, str]):
    from langchain_community.agent_toolkits import FileManagementToolkit
    toolkit = FileManagementToolkit(root_dir="./workspace")
    return toolkit.get_tools()


# ── Browser ───────────────────────────────────────────────────────────────

def _make_nova_act_browser(cfg: dict[str, str]):
    from nova_act import NovaAct

    api_key = cfg.get("NOVA_ACT_API_KEY") or os.getenv("NOVA_ACT_API_KEY", "")

    @tool
    def nova_act_browse(url: str, instruction: str) -> str:
        """Automate browser workflows using Amazon Nova Act.
        Navigate to a URL and execute a natural language instruction
        such as filling forms, clicking buttons, or extracting data."""
        try:
            with NovaAct(starting_page=url, nova_act_api_key=api_key) as nova:
                result = nova.act(instruction)
                return str(result.response) if result.response else "Action completed."
        except Exception as exc:
            return f"Nova Act error: {exc}"

    return nova_act_browse


def _make_playwright_toolkit(_cfg: dict[str, str]):
    from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
    from langchain_community.tools.playwright.utils import create_sync_playwright_browser
    browser = create_sync_playwright_browser()
    toolkit = PlayWrightBrowserToolkit.from_browser(sync_browser=browser)
    return toolkit.get_tools()


# ── Project Management ────────────────────────────────────────────────────

def _make_jira_toolkit(cfg: dict[str, str]):
    from langchain_community.agent_toolkits.jira.toolkit import JiraToolkit
    from langchain_community.utilities.jira import JiraAPIWrapper
    os.environ.setdefault("JIRA_API_TOKEN", cfg.get("JIRA_API_TOKEN", ""))
    os.environ.setdefault("JIRA_USERNAME", cfg.get("JIRA_USERNAME", ""))
    os.environ.setdefault("JIRA_INSTANCE_URL", cfg.get("JIRA_INSTANCE_URL", ""))
    wrapper = JiraAPIWrapper()
    toolkit = JiraToolkit.from_jira_api_wrapper(wrapper)
    return toolkit.get_tools()


def _make_github_toolkit(cfg: dict[str, str]):
    from langchain_community.agent_toolkits.github.toolkit import GitHubToolkit
    from langchain_community.utilities.github import GitHubAPIWrapper
    os.environ.setdefault("GITHUB_APP_ID", cfg.get("GITHUB_APP_ID", ""))
    os.environ.setdefault("GITHUB_APP_PRIVATE_KEY", cfg.get("GITHUB_APP_PRIVATE_KEY", ""))
    os.environ.setdefault("GITHUB_REPOSITORY", cfg.get("GITHUB_REPOSITORY", ""))
    wrapper = GitHubAPIWrapper()
    toolkit = GitHubToolkit.from_github_api_wrapper(wrapper)
    return toolkit.get_tools()


# ── Creative / Generative ────────────────────────────────────────────────

def _make_dalle(cfg: dict[str, str]):
    from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
    from langchain_core.tools import Tool
    os.environ.setdefault("OPENAI_API_KEY", cfg.get("OPENAI_API_KEY", ""))
    wrapper = DallEAPIWrapper(model="dall-e-3")
    return Tool(name="dall_e", func=wrapper.run,
                description="Generate images using DALL-E 3. Provide a text description.")


def _make_openai_image(cfg: dict[str, str]):
    """Factory for an OpenAI gpt-image-1 image generation tool."""
    from openai import OpenAI  # Lazy import to avoid hard dependency at module import time

    api_key = cfg.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    client = OpenAI(api_key=api_key) if api_key else OpenAI()

    @tool
    def generate_image(prompt: str) -> str:
        """Generate an image using OpenAI gpt-image-1 and return an image URL."""
        try:
            response = client.images.generate(
                model="gpt-image-1",
                prompt=prompt,
                size="1024x1024",
            )
            # The v1 OpenAI client returns a data array; first element holds the image
            return response.data[0].url  # type: ignore[union-attr]
        except Exception as exc:  # pragma: no cover - defensive
            return f"Image generation failed: {exc}"

    return generate_image


# ── Weather & Data APIs ──────────────────────────────────────────────────

def _make_openweather(cfg: dict[str, str]):
    from langchain_community.utilities import OpenWeatherMapAPIWrapper
    from langchain_core.tools import Tool
    os.environ.setdefault("OPENWEATHERMAP_API_KEY", cfg.get("OPENWEATHERMAP_API_KEY", ""))
    wrapper = OpenWeatherMapAPIWrapper()
    return Tool(name="open_weather", func=wrapper.run,
                description="Get current weather data for a city.")


# ---------------------------------------------------------------------------
# The master catalog – ~30 tools & toolkits
# ---------------------------------------------------------------------------

TOOL_CATALOG: dict[str, dict[str, Any]] = {

    # ── Search ────────────────────────────────────────────────────────────
    "tavily_search": {
        "name": "Tavily Search",
        "description": "Premium web search powered by Tavily AI. Structured, relevant results.",
        "category": "search",
        "required_config": [
            {"key": "TAVILY_API_KEY", "label": "Tavily API Key", "type": "password", "hint": "Get from tavily.com"},
        ],
        "factory": _make_tavily,
    },
    "duckduckgo": {
        "name": "DuckDuckGo Search",
        "description": "Free web search via DuckDuckGo. No API key needed.",
        "category": "search",
        "required_config": [],
        "factory": _make_duckduckgo,
    },
    "brave_search": {
        "name": "Brave Search",
        "description": "Privacy-focused web search via Brave. Fast and accurate.",
        "category": "search",
        "required_config": [
            {"key": "BRAVE_SEARCH_API_KEY", "label": "Brave API Key", "type": "password", "hint": "Get from brave.com/search/api"},
        ],
        "factory": _make_brave,
    },
    "google_serper": {
        "name": "Google Search (Serper)",
        "description": "Google search via Serper API. Low-cost, high-quality results.",
        "category": "search",
        "required_config": [
            {"key": "SERPER_API_KEY", "label": "Serper API Key", "type": "password", "hint": "Get from serper.dev"},
        ],
        "factory": _make_google_serper,
    },
    "google_search": {
        "name": "Google Custom Search",
        "description": "Google search via Custom Search Engine API.",
        "category": "search",
        "required_config": [
            {"key": "GOOGLE_API_KEY", "label": "Google API Key", "type": "password", "hint": "Google Cloud Console"},
            {"key": "GOOGLE_CSE_ID", "label": "Custom Search Engine ID", "type": "text", "hint": "programmablesearchengine.google.com"},
        ],
        "factory": _make_google_search,
    },
    "serpapi": {
        "name": "SerpAPI",
        "description": "Google search via SerpAPI. Rich snippets, knowledge graph, PAA.",
        "category": "search",
        "required_config": [
            {"key": "SERPAPI_API_KEY", "label": "SerpAPI Key", "type": "password", "hint": "Get from serpapi.com"},
        ],
        "factory": _make_serpapi,
    },
    "youtube_search": {
        "name": "YouTube Search",
        "description": "Search YouTube videos by query. Returns video links.",
        "category": "search",
        "required_config": [],
        "factory": _make_youtube_search,
    },

    # ── Knowledge & Research ──────────────────────────────────────────────
    "wikipedia": {
        "name": "Wikipedia",
        "description": "Query Wikipedia for factual information and summaries.",
        "category": "knowledge",
        "required_config": [],
        "factory": _make_wikipedia,
    },
    "arxiv": {
        "name": "ArXiv",
        "description": "Search academic papers on arXiv. Great for research and science.",
        "category": "knowledge",
        "required_config": [],
        "factory": _make_arxiv,
    },
    "pubmed": {
        "name": "PubMed",
        "description": "Search PubMed for biomedical and life-science literature.",
        "category": "knowledge",
        "required_config": [],
        "factory": _make_pubmed,
    },
    "stack_exchange": {
        "name": "Stack Exchange",
        "description": "Search Stack Overflow / Stack Exchange for programming Q&A.",
        "category": "knowledge",
        "required_config": [],
        "factory": _make_stackexchange,
    },
    "semantic_scholar": {
        "name": "Semantic Scholar",
        "description": "Search academic papers on Semantic Scholar. AI-powered research discovery.",
        "category": "knowledge",
        "required_config": [],
        "factory": _make_semanticscholar,
    },

    # ── Computation ───────────────────────────────────────────────────────
    "calculator": {
        "name": "Calculator",
        "description": "Evaluate math expressions (arithmetic, sqrt, power, trig).",
        "category": "computation",
        "required_config": [],
        "factory": lambda _cfg: calculator,
    },
    "wolfram_alpha": {
        "name": "Wolfram Alpha",
        "description": "Computational knowledge engine. Complex math, science, data.",
        "category": "computation",
        "required_config": [
            {"key": "WOLFRAM_ALPHA_APPID", "label": "Wolfram Alpha App ID", "type": "password", "hint": "developer.wolframalpha.com"},
        ],
        "factory": _make_wolfram,
    },

    # ── Code & Sandbox ────────────────────────────────────────────────────
    "python_repl": {
        "name": "Python REPL",
        "description": "Execute Python code locally. Useful for data analysis and scripting.",
        "category": "code",
        "required_config": [],
        "factory": _make_python_repl,
    },
    "shell": {
        "name": "Shell / Bash",
        "description": "Execute shell commands on the host. Use with caution.",
        "category": "code",
        "required_config": [],
        "factory": _make_shell,
    },
    "e2b_sandbox": {
        "name": "E2B Code Sandbox",
        "description": "Cloud-sandboxed Python execution. Safe code running, chart generation, pip install.",
        "category": "code",
        "required_config": [
            {"key": "E2B_API_KEY", "label": "E2B API Key", "type": "password", "hint": "Get from e2b.dev"},
        ],
        "factory": _make_e2b_sandbox,
    },

    # ── Web / HTTP ────────────────────────────────────────────────────────
    "requests_get": {
        "name": "HTTP GET",
        "description": "Make HTTP GET requests to any URL. APIs, web scraping.",
        "category": "web",
        "required_config": [],
        "factory": _make_requests_get,
    },
    "requests_post": {
        "name": "HTTP POST",
        "description": "Make HTTP POST requests. Send data to APIs.",
        "category": "web",
        "required_config": [],
        "factory": _make_requests_post,
    },
    "requests_put": {
        "name": "HTTP PUT",
        "description": "Make HTTP PUT requests. Update resources via APIs.",
        "category": "web",
        "required_config": [],
        "factory": _make_requests_put,
    },
    "requests_patch": {
        "name": "HTTP PATCH",
        "description": "Make HTTP PATCH requests. Partial updates via APIs.",
        "category": "web",
        "required_config": [],
        "factory": _make_requests_patch,
    },
    "requests_delete": {
        "name": "HTTP DELETE",
        "description": "Make HTTP DELETE requests. Remove resources via APIs.",
        "category": "web",
        "required_config": [],
        "factory": _make_requests_delete,
    },

    # ── Email & Communication ─────────────────────────────────────────────
    "gmail_toolkit": {
        "name": "Gmail Toolkit",
        "description": "Read, search, draft, and send Gmail emails. Full inbox management.",
        "category": "email",
        "required_config": [
            {"key": "GMAIL_CREDENTIALS_FILE", "label": "Gmail credentials.json path", "type": "text", "hint": "Path to Google OAuth credentials.json"},
        ],
        "is_toolkit": True,
        "factory": _make_gmail_toolkit,
    },
    "o365_toolkit": {
        "name": "Office 365 Toolkit",
        "description": "Microsoft O365 — search emails, events, send messages, create drafts.",
        "category": "email",
        "required_config": [
            {"key": "O365_CLIENT_ID", "label": "O365 Client ID", "type": "text", "hint": "Azure App Registration"},
            {"key": "O365_CLIENT_SECRET", "label": "O365 Client Secret", "type": "password", "hint": "Azure App Secret"},
            {"key": "O365_TENANT_ID", "label": "O365 Tenant ID", "type": "text", "hint": "Azure tenant ID"},
        ],
        "is_toolkit": True,
        "factory": _make_o365_toolkit,
    },

    # ── File & Data ───────────────────────────────────────────────────────
    "file_management": {
        "name": "File Management Toolkit",
        "description": "Read, write, copy, move, delete, and list local files in ./workspace.",
        "category": "file",
        "required_config": [],
        "is_toolkit": True,
        "factory": _make_file_management,
    },
    "json_explorer": {
        "name": "JSON Explorer",
        "description": "Parse and pretty-print JSON strings. Inspect API responses.",
        "category": "file",
        "required_config": [],
        "factory": lambda _cfg: json_explorer,
    },

    # ── Browser Automation ────────────────────────────────────────────────
    "nova_act_browser": {
        "name": "Nova Act Browser",
        "description": "Automate browser workflows using Amazon Nova Act. Navigate pages, fill forms, extract data with natural language instructions.",
        "category": "browser",
        "required_config": [
            {"key": "NOVA_ACT_API_KEY", "label": "Nova Act API Key", "type": "password", "hint": "Get from nova.amazon.com/act"},
        ],
        "factory": _make_nova_act_browser,
    },
    "playwright_browser": {
        "name": "Playwright Browser Toolkit",
        "description": "Automate a browser — navigate, click, fill forms, extract text.",
        "category": "browser",
        "required_config": [],
        "is_toolkit": True,
        "factory": _make_playwright_toolkit,
    },

    # ── Project Management ────────────────────────────────────────────────
    "jira_toolkit": {
        "name": "Jira Toolkit",
        "description": "Create, update, search, and manage Jira issues and projects.",
        "category": "project",
        "required_config": [
            {"key": "JIRA_INSTANCE_URL", "label": "Jira URL", "type": "text", "hint": "https://yourcompany.atlassian.net"},
            {"key": "JIRA_USERNAME", "label": "Jira Email", "type": "text", "hint": "your@email.com"},
            {"key": "JIRA_API_TOKEN", "label": "Jira API Token", "type": "password", "hint": "Get from id.atlassian.com"},
        ],
        "is_toolkit": True,
        "factory": _make_jira_toolkit,
    },
    "github_toolkit": {
        "name": "GitHub Toolkit",
        "description": "Manage GitHub repos — issues, PRs, files, comments, and more.",
        "category": "project",
        "required_config": [
            {"key": "GITHUB_APP_ID", "label": "GitHub App ID", "type": "text", "hint": "GitHub App settings"},
            {"key": "GITHUB_APP_PRIVATE_KEY", "label": "Private Key (PEM)", "type": "password", "hint": "GitHub App private key"},
            {"key": "GITHUB_REPOSITORY", "label": "Repository", "type": "text", "hint": "owner/repo"},
        ],
        "is_toolkit": True,
        "factory": _make_github_toolkit,
    },

    # ── Creative / Generative ────────────────────────────────────────────
    "dall_e": {
        "name": "DALL-E Image Generation",
        "description": "Generate images using OpenAI DALL-E 3 from text descriptions.",
        "category": "creative",
        "required_config": [
            {"key": "OPENAI_API_KEY", "label": "OpenAI API Key", "type": "password", "hint": "Uses OpenAI API key for DALL-E"},
        ],
        "factory": _make_dalle,
    },
    "openai_image": {
        "name": "OpenAI Image (gpt-image-1)",
        "description": "Generate images using OpenAI gpt-image-1 from text prompts.",
        "category": "creative",
        "required_config": [
            {
                "key": "OPENAI_API_KEY",
                "label": "OpenAI API Key",
                "type": "password",
                "hint": "Optional here if already set in environment",
            },
        ],
        "factory": _make_openai_image,
    },

    # ── Data APIs ─────────────────────────────────────────────────────────
    "open_weather": {
        "name": "OpenWeatherMap",
        "description": "Get current weather data for any city worldwide.",
        "category": "data",
        "required_config": [
            {"key": "OPENWEATHERMAP_API_KEY", "label": "OpenWeatherMap API Key", "type": "password", "hint": "openweathermap.org/api"},
        ],
        "factory": _make_openweather,
    },
    "meilisearch_mcp": {
        "name": "Meilisearch (Chat with Data)",
        "description": "Full Meilisearch access via MCP — search with filters/facets, manage documents and indexes, monitor health. Powers 'chat with your data'.",
        "category": "data",
        "required_config": [
            {"key": "MEILI_HTTP_ADDR", "label": "Meilisearch URL", "type": "text", "hint": "http://172.30.6.3:7700"},
        ],
        "is_toolkit": True,
        "factory": lambda _cfg: [],
    },

    # ── Utility ───────────────────────────────────────────────────────────
    "current_datetime": {
        "name": "Current Date/Time",
        "description": "Returns the current date and time in UTC ISO-8601 format.",
        "category": "utility",
        "required_config": [],
        "factory": lambda _cfg: current_datetime,
    },
    "sleep": {
        "name": "Sleep / Wait",
        "description": "Pause execution for a number of seconds (1-30). Useful for rate limiting.",
        "category": "utility",
        "required_config": [],
        "factory": lambda _cfg: sleep_tool,
    },
    "human_input": {
        "name": "Human Input",
        "description": "Ask the human user a clarifying question before proceeding.",
        "category": "utility",
        "required_config": [],
        "factory": lambda _cfg: human_input,
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_tools_by_ids(
    tool_ids: list[str],
    tool_configs: dict[str, dict[str, str]] | None = None,
) -> list[Callable[..., Any]]:
    """Instantiate and return tools for the given catalog IDs.

    Handles both single-tool entries and toolkit entries (is_toolkit=True)
    that return a list of tools from their factory.
    """
    configs = tool_configs or {}
    tools: list[Callable[..., Any]] = []
    for tid in tool_ids:
        entry = TOOL_CATALOG.get(tid)
        if entry is None:
            continue
        cfg = configs.get(tid, {})
        try:
            result = entry["factory"](cfg)
            if entry.get("is_toolkit") and isinstance(result, list):
                tools.extend(result)
            else:
                tools.append(result)
        except Exception:
            pass
    return tools


def get_catalog_info() -> list[dict[str, Any]]:
    """Return serialisable catalog metadata for the frontend."""
    return [
        {
            "id": tid,
            "name": entry["name"],
            "description": entry["description"],
            "category": entry.get("category", "general"),
            "required_config": entry.get("required_config", []),
            "is_toolkit": entry.get("is_toolkit", False),
        }
        for tid, entry in TOOL_CATALOG.items()
    ]
