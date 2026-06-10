"""
Configuration management for MarketCrunch Trading Agents
Loads all environment variables and constants
"""
import contextvars
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ============================================================================
# PATHS
# ============================================================================
ROOT = Path(__file__).parent.parent.parent
SRC_DIR = ROOT / "src"
DATA_DIR = ROOT / "data"
LOGS_DIR = ROOT / "logs"
REPORTS_DIR = ROOT / "reports"

# If we don't have write permissions in ROOT (e.g., on Vertex AI Reasoning Engine),
# redirect data, logs, and reports to a writable temporary directory.
try:
    for d in [DATA_DIR, LOGS_DIR, REPORTS_DIR]:
        d.mkdir(exist_ok=True)
except (PermissionError, OSError):
    import tempfile
    tmp_root = Path(tempfile.gettempdir()) / "mktcrunch"
    DATA_DIR = tmp_root / "data"
    LOGS_DIR = tmp_root / "logs"
    REPORTS_DIR = tmp_root / "reports"
    for d in [DATA_DIR, LOGS_DIR, REPORTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

# ============================================================================
# MARKETCRUNCH API
# ============================================================================
MC_API_KEY_ID = os.getenv("MC_API_KEY_ID")
MC_API_SECRET_KEY = os.getenv("MC_API_SECRET_KEY")
MC_API_URL = os.getenv("MC_API_URL", "https://mktcrunch-api-52245432644.us-central1.run.app")

if not MC_API_KEY_ID or not MC_API_SECRET_KEY:
    raise ValueError("Missing MC_API_KEY_ID or MC_API_SECRET_KEY environment variables")

# ============================================================================
# ALPACA TRADING (2 ACCOUNTS)
# ============================================================================
# Account 1: Baseline System
ALPACA_API_KEY_BASELINE = os.getenv("ALPACA_API_KEY_BASELINE")
ALPACA_SECRET_KEY_BASELINE = os.getenv("ALPACA_SECRET_KEY_BASELINE")

# Account 2: Internal System (with predictions)
ALPACA_API_KEY_INTERNAL = os.getenv("ALPACA_API_KEY_INTERNAL")
ALPACA_SECRET_KEY_INTERNAL = os.getenv("ALPACA_SECRET_KEY_INTERNAL")

# Alpaca base URL
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"

if not all([ALPACA_API_KEY_BASELINE, ALPACA_SECRET_KEY_BASELINE,
            ALPACA_API_KEY_INTERNAL, ALPACA_SECRET_KEY_INTERNAL]):
    raise ValueError("Missing Alpaca API keys for one or both accounts")

# ============================================================================
# POSTGRESQL DATABASE
# ============================================================================
DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", 5432))
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

DB_CONFIGURED = all([DB_HOST, DB_NAME, DB_USER, DB_PASSWORD])
DB_CONNECTION_STRING = (
    f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    if DB_CONFIGURED
    else None
)

# ============================================================================
# GOOGLE CLOUD / GEMINI
# ============================================================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_FLASH_MODEL = os.getenv("GEMINI_FLASH_MODEL", "gemini-3.5-flash")

# ============================================================================
# ADK (Agent Development Kit) — Google AI Agents Challenge
# ============================================================================
USE_ADK = os.getenv("USE_ADK", "true").lower() in ("1", "true", "yes")
USE_ADK_WORKFLOW = os.getenv("USE_ADK_WORKFLOW", "true").lower() in ("1", "true", "yes")
USE_ADK_MCP = os.getenv("USE_ADK_MCP", "false").lower() in ("1", "true", "yes")
# If coordinator LLM finishes without executing daily orders, run deterministic pipeline.
DAILY_COORDINATOR_FALLBACK = os.getenv("DAILY_COORDINATOR_FALLBACK", "true").lower() in (
    "1", "true", "yes",
)
USE_VERTEX_AI = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "false").lower() in ("1", "true", "yes")

_DISPLAY_NAME_PROJECT_IDS = frozenset({
    "mktcrunch-mvp", "MKTCrunch-MVP", "MKCRUNCH-MVP",
})


def _normalize_gcp_project(project: str) -> str:
    if not project:
        return ""
    if project in _DISPLAY_NAME_PROJECT_IDS:
        raise ValueError(
            f"GCP_PROJECT={project!r} looks like a console display name, not a project ID. "
            "Run: gcloud projects list --format='value(projectId)'"
        )
    return project


_raw_gcp_project = os.getenv("GCP_PROJECT", os.getenv("GOOGLE_CLOUD_PROJECT", ""))
try:
    GCP_PROJECT = _normalize_gcp_project(_raw_gcp_project)
except ValueError as e:
    import warnings
    warnings.warn(str(e))
    GCP_PROJECT = ""
GCP_REGION = os.getenv("GCP_REGION", "us-central1")
# Gemini 3.5+ on Vertex is served from the global endpoint; Agent Engine / Cloud Run stay regional.
GEMINI_VERTEX_LOCATION = os.getenv("GEMINI_VERTEX_LOCATION", "global")
if GCP_PROJECT:
    os.environ.setdefault("GOOGLE_CLOUD_PROJECT", GCP_PROJECT)
if USE_VERTEX_AI:
    os.environ.setdefault("GOOGLE_CLOUD_LOCATION", GEMINI_VERTEX_LOCATION)
else:
    os.environ.setdefault("GOOGLE_CLOUD_LOCATION", GCP_REGION)

# ============================================================================
# DATABENTO DATA DISCOVERY
# ============================================================================
DATABENTO_API_KEY = os.getenv("DATABENTO_API_KEY", "db-9Wivtyr8PmQGT7FuJsdPKDqCQnPpm")

# ============================================================================
# TRADING UNIVERSE & CONFIGURATION
# ============================================================================

# Twin Ledger live paper-trading experiment (both traders)
FIRST_TRADE_DATE = "2026-06-09"
FIRST_TRADE_DATE_LABEL = "June 9, 2026"

# 12 liquid ETFs ($1B+ AUM) spanning equities, sectors, bonds, commodities, intl.
# Chosen so agents must allocate across regimes — not win on a single-theme bet.
TRADING_UNIVERSE = [
    {"ticker": "SPY", "name": "S&P 500", "bucket": "US equity", "why": "Large-cap core — baseline US risk exposure"},
    {"ticker": "QQQ", "name": "Nasdaq-100", "bucket": "US equity", "why": "Growth/tech tilt — captures momentum regimes"},
    {"ticker": "IWM", "name": "Russell 2000", "bucket": "US equity", "why": "Small-cap — different cycle than large caps"},
    {"ticker": "VTI", "name": "Total Stock Market", "bucket": "US equity", "why": "Broadest US equity benchmark"},
    {"ticker": "XLF", "name": "Financials", "bucket": "Sector", "why": "Rate-sensitive sector rotation signal"},
    {"ticker": "XLE", "name": "Energy", "bucket": "Sector", "why": "Inflation/commodity cycle exposure"},
    {"ticker": "EFA", "name": "Intl Developed", "bucket": "International", "why": "Non-US developed markets diversification"},
    {"ticker": "TLT", "name": "20+ Year Treasuries", "bucket": "Rates", "why": "Duration/rate-move proxy — flight-to-quality"},
    {"ticker": "HYG", "name": "High Yield Bonds", "bucket": "Credit", "why": "Credit risk appetite vs. safe bonds"},
    {"ticker": "GLD", "name": "Gold", "bucket": "Commodity", "why": "Inflation hedge / risk-off alternative"},
    {"ticker": "SLV", "name": "Silver", "bucket": "Commodity", "why": "Industrial + precious metal beta"},
    {"ticker": "USO", "name": "Oil", "bucket": "Commodity", "why": "Energy price shock / geopolitical stress"},
]

UNIVERSE_RATIONALE = (
    "Twelve highly liquid ETFs ($1B+ AUM) across US equity sizes, sectors, "
    "international, rates, credit, and commodities — so neither agent can win "
    "by betting one theme; they must read the regime and rotate."
)

TICKER_UNIVERSE: List[str] = [t["ticker"] for t in TRADING_UNIVERSE]

# Trading configuration
TRADING_ENABLED = os.getenv("TRADING_ENABLED", "true").lower() == "true"
BACKTEST_MODE = os.getenv("BACKTEST_MODE", "false").lower() == "true"
DRY_RUN = os.getenv("DRY_RUN", "false").lower() == "true"

_dry_run_ctx: contextvars.ContextVar[bool] = contextvars.ContextVar("dry_run", default=False)


def is_dry_run() -> bool:
    """True when env DRY_RUN=true or a CLI session enabled dry_run_mode()."""
    return DRY_RUN or _dry_run_ctx.get()


@contextmanager
def dry_run_mode(enabled: bool = True) -> Iterator[None]:
    """Temporarily enable daily-workflow dry run (no Alpaca order placement)."""
    if not enabled:
        yield
        return
    token = _dry_run_ctx.set(True)
    try:
        yield
    finally:
        _dry_run_ctx.reset(token)

# ============================================================================
# SIGNAL & ALLOCATION CONFIG
# ============================================================================

# System A (Baseline) - Simple allocation
BASELINE_CONFIG = {
    "allocation_method": "twin_ledger",    # LLM structured decisions with size_pct
    "selection_method": "twin_ledger",     # Compete vs internal agent on leaderboard
    "competitor_system": "internal",
    "confidence_threshold": 0.5,           # Min confidence for BUY execution
    "max_positions": 8,                    # Max number of open positions
    "position_size_pct": 0.10,             # Max 10% per position
}

# System B (Internal) - Kelly-optimized with predictions
INTERNAL_CONFIG = {
    "allocation_method": "twin_ledger_kelly",  # Ledger decisions + Kelly BUY sizing
    "selection_method": "twin_ledger",           # Compete vs baseline on leaderboard
    "competitor_system": "baseline",
    "confidence_threshold": 0.55,                # Higher bar for BUY execution
    "max_positions": 8,
    "kelly_fraction": 0.25,                      # Conservative Kelly (1/4 full Kelly)
    "use_databento": True,                       # Enable DataBento data enrichment
}

# ============================================================================
# ORDER MANAGEMENT
# ============================================================================

ORDER_CONFIG = {
    "overnight_order_type": "limit",        # LIMIT orders overnight (preserve capital)
    "overnight_limit_offset_pct": 0.005,    # ±0.5% from MOC price
    "post_open_chase_threshold": 0.7,       # Chase if <70% filled
    "post_open_order_type": "market",       # Use market for unfilled chases
    "min_order_value": 100,                 # Min order size in dollars
    "max_order_value": 10000,               # Max order size in dollars
    "sell_delay_seconds": 30,               # Delay between sells and buys
    "reconcile_open_orders": True,          # Cancel duplicate OPG orders per symbol+price
    "delta_placement": True,                # Only place gap vs pending open orders
}

# ============================================================================
# DATA DISCOVERY (DATABENTO)
# ============================================================================

DISCOVERY_CONFIG = {
    "dataset": "EQUS.MINI",
    "schema": "ohlcv-1d",
    "min_history_years": 5,
    "min_universe_coverage_pct": 20.0,
    "max_sample_cost_usd": 1.00,
    "gate_1_mi_threshold": 0.02,
    "gate_2_ic_threshold": 0.02,
    "gate_2_t_stat_threshold": 1.25,
    "gate_3_incremental_alpha_threshold": 0.01,
    "sample_days": 90,
    "max_age_hours": 24,
    "end_lag_days": 3,
    # Agentic discovery: daily catalog scan + LLM probe planner
    "agentic_discovery": True,
    "llm_planner": True,
    "max_probes_per_day": 5,
    "probe_cooldown_hours": 168,      # 7 days before retrying rejected probes
    "reprobe_approved_hours": 24,     # refresh approved sources daily
    # LLM proposes new feature formulas per dataset/schema probe
    "llm_feature_planner": True,
    "max_features_per_probe": 6,
    "include_baseline_features": True,
}

APPROVED_SOURCES_PATH = DATA_DIR / "approved_datasources.json"

# ============================================================================
# INTRADAY RISK MONITORING
# ============================================================================

PREDICT_15MIN_URL = os.getenv(
    "PREDICT_15MIN_URL",
    "https://mktcrunch-prediction-min-api-fall-52245432644.us-central1.run.app/predict15min",
)

# Baseline: fixed base stop + LLM-generated trailing (NOT internal's 1%/70%)
BASELINE_RISK_CONFIG = {
    "base_stop_loss_threshold": -0.01,
    "trailing_mode": "llm",
    "use_trailing_stop": True,
    "llm_trailing_planner": True,
    "llm_trailing_bounds": {
        "activation_min": 0.005,
        "activation_max": 0.04,
        "lock_min": 0.45,
        "lock_max": 0.80,
    },
    "use_atr_base_stop": False,
    "use_15min_prediction_gate": False,
    "eod_exit_enabled": True,
    "eod_exit_threshold": -0.0095,
    "eod_window_minutes": 6.3,
    "atr_period": 14,
    "atr_stop_multiplier": 1.5,
}

# Internal: scripted floor (1%/70%) merged with LLM tightening
INTERNAL_RISK_CONFIG = {
    "base_stop_loss_threshold": -0.01,
    "trailing_mode": "hybrid",
    "trailing_activation_threshold": 0.01,
    "profit_lock_fraction": 0.70,
    "use_trailing_stop": True,
    "llm_trailing_planner": True,
    "llm_trailing_bounds": {
        "activation_min": 0.005,
        "activation_max": 0.02,
        "lock_min": 0.70,
        "lock_max": 0.85,
    },
    "use_atr_base_stop": True,
    "use_15min_prediction_gate": True,
    "eod_exit_enabled": True,
    "eod_exit_threshold": -0.0095,
    "eod_window_minutes": 6.3,
    "atr_period": 14,
    "atr_stop_multiplier": 1.5,
}

# Cloud Run / scheduler
SCHEDULER_SECRET = os.getenv("SCHEDULER_SECRET", "")
GCS_AUDIT_BUCKET = os.getenv("GCS_AUDIT_BUCKET", "")
GCS_DATA_BUCKET = os.getenv("GCS_DATA_BUCKET", os.getenv("GCS_RISK_STATE_BUCKET", ""))
GCS_RISK_STATE_BUCKET = os.getenv("GCS_RISK_STATE_BUCKET", GCS_DATA_BUCKET)
CLOUD_RUN_PORT = int(os.getenv("PORT", "8080"))

# Dashboard interactive chat (Decisions tab)
DASHBOARD_CHAT_ENABLED = os.getenv("DASHBOARD_CHAT_ENABLED", "true").lower() in (
    "1", "true", "yes",
)
# Read-only chat: status + audit tools only; no orders, risk runs, or workflows from the web UI
DASHBOARD_CHAT_READ_ONLY = os.getenv("DASHBOARD_CHAT_READ_ONLY", "true").lower() in (
    "1", "true", "yes",
)
# auto = Vertex Agent Engine when IDs are set, else local ADK coordinator
CHAT_BACKEND = os.getenv("CHAT_BACKEND", "auto")
AGENT_ENGINE_BASELINE_ID = os.getenv("AGENT_ENGINE_BASELINE_ID", "")
AGENT_ENGINE_INTERNAL_ID = os.getenv("AGENT_ENGINE_INTERNAL_ID", "")

# Audit trail
AUDIT_LOG_PATH = DATA_DIR / "audit_events.jsonl"
AUDIT_ENABLED = os.getenv("AUDIT_ENABLED", "true").lower() == "true"

# Agent learning loops (audit → reflection → prompt injection)
LEARNING_ENABLED = os.getenv("LEARNING_ENABLED", "true").lower() in ("1", "true", "yes")
LEARNING_USE_LLM = os.getenv("LEARNING_USE_LLM", "true").lower() in ("1", "true", "yes")
LEARNING_LOOKBACK_DAYS = int(os.getenv("LEARNING_LOOKBACK_DAYS", "7"))

# Baseline signal: Grounding with Google Search (public macro/news context)
BASELINE_GOOGLE_SEARCH_GROUNDING = os.getenv(
    "BASELINE_GOOGLE_SEARCH_GROUNDING", "true"
).lower() in ("1", "true", "yes")

# ============================================================================
# LOGGING
# ============================================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
LOG_FILE = LOGS_DIR / "trading_system.log"

# ============================================================================
# BACKTESTING
# ============================================================================

BACKTEST_CONFIG = {
    "start_date": "2026-01-01",
    "end_date": "2026-05-31",
    "initial_cash": 100000.0,
    "slippage_pct": 0.001,  # 0.1% slippage on all trades
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def print_config():
    """Print active configuration"""
    print("\n" + "=" * 80)
    print("MARKETCRUNCH TRADING AGENTS - CONFIGURATION")
    print("=" * 80)
    print(f"Root Directory:        {ROOT}")
    print(f"Data Directory:        {DATA_DIR}")
    print(f"Logs Directory:        {LOGS_DIR}")
    print(f"Reports Directory:     {REPORTS_DIR}")
    print(f"\nMarketCrunch API:      {MC_API_URL}")
    print(f"PostgreSQL:            {DB_HOST}:{DB_PORT}/{DB_NAME}")
    print(f"Alpaca Base URL:       {ALPACA_BASE_URL}")
    print(f"\nTicker Universe:       {len(TICKER_UNIVERSE)} tickers")
    print(f"Trading Enabled:       {TRADING_ENABLED}")
    print(f"Backtest Mode:         {BACKTEST_MODE}")
    print(f"Dry Run:               {DRY_RUN}")
    print(f"Log Level:             {LOG_LEVEL}")
    print("\n" + "=" * 80 + "\n")

if __name__ == "__main__":
    print_config()
