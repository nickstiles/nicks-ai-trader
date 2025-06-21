from fastapi import FastAPI
from apscheduler.schedulers.background import BackgroundScheduler
from api.trade_manager.router import router as trade_router
from api.trade_manager.service import evaluate_trades, snapshot_account
from api.utils import is_market_open
import logging
import pytz
import datetime as dt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ──────────────────────────────
# Initialize FastAPI
# ──────────────────────────────
app = FastAPI()
app.include_router(trade_router, prefix="/trade")

# ──────────────────────────────
# Scheduler Setup
# ──────────────────────────────
scheduler = BackgroundScheduler()

def scheduled_trade_evaluation():
    logger.info("🔁 Scheduled trade evaluation triggered")
    if not is_market_open():
        logger.info("⏱ Market closed — skipping trade evaluation")
        return
    try:
        result = evaluate_trades()
        if result.get("closed_trades"):
            logger.info(f"✅ Auto-closed trades: {result['closed_trades']}")
    except Exception as e:
        logger.error(f"Trade evaluation error: {e}")

# only run when market’s open
def scheduled_snapshot_job():
    if is_market_open():
        try:
            snapshot_account()
        except Exception as e:
            logger.error("Failed to snapshot account: %s", e)

@app.on_event("startup")
def startup():
    logger.info("🚀 Starting Trade Manager + loading DB")
    tz = pytz.timezone("US/Eastern")

    # kick off the scheduler
    scheduler.add_job(
        scheduled_trade_evaluation,
        trigger="interval",
        seconds=900,
        next_run_time=dt.datetime.now(tz)
    )
    scheduler.add_job(
        scheduled_snapshot_job,
        trigger="interval",
        seconds=900,
        next_run_time=dt.datetime.now(tz)
    )
    scheduler.start()
    logger.info("✅ APScheduler started for trade evaluation")