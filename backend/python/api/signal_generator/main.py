from fastapi import FastAPI
from apscheduler.schedulers.background import BackgroundScheduler
from api.signal_generator.router import router as signal_router
from api.signal_generator.service import run_signal_generation
from api.utils import is_market_open
import logging
import datetime as dt
import pytz

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ──────────────────────────────
# Initialize FastAPI
# ──────────────────────────────
app = FastAPI()
app.include_router(signal_router, prefix="/signal")

# ──────────────────────────────
# Scheduler Logic
# ──────────────────────────────
scheduler = BackgroundScheduler()

def scheduled_signal_job():
    logger.info("🔁 Scheduled signal generation triggered")
    if not is_market_open():
        logger.info("⏱ Market closed — skipping signal generation")
        return
    try:
        run_signal_generation()
        logger.info("✅ Signals generated")
    except Exception as e:
        logger.error(f"Signal generation error: {e}")

@app.on_event("startup")
async def start_scheduler():
    logger.info("🚀 Starting APScheduler for signal generation")
    tz = pytz.timezone("US/Eastern")
    # Schedule job every 15 minutes, with immediate first run
    scheduler.add_job(
        scheduled_signal_job,
        trigger="interval",
        seconds=900,
        next_run_time=dt.datetime.now(tz)
    )
    scheduler.start()
