from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import JSONResponse
from api.signal_generator.service import run_signal_generation
import logging
import os
from pathlib import Path
import pandas as pd

router = APIRouter()

@router.post("/generate-signals")
async def generate_signals(background_tasks: BackgroundTasks):
    """Trigger signal generation in background and return immediately."""
    background_tasks.add_task(run_signal_generation)
    logging.info("🟢 Signal generation enqueued")
    return {"status": "signal generation started"}

@router.get("/latest")
def get_latest_signals():
    try:
        BASE = Path(__file__).parent
        latest_file = BASE / "data" / "sent_signals" / "sent_signals_latest.csv"

        if not os.path.exists(latest_file):
            return JSONResponse(status_code=404, content={"error": "Latest signal file not found"})

        df = pd.read_csv(latest_file)
        return df.to_dict(orient="records")

    except Exception as e:
        logging.exception("❌ Error loading latest signal file")
        return JSONResponse(status_code=500, content={"error": str(e)})
