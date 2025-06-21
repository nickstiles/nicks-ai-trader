from typing import List, Any
import json

from fastapi import APIRouter, Request, BackgroundTasks
from fastapi.responses import StreamingResponse

from api.signal_generator.models import TradeSignal
from api.trade_manager.models import TradeResponse
from api.trade_manager.service import handle_trade_signal, evaluate_trades, compute_metrics
from api.trade_manager.storage import get_open_trades, get_closed_trades, get_account_snapshots
from api.trade_manager.events import get_event_generator

router = APIRouter()

@router.post("/signal", response_model=Any)
def receive_trade_signal(signal: TradeSignal):
    """
    Accept a TradeSignal and forward to the trade manager service.
    """
    print(f"📥 Trade Manager received signal: {signal.option_symbol}")
    return handle_trade_signal(signal)

@router.post("/evaluate", response_model=Any)
async def run_evaluate_trades(background_tasks: BackgroundTasks):
    """
    Trigger evaluation of open trades (stop-loss/take-profit/time expiry) in the background.
    """
    background_tasks.add_task(evaluate_trades)
    return {"status": "trade evaluation started"}

@router.get("/open", response_model=List[TradeResponse])
def list_open_trades():
    """
    Return the list of open trades, including live price and PnL stats.
    """
    return get_open_trades()

@router.get("/closed", response_model=List[TradeResponse])
def list_closed_trades():
    """
    Return the list of closed trades and their exit details.
    """
    return get_closed_trades()

@router.get("/metrics", response_model=Any)
def list_metrics():
    """
    Return aggregated performance metrics for the trade manager.
    """
    return compute_metrics()

@router.get("/snapshots")
def list_snapshots(limit: int = 100):
    """
    Return account snapshots for the trade manager.
    """
    return get_account_snapshots()

@router.get("/stream")
async def trade_updates(request: Request):
    """
    Server-Sent Events endpoint. Emits a JSON payload
    each time you call publish_update(...) server-side.
    """
    async def event_generator():
        async for payload in get_event_generator():
            # if client disconnected, stop
            if await request.is_disconnected():
                break
            # SSE format: "data: <json>\n\n"
            yield f"data: {json.dumps(payload)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")