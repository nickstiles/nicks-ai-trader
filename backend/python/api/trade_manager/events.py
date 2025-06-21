import asyncio

_queue: asyncio.Queue = asyncio.Queue()

def publish_update(payload: dict):
    """
    Call this whenever you want everyone listening to
    refresh their data. E.g. after handle_trade_signal()
    or after evaluate_trades().
    """
    _queue.put_nowait(payload)

async def get_event_generator():
    """
    Async generator that yields each payload as it arrives.
    """
    while True:
        data = await _queue.get()
        yield data