import datetime as dt
import pytz
import pandas_market_calendars as mcal

def is_market_open(now=None):
    if now is None:
        now = dt.datetime.now(pytz.timezone("US/Eastern"))

    # First: Check day + time
    if not (now.weekday() < 5 and dt.time(9, 30) <= now.time() <= dt.time(16, 0)):
        return False

    # Second: Check holiday calendar
    nyse = mcal.get_calendar("NYSE")
    schedule = nyse.schedule(start_date=now.date(), end_date=now.date())
    return not schedule.empty

def get_last_n_trading_days(end_date: dt.date, n: int) -> list[dt.date]:
    """
    Return the last `n` NYSE trading dates strictly _before_ `end_date`.
    """
    # grab a calendar window that’s guaranteed to contain n trading days
    window_start = end_date - dt.timedelta(days=n * 3)
    nyse = mcal.get_calendar("NYSE")
    sched = nyse.schedule(start_date=window_start, end_date=end_date)
    trading_dates = list(sched.index.normalize().date)
    # drop end_date itself if it’s in the list
    trading_dates = [d for d in trading_dates if d < end_date]
    if len(trading_dates) < n:
        raise RuntimeError(f"Not enough trading days before {end_date!r}")
    return trading_dates[-n:]