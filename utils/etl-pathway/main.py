# 1. Generate OHLCV streams (современный синтаксис windowby)
logger.info("Генерация OHLCV 1s...")
ohlcv_1s = trades.windowby(
    time_expr = pw.this.ts,
    window    = pw.temporal.tumbling(duration=1.0),
    instance  = pw.this.key,
).reduce(
    open=pw.reducers.arg_min(trades.price, trades.ts),
    high=pw.reducers.max(trades.price),
    low=pw.reducers.min(trades.price),
    close=pw.reducers.arg_max(trades.price, trades.ts),
    volume=pw.reducers.sum(trades.quantity),
    key=pw.this.key,
    ts=pw.this.ts,
)

logger.info("Генерация OHLCV 1m...")
ohlcv_1m = trades.windowby(
    time_expr = pw.this.ts,
    window    = pw.temporal.tumbling(duration=60.0),
    instance  = pw.this.key,
).reduce(
    open=pw.reducers.arg_min(trades.price, trades.ts),
    high=pw.reducers.max(trades.price),
    low=pw.reducers.min(trades.price),
    close=pw.reducers.arg_max(trades.price, trades.ts),
    volume=pw.reducers.sum(trades.quantity),
    key=pw.this.key,
    ts=pw.this.ts,
)

# 2. Generate features from OHLCV
logger.info("Generating 1s trade features...")
features_1s = create_trade_features(ohlcv_1s, "_s", feature_windows, beat_fee)

logger.info("Generating 1m trade features...")
features_1m = create_trade_features(ohlcv_1m, "_m", feature_windows, beat_fee) 