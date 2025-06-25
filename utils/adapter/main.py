import asyncio
import time
import json
from collections import defaultdict
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
# from tritonclient.grpc.aio import InferenceServerClient  # если нужен async Triton

DEBUG_MODE = True  # True — только print, False — боевой режим с Kafka

KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
ORDERBOOK_FEATURES_TOPIC = "features_order_book"
TRADE_FEATURES_1S_TOPIC = "features_trade_1s"
TRADE_FEATURES_1M_TOPIC = "features_trade_1m"
HIGH_LEVEL_SIGNAL_TOPIC = "robot_signal_high"  # верхний уровень (1m)
LOW_LEVEL_SIGNAL_TOPIC = "robot_signal_low"    # нижний уровень (ob+1s)
TRITON_SERVER_URL = "localhost:8001"  # или порт gRPC

async def process_low_level_signals(buffer_low, key, second, timestamp, producer):
    features_orderbook = buffer_low[(key, second)]["orderbook"]
    features_1s = buffer_low[(key, second)]["trade_1s"]
    features = {**features_orderbook, **features_1s}
    if DEBUG_MODE:
        print(f"[DEBUG] Would send to Triton (LOW): key={key}, ts={timestamp}, features={features}")
    else:
        # signal = await triton_low.infer(features)
        signal = {"signal": "stub_low"}
        await producer.send_and_wait(
            LOW_LEVEL_SIGNAL_TOPIC,
            json.dumps({"key": key, "ts": timestamp, "signal": signal}).encode("utf-8")
        )
    del buffer_low[(key, second)]

async def process_high_level_signals(buffer_high, key, minute, timestamp, producer):
    features_1m = buffer_high[(key, minute)]
    if DEBUG_MODE:
        print(f"[DEBUG] Would send to Triton (HIGH): key={key}, ts={timestamp}, features={features_1m}")
    else:
        # signal = await triton_high.infer(features_1m)
        signal = {"signal": "stub_high"}
        await producer.send_and_wait(
            HIGH_LEVEL_SIGNAL_TOPIC,
            json.dumps({"key": key, "ts": timestamp, "signal": signal}).encode("utf-8")
        )
    del buffer_high[(key, minute)]

async def consume_and_route_signals(consumer, producer):
    buffer_low = defaultdict(dict)   # (key, second) -> {"orderbook": ..., "trade_1s": ...}
    buffer_high = {}                # (key, minute) -> features_1m

    async for message in consumer:
        value = message.value
        topic = message.topic
        key = value.get("key")
        timestamp = int(float(value.get("ts", time.time())))
        second = timestamp
        minute = timestamp // 60

        if topic == ORDERBOOK_FEATURES_TOPIC:
            buffer_low[(key, second)]["orderbook"] = value
        elif topic == TRADE_FEATURES_1S_TOPIC:
            buffer_low[(key, second)]["trade_1s"] = value
        elif topic == TRADE_FEATURES_1M_TOPIC:
            buffer_high[(key, minute)] = value

        # Если есть оба типа признаков для нижнего уровня — отправляем сигнал
        if all(x in buffer_low[(key, second)] for x in ["orderbook", "trade_1s"]):
            await process_low_level_signals(buffer_low, key, second, timestamp, producer)

        # Если есть признаки для верхнего уровня — отправляем сигнал
        if (key, minute) in buffer_high:
            await process_high_level_signals(buffer_high, key, minute, timestamp, producer)

async def debug_simulate_messages():
    """
    Генерирует тестовые сообщения для debug-режима (без Kafka).
    """
    buffer_low = defaultdict(dict)
    buffer_high = {}
    key = "test_key"
    now = int(time.time())
    # Симулируем приход orderbook и trade_1s
    buffer_low[(key, now)]["orderbook"] = {"key": key, "ts": now, "ob": 123}
    buffer_low[(key, now)]["trade_1s"] = {"key": key, "ts": now, "t1s": 456}
    await process_low_level_signals(buffer_low, key, now, now, None)
    # Симулируем приход trade_1m
    buffer_high[(key, now // 60)] = {"key": key, "ts": now, "t1m": 789}
    await process_high_level_signals(buffer_high, key, now // 60, now, None)

async def main():
    if DEBUG_MODE:
        print("[DEBUG] Debug mode enabled. No Kafka will be used.")
        await debug_simulate_messages()
        return
    consumer = AIOKafkaConsumer(
        ORDERBOOK_FEATURES_TOPIC,
        TRADE_FEATURES_1S_TOPIC,
        TRADE_FEATURES_1M_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset="latest",
        enable_auto_commit=True,
        group_id="triton-adapter"
    )
    producer = AIOKafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
    )
    await consumer.start()
    await producer.start()
    try:
        await consume_and_route_signals(consumer, producer)
    finally:
        await consumer.stop()
        await producer.stop()

if __name__ == "__main__":
    asyncio.run(main()) 