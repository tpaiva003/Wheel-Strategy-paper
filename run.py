"""Scheduler: run wheel_strategy.tick() every 15 minutes during market hours."""

import logging
import time
from datetime import datetime

from wheel_strategy import Broker, State, Wheel

POLL_SECONDS = 15 * 60  # 15 minutes
SLEEP_WHEN_CLOSED = 5 * 60  # 5 minutes when market closed

LOG = logging.getLogger("wheel.run")


def main() -> None:
    broker = Broker()
    while True:
        try:
            state = State.load()
            wheel = Wheel(broker, state)
            if broker.is_market_open():
                wheel.tick()
                sleep_for = POLL_SECONDS
            else:
                LOG.info("Market closed at %s - sleeping", datetime.now().isoformat(timespec="seconds"))
                sleep_for = SLEEP_WHEN_CLOSED
        except Exception as e:
            LOG.exception("Tick failed: %s", e)
            sleep_for = 60
        time.sleep(sleep_for)


if __name__ == "__main__":
    main()
