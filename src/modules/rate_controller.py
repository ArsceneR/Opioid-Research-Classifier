import time
import logging
import instaloader

class MyRateController(instaloader.RateController):
    def sleep(self, secs: float):
        logging.info(f"Sleeping for {secs:.2f} seconds due to rate limiting...")
        time.sleep(secs)