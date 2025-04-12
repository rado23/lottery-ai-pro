# backend/scheduler.py

from apscheduler.schedulers.background import BackgroundScheduler
from src.euromillions.euromillions_fetcher import fetch_draws as fetch_euromillions
from src.thunderball.download_thunderball_csv import save_thunderball_draws_csv
from src.lotto.download_lotto_csv import save_lotto_draws_csv
from src.setforlife.download_setforlife_csv import save_setforlife_draws_csv

def start_scheduler():
    scheduler = BackgroundScheduler()

    # EuroMillions – Daily
    scheduler.add_job(fetch_euromillions, 'cron', hour=6, minute=0, id="euromillions_refresh")

    # Thunderball – Mon–Sat
    scheduler.add_job(save_thunderball_draws_csv, 'cron', day_of_week='mon-sat', hour=6, minute=15, id="thunderball_refresh")

    # Lotto – Wed, Sat
    scheduler.add_job(save_lotto_draws_csv, 'cron', day_of_week='wed,sat', hour=6, minute=30, id="lotto_refresh")

    # Set for Life – Mon, Thu
    scheduler.add_job(save_setforlife_draws_csv, 'cron', day_of_week='mon,thu', hour=6, minute=45, id="setforlife_refresh")

    scheduler.start()
