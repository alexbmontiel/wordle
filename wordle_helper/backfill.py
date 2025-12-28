import requests
import time
import os
from datetime import date, timedelta

BASE_URL = "https://www.nytimes.com/svc/wordle/v2/{date}.json"

START_DATE = date(2021, 6, 19)   # Wordle launch
END_DATE = date.today()

SLEEP_SECONDS = 1.5
OUTPUT_FILE = "wordle_answers.txt"


def daterange(start, end):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def load_existing_dates(path):
    """
    Reads existing file and returns a set of YYYY-MM-DD strings
    """
    dates = set()

    if not os.path.exists(path):
        return dates

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Expected format: YYYY-MM-DD WORD
            parts = line.split()
            if len(parts) >= 1:
                dates.add(parts[0])

    return dates


def fetch_wordle(date_obj):
    url = BASE_URL.format(date=date_obj.isoformat())
    r = requests.get(url, timeout=10)

    if r.status_code == 200:
        data = r.json()
        return data["print_date"], data["solution"]
    elif r.status_code == 404:
        return None
    else:
        r.raise_for_status()


def main():
    existing_dates = load_existing_dates(OUTPUT_FILE)
    print(f"Loaded {len(existing_dates)} existing entries")

    new_lines = []

    for d in daterange(START_DATE, END_DATE):
        date_str = d.isoformat()

        if date_str in existing_dates:
            continue

        try:
            result = fetch_wordle(d)
            if result:
                date_str, word = result
                line = f"{date_str} {word.upper()}"
                new_lines.append(line)
                print(f"✓ {line}")
            else:
                print(f"– {date_str} (no puzzle)")
        except Exception as e:
            print(f"⚠️  Error on {date_str}: {e}")

        time.sleep(SLEEP_SECONDS)

    if new_lines:
        with open(OUTPUT_FILE, "a") as f:
            for line in new_lines:
                f.write(line + "\n")

        print(f"\nAppended {len(new_lines)} new entries")
    else:
        print("\nNo new entries to add")


if __name__ == "__main__":
    main()

