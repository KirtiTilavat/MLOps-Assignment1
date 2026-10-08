"""Send a stream of synthetic patient requests to the API so the monitoring
dashboard has something to show. Uses only the standard library.

Usage:
    python scripts/generate_traffic.py                       # http://localhost:8000, 200 requests
    python scripts/generate_traffic.py --url http://localhost --n 500 --delay 0.05
"""

import argparse
import json
import random
import time
import urllib.error
import urllib.request


def random_patient(rng: random.Random) -> dict:
    sick = rng.random() < 0.45  # roughly the dataset's class balance
    return {
        "age": rng.randint(35, 75),
        "sex": rng.choice([0, 1, 1]),
        "cp": rng.choice([4, 4, 3]) if sick else rng.choice([1, 2, 3]),
        "trestbps": rng.randint(100, 180),
        "chol": rng.randint(160, 340),
        "fbs": int(rng.random() < 0.15),
        "restecg": rng.choice([0, 2]),
        "thalach": rng.randint(95, 150) if sick else rng.randint(140, 195),
        "exang": int(rng.random() < (0.6 if sick else 0.15)),
        "oldpeak": round(rng.uniform(0.8, 4.0) if sick else rng.uniform(0, 1.2), 1),
        "slope": rng.choice([2, 2, 3]) if sick else rng.choice([1, 1, 2]),
        "ca": rng.choice([1, 2, 3]) if sick else rng.choice([0, 0, 0, 1]),
        "thal": rng.choice([7, 7, 6]) if sick else rng.choice([3, 3, 7]),
    }


def post(url: str, body: dict) -> int:
    req = urllib.request.Request(
        f"{url}/predict",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        return e.code


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--n", type=int, default=200, help="number of requests")
    parser.add_argument("--delay", type=float, default=0.1, help="seconds between requests")
    parser.add_argument("--invalid-rate", type=float, default=0.05, help="share of bad payloads")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    counts: dict[int, int] = {}
    for i in range(1, args.n + 1):
        patient = random_patient(rng)
        if rng.random() < args.invalid_rate:
            patient["age"] = 500  # out of range -> 422, shows up in the error panels
        status = post(args.url.rstrip("/"), patient)
        counts[status] = counts.get(status, 0) + 1
        if i % 50 == 0:
            print(f"{i}/{args.n} sent, status counts so far: {counts}")
        time.sleep(args.delay)
    print(f"Done. Status counts: {counts}")


if __name__ == "__main__":
    main()
