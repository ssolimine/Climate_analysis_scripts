import os
import subprocess
from datetime import datetime, timedelta
from calendar import monthrange

"""
Download GOES-16 data using Homebrew for a given year range and month list.
This script uses the AWS CLI S3 sync command to copy data from the public NOAA GOES-16 bucket.

- start_year, end_year: inclusive
- months: list of ints (1..12), e.g. [2,3,4,5]
- product: GOES-16 product, e.g. 'ABI-L1b-RadC'
- hours: iterable of hours (0..23)
- dry_run: if True, just print commands without running
"""

def iter_days_in_month(year: int, month: int):
    last = monthrange(year, month)[1]
    d = datetime(year, month, 1)
    end = datetime(year, month, last) + timedelta(days=1)
    while d < end:
        yield d
        d += timedelta(days=1)

def download_goes16_range(
    start_year: int,
    end_year: int,
    months: list[int],
    product: str = "ABI-L1b-RadC",
    local_base: str = "/local/path/to/GOES16",  # Local folder to store downloads
    aws_path: str = "/opt/homebrew/bin/aws",  # Typical Homebrew path on Apple Silicon Macs (M1/M2/M3) - change as needed
    hours = range(24),
    dry_run: bool = False,
):

    for year in range(start_year, end_year + 1):
        for month in months:
            for day_dt in iter_days_in_month(year, month):
                day_of_year = day_dt.timetuple().tm_yday

                for hour in hours:
                    s3_path = f"s3://noaa-goes16/{product}/{year:04d}/{day_of_year:03d}/{hour:02d}/"
                    local_path = os.path.join(
                        local_base, product, f"{year:04d}", f"{day_of_year:03d}", f"{hour:02d}"
                    )
                    os.makedirs(local_path, exist_ok=True)

                    cmd = [aws_path, "s3", "sync", s3_path, local_path, "--no-sign-request"]
                    print(f"Syncing {s3_path} -> {local_path}")
                    if dry_run:
                        print("DRY RUN:", " ".join(cmd + ["--dryrun"]))
                        subprocess.run(cmd + ["--dryrun"], check=False)
                    else:
                        subprocess.run(cmd, check=False)  # set check=True if you want to abort on error

if __name__ == "__main__":
    # ======= Configuration - change as desired =======
    START_YEAR = 2022
    END_YEAR   = 2022
    MONTHS     = [2, 3, 4, 5]     # Feb–May
    PRODUCT    = "ABI-L1b-RadC"
    LOCAL_BASE = "/local/path/to/GOES16"  # Local folder to store downloads
    AWS_CLI    = "/opt/homebrew/bin/aws"  # Typical Homebrew path on Apple Silicon Macs (M1/M2/M3) - change as needed
    HOURS      = range(24)
    DRY_RUN    = False            # True to print commands only
    # ==========================================

    download_goes16_range(
        start_year=START_YEAR,
        end_year=END_YEAR,
        months=MONTHS,
        product=PRODUCT,
        local_base=LOCAL_BASE,
        aws_path=AWS_CLI,
        hours=HOURS,
        dry_run=DRY_RUN,
    )
