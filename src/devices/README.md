# Devices API

Per-user device-timeline accessor. Returns the iPhone and Apple Watch a
participant was using at any given timestamp.

## Public surface

- `devices.get_devices(health_code, timestamp) -> DeviceSnapshot` — point-in-time lookup
- `devices.get_device_timeline(health_code) -> DeviceTimeline` — full interval history
- `devices.DeviceInterval` — one `[start, end, name]` row
- `devices.DeviceTimeline` — sorted, non-overlapping intervals for phone and watch
- `devices.DeviceSnapshot` — the resolved phone and watch at one timestamp

## Usage

```python
import pandas as pd
from devices import get_devices, get_device_timeline

snap = get_devices("user-123", pd.Timestamp("2020-06-01"))
if snap.phone is not None:
    print(snap.phone.name)       # 'iPhone 8 Plus'
if snap.watch is not None:
    print(snap.watch.name)       # 'Apple Watch Series 5 44mm GPS'

timeline = get_device_timeline("user-123")
for iv in timeline.phone:
    print(iv.start.date(), iv.end.date(), iv.name)
```

Either field may be `None` if no interval covers the query timestamp
(before the user's first record, or in a gap between devices). Query
timestamps must be tz-naive.

> Set `USER_DEVICE_INFO_PATH` *before* importing `devices` — the
> module-level `STORE` captures the path at import time.

## Data source

Reads `data/labels/user_device_info.json`. The path is overridable via the
`USER_DEVICE_INFO_PATH` environment variable.

The data file ships separately from `last_labels.json` and is not
guaranteed to be in every Dataverse bundle. Importing `devices` works
either way; the first lookup raises `FileNotFoundError` with guidance if
the file is missing.

## Schema

```json
{
  "<healthCode>": {
    "phone": [["2017-12-18", "2023-05-18", "iPhone 8 Plus"], ...],
    "watch": [["2019-04-05", "2023-12-19", "Apple Watch Series 5 44mm GPS"], ...]
  }
}
```

Intervals are inclusive on both ends, contiguous-run (calendar gaps inside
a same-model run are absorbed upstream), non-overlapping, and sorted by
start date. Unrecognised device strings fall through to
`iPhone Unknown Model` / `Apple Watch Unknown Model`; roughly 85% of the
Unknown bucket reflects user-renamed Apple devices in iOS Settings rather
than third-party hardware.

## Running tests

```bash
pytest src/devices/test_api.py
```
