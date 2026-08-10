# analyze_timeline.py

import numpy as np


filename = "task_timeline.txt"


tasks = {}

with open(filename) as f:
    for line in f:
        name, event, ts = line.strip().split(",")

        ts = float(ts)

        if name not in tasks:
            tasks[name] = []

        tasks[name].append(
            (event, ts)
        )


def extract_duration(name):

    records = tasks[name]

    starts = []
    ends = []

    for event, ts in records:
        if event == "start":
            starts.append(ts)
        elif event == "end":
            ends.append(ts)

    assert len(starts) == len(ends)

    durations = []

    for s, e in zip(starts, ends):
        durations.append(
            e - s
        )

    return durations



dit_times = extract_duration(
    "DiT"
)

vae_times = extract_duration(
    "VAE"
)


def trim_mean(values):

    values = sorted(values)

    print(
        "raw:",
        [round(x,3) for x in values]
    )

    values = values[1:-1]

    return np.mean(values)



print("===================")

print(
    "DiT count:",
    len(dit_times)
)

print(
    "VAE count:",
    len(vae_times)
)


print("===================")


print(
    "DiT avg (trim max/min):",
    trim_mean(dit_times),
    "ms"
)


print(
    "VAE avg (trim max/min):",
    trim_mean(vae_times),
    "ms"
)