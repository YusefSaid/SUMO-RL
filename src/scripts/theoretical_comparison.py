#!/usr/bin/env python3
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

# Configure logging for info and warnings
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Attempt to import pandas for nicer table output; catch any exception
HAS_PANDAS = False
try:
    import pandas as pd
    HAS_PANDAS = True
except Exception as e:
    logging.warning(f"Pandas import failed ({e.__class__.__name__}: {e}), using plain-text summary instead.")


def windowed_average(xs, ys, window_size):
    """
    Compute average of `ys` over sliding windows in `xs`.
    Returns bins and corresponding average values.
    """
    bins = np.arange(0, max(xs) + window_size, window_size)
    avgs = []
    for start, end in zip(bins[:-1], bins[1:]):
        chunk = [y for x, y in zip(xs, ys) if start <= x < end]
        avgs.append(np.mean(chunk) if chunk else 0.0)
    return bins, avgs


def analyze_vehicle_trajectories(tripinfo_file="tripinfo.xml"):
    """Parse SUMO tripinfo, generate analysis plots, and print summary."""
    if not os.path.exists(tripinfo_file):
        logging.error(f"File not found: {tripinfo_file}")
        return

    try:
        logging.info(f"Analyzing data from: {tripinfo_file}")
        tree = ET.parse(tripinfo_file)
        root = tree.getroot()
        trips = root.findall("tripinfo")

        if not trips:
            logging.error("No <tripinfo> elements found.")
            return

        logging.info(f"Found {len(trips)} vehicle trips.")

        # Extract trip metrics
        trip_data = []
        for trip in trips:
            trip_data.append({
                "id": trip.get("id"),
                "depart": float(trip.get("depart", 0)),
                "arrival": float(trip.get("arrival", 0)),
                "duration": float(trip.get("duration", 0)),
                "waiting_time": float(trip.get("waitingTime", 0)),
                "time_loss": float(trip.get("timeLoss", 0))
            })

        # Sort and separate arrays
        trip_data.sort(key=lambda t: t["depart"])
        departures = [t["depart"] for t in trip_data]
        waits      = [t["waiting_time"] for t in trip_data]

        # 1) Scatter + windowed average plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        ax1.scatter(departures, waits, alpha=0.7)
        ax1.set(xlabel="Depart (s)", ylabel="Wait (s)", title="Waiting Times by Departure")
        ax1.grid(True, alpha=0.3)

        window = 10  # seconds
        bins, avg_waits = windowed_average(departures, waits, window)
        ax2.plot(bins[:-1], avg_waits, linewidth=2)
        ax2.set(xlabel="Time (s)", ylabel="Avg Wait (s)", title="Average Waiting per Window")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("vehicle_waiting_patterns.png")
        plt.close(fig)
        logging.info("Saved: vehicle_waiting_patterns.png")

        # 2) Waiting time distribution histogram
        ranges = [0, 5, 10, 20, 30, 60, np.inf]
        counts = [0] * (len(ranges) - 1)
        for t in trip_data:
            w = t["waiting_time"]
            for i in range(len(ranges)-1):
                if ranges[i] <= w < ranges[i+1]:
                    counts[i] += 1
                    break

        labels = [
            f">{ranges[i]}s" if ranges[i+1] == np.inf else f"{ranges[i]}-{ranges[i+1]}s"
            for i in range(len(ranges)-1)
        ]
        fig = plt.figure(figsize=(10,6))
        plt.bar(labels, counts)
        plt.xlabel("Wait Range")
        plt.ylabel("Vehicles")
        plt.title("Waiting Time Distribution")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("waiting_time_distribution.png")
        plt.close(fig)
        logging.info("Saved: waiting_time_distribution.png")

        # 3) Peak detection for cycle inference
        if len(avg_waits) > 1:
            peaks = [bins[i] for i in range(1, len(avg_waits)-1)
                     if avg_waits[i] > avg_waits[i-1] and avg_waits[i] > avg_waits[i+1] and avg_waits[i] > 5]

            if len(peaks) >= 2:
                intervals = np.diff(peaks)
                cycle = np.mean(intervals)
                logging.info(f"Cycle time ~ {cycle:.1f}s over {len(peaks)} peaks")

                fig = plt.figure(figsize=(10,6))
                plt.plot(bins[:-1], avg_waits, linewidth=2)
                y_peaks = [avg_waits[int(p/window)] for p in peaks]
                plt.plot(peaks, y_peaks, marker='o', linestyle='')
                plt.xlabel("Time (s)")
                plt.ylabel("Avg Wait (s)")
                plt.title("Detected Signal Cycles")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig("traffic_light_cycles.png")
                plt.close(fig)
                logging.info("Saved: traffic_light_cycles.png")
            else:
                logging.info("No clear cycle peaks detected.")

        # 4) Summary stats
        total = len(trip_data)
        avg_wait = np.mean(waits)
        max_wait = np.max(waits)
        waited = sum(1 for w in waits if w>0)
        pct    = waited/total * 100
        efficiency = 100 - (avg_wait / np.mean([t["duration"] for t in trip_data])) * 100

        logging.info("\nSummary Statistics:")
        if HAS_PANDAS:
            df = pd.DataFrame({
                "Total": [total],
                "AvgWait(s)": [avg_wait],
                "MaxWait(s)": [max_wait],
                "%Waited": [pct],
                "Efficiency": [efficiency]
            })
            print(df.to_string(index=False))
        else:
            print(f"Total trips:       {total}")
            print(f"Avg wait (s):      {avg_wait:.2f}")
            print(f"Max wait (s):      {max_wait:.2f}")
            print(f"% vehicles waited: {pct:.1f}%")
            print(f"Efficiency:        {efficiency:.1f}%")

    except Exception:
        logging.exception("Error during trajectory analysis.")


if __name__ == "__main__":
    analyze_vehicle_trajectories()
