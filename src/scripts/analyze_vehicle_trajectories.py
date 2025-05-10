import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")

def windowed_average(xs, ys, window_size):
    """
    Compute the average of `ys` over sliding windows in `xs`.
    Returns the bin edges and the corresponding averages.
    """
    bins = np.arange(0, max(xs) + window_size, window_size)
    avgs = []
    for start, end in zip(bins[:-1], bins[1:]):
        chunk = [y for x, y in zip(xs, ys) if start <= x < end]
        avgs.append(np.mean(chunk) if chunk else 0.0)
    return bins, avgs


def analyze_vehicle_trajectories(tripinfo_file="tripinfo.xml"):
    """Analyze vehicle trajectories to infer traffic light behavior and save plots."""
    if not os.path.exists(tripinfo_file):
        logging.error(f"{tripinfo_file} not found")
        return

    try:
        logging.info(f"Analyzing vehicle movement patterns from {tripinfo_file}")
        tree = ET.parse(tripinfo_file)
        root = tree.getroot()
        trips = root.findall("tripinfo")

        if not trips:
            logging.error("No trip information found in the file")
            return

        logging.info(f"Found {len(trips)} completed vehicle trips")

        # Build list of trip dictionaries
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

        # Sort by departure time
        trip_data.sort(key=lambda t: t["depart"])
        departure_times = [t["depart"] for t in trip_data]
        waiting_times = [t["waiting_time"] for t in trip_data]

        # 1) Scatter + windowed-average plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        ax1.scatter(departure_times, waiting_times, alpha=0.7)
        ax1.set(
            xlabel="Departure Time (s)",
            ylabel="Waiting Time (s)",
            title="Vehicle Waiting Times Throughout Simulation"
        )
        ax1.grid(True, alpha=0.3)

        window_size = 10  # seconds
        bins, avg_wait = windowed_average(departure_times, waiting_times, window_size)
        ax2.plot(bins[:-1], avg_wait, linewidth=2)
        ax2.set(
            xlabel="Simulation Time (s)",
            ylabel="Average Waiting Time (s)",
            title="Average Waiting Time in Time Windows"
        )
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("vehicle_waiting_patterns.png")
        plt.close(fig)
        logging.info("Saved: vehicle_waiting_patterns.png")

        # 2) Waiting-time distribution histogram
        waiting_ranges = [0, 5, 10, 20, 30, 60, np.inf]
        waiting_counts = [0] * (len(waiting_ranges) - 1)
        for t in trip_data:
            w = t["waiting_time"]
            for i in range(len(waiting_ranges) - 1):
                if waiting_ranges[i] <= w < waiting_ranges[i + 1]:
                    waiting_counts[i] += 1
                    break

        labels = [
            f">{waiting_ranges[i]}s" if waiting_ranges[i+1] == np.inf
            else f"{waiting_ranges[i]}-{waiting_ranges[i+1]}s"
            for i in range(len(waiting_ranges) - 1)
        ]
        fig = plt.figure(figsize=(10, 6))
        plt.bar(labels, waiting_counts)
        plt.xlabel("Waiting Time Range")
        plt.ylabel("Number of Vehicles")
        plt.title("Distribution of Vehicle Waiting Times")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("waiting_time_distribution.png")
        plt.close(fig)
        logging.info("Saved: waiting_time_distribution.png")

        # 3) Peak detection for signal cycle inference
        if len(avg_wait) > 1:
            peaks = []
            for i in range(1, len(avg_wait) - 1):
                if (
                    avg_wait[i] > avg_wait[i-1] and 
                    avg_wait[i] > avg_wait[i+1] and 
                    avg_wait[i] > 5  # min peak height
                ):
                    peaks.append(bins[i])

            if len(peaks) >= 2:
                intervals = np.diff(peaks)
                avg_interval = np.mean(intervals)
                logging.info(f"Potential cycle time: {avg_interval:.2f} s")
                logging.info(f"Detected {len(peaks)} peaks")

                fig = plt.figure(figsize=(10, 6))
                plt.plot(bins[:-1], avg_wait, linewidth=2)
                y_peaks = [avg_wait[int(p/window_size)] for p in peaks]
                plt.plot(peaks, y_peaks, marker='o', linestyle='')
                plt.xlabel("Simulation Time (s)")
                plt.ylabel("Average Waiting Time (s)")
                plt.title("Detected Traffic Light Cycles")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig("traffic_light_cycles.png")
                plt.close(fig)
                logging.info("Saved: traffic_light_cycles.png")
            else:
                logging.info("No clear cycle pattern detected")

        # 4) Summary statistics table
        total = len(trip_data)
        avg_wait_all = np.mean(waiting_times)
        max_wait_all = np.max(waiting_times)
        waited_count = sum(1 for t in trip_data if t["waiting_time"] > 0)
        pct_waited = waited_count / total * 100
        efficiency = 100 - (avg_wait_all / np.mean([t["duration"] for t in trip_data])) * 100

        logging.info("\nSummary Statistics:")
        summary = pd.DataFrame({
            "Total vehicles": [total],
            "Avg waiting (s)": [avg_wait_all],
            "Max waiting (s)": [max_wait_all],
            "Pct waited (%)": [pct_waited],
            "Efficiency (%)": [efficiency]
        })
        print(summary.to_string(index=False))

    except Exception:
        logging.exception("Error analyzing vehicle trajectories")


if __name__ == "__main__":
    analyze_vehicle_trajectories()
