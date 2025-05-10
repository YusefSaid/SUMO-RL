import os
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

def analyze_vehicle_trajectories(tripinfo_file="tripinfo.xml"):
    """Analyze vehicle trajectories to infer traffic light behavior"""
    print(f"Analyzing vehicle movement patterns from {tripinfo_file}")
    
    if not os.path.exists(tripinfo_file):
        print(f"Error: {tripinfo_file} not found")
        return
    
    try:
        # Parse the XML file
        tree = ET.parse(tripinfo_file)
        root = tree.getroot()
        
        # Extract trip information
        trips = root.findall("tripinfo")
        
        if not trips:
            print("No trip information found in the file")
            return
        
        print(f"Found {len(trips)} completed vehicle trips")
        
        # Extract relevant data
        trip_data = []
        for trip in trips:
            trip_id = trip.get("id")
            depart = float(trip.get("depart", 0))
            arrival = float(trip.get("arrival", 0))
            duration = float(trip.get("duration", 0))
            waiting_time = float(trip.get("waitingTime", 0))
            time_loss = float(trip.get("timeLoss", 0))
            
            trip_data.append({
                "id": trip_id,
                "depart": depart,
                "arrival": arrival,
                "duration": duration,
                "waiting_time": waiting_time,
                "time_loss": time_loss
            })
        
        # Sort by departure time
        trip_data.sort(key=lambda x: x["depart"])
        
        # Analyze waiting patterns over time
        departure_times = [trip["depart"] for trip in trip_data]
        waiting_times = [trip["waiting_time"] for trip in trip_data]
        
        # Create visualizations
        plt.figure(figsize=(12, 8))
        
        # Plot vehicle waiting times by departure time
        plt.subplot(2, 1, 1)
        plt.scatter(departure_times, waiting_times, alpha=0.7, c='blue')
        plt.xlabel('Departure Time (s)')
        plt.ylabel('Waiting Time (s)')
        plt.title('Vehicle Waiting Times Throughout Simulation')
        plt.grid(True, alpha=0.3)
        
        # Identify potential traffic light patterns through time series analysis
        # Group waiting times into time windows
        window_size = 10  # seconds
        max_time = max(departure_times) + window_size
        time_windows = np.arange(0, max_time, window_size)
        avg_waiting_per_window = []
        
        for i in range(len(time_windows)-1):
            window_start = time_windows[i]
            window_end = time_windows[i+1]
            
            # Get waiting times for vehicles that departed in this window
            window_waiting_times = [
                trip["waiting_time"] for trip in trip_data 
                if window_start <= trip["depart"] < window_end
            ]
            
            if window_waiting_times:
                avg_waiting = np.mean(window_waiting_times)
            else:
                avg_waiting = 0
                
            avg_waiting_per_window.append(avg_waiting)
        
        # Plot average waiting time per time window
        plt.subplot(2, 1, 2)
        plt.plot(time_windows[:-1], avg_waiting_per_window, 'r-', linewidth=2)
        plt.xlabel('Simulation Time (s)')
        plt.ylabel('Average Waiting Time (s)')
        plt.title('Average Waiting Time in Time Windows')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('vehicle_waiting_patterns.png')
        plt.close()
        
        print("Vehicle waiting pattern analysis saved to vehicle_waiting_patterns.png")
        
        # Identify potential traffic signal coordination patterns
        # Count vehicles in different waiting time ranges
        waiting_ranges = [0, 5, 10, 20, 30, 60, float('inf')]
        waiting_counts = [0] * (len(waiting_ranges) - 1)
        
        for trip in trip_data:
            wait = trip["waiting_time"]
            for i in range(len(waiting_ranges) - 1):
                if waiting_ranges[i] <= wait < waiting_ranges[i+1]:
                    waiting_counts[i] += 1
                    break
        
        # Plot waiting time distribution
        plt.figure(figsize=(10, 6))
        
        range_labels = []
        for i in range(len(waiting_ranges) - 1):
            if waiting_ranges[i+1] == float('inf'):
                range_labels.append(f">{waiting_ranges[i]}s")
            else:
                range_labels.append(f"{waiting_ranges[i]}-{waiting_ranges[i+1]}s")
        
        plt.bar(range_labels, waiting_counts, color='green', alpha=0.7)
        plt.xlabel('Waiting Time Range')
        plt.ylabel('Number of Vehicles')
        plt.title('Distribution of Vehicle Waiting Times')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('waiting_time_distribution.png')
        plt.close()
        
        print("Waiting time distribution saved to waiting_time_distribution.png")
        
        # Look for periodic patterns in the waiting times
        if len(avg_waiting_per_window) > 1:
            # Simple peak detection
            peaks = []
            for i in range(1, len(avg_waiting_per_window)-1):
                if (avg_waiting_per_window[i] > avg_waiting_per_window[i-1] and 
                    avg_waiting_per_window[i] > avg_waiting_per_window[i+1] and
                    avg_waiting_per_window[i] > 5):  # Minimum peak height
                    peaks.append(time_windows[i])
            
            # Calculate time between peaks
            if len(peaks) >= 2:
                peak_intervals = [peaks[i+1] - peaks[i] for i in range(len(peaks)-1)]
                
                if peak_intervals:
                    avg_interval = np.mean(peak_intervals)
                    print(f"\nPotential traffic light cycle time: {avg_interval:.2f} seconds")
                    print(f"Found {len(peaks)} distinct waiting time peaks")
                    
                    # Plot detected peaks
                    plt.figure(figsize=(10, 6))
                    plt.plot(time_windows[:-1], avg_waiting_per_window, 'b-', linewidth=2)
                    plt.plot(peaks, [avg_waiting_per_window[int(p/window_size)] for p in peaks], 'ro', markersize=8)
                    plt.xlabel('Simulation Time (s)')
                    plt.ylabel('Average Waiting Time (s)')
                    plt.title('Detected Traffic Light Cycles')
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig('traffic_light_cycles.png')
                    plt.close()
                    
                    print("Traffic light cycle analysis saved to traffic_light_cycles.png")
            else:
                print("\nNo clear traffic light cycle pattern detected")
        
        # Calculate summary statistics
        print("\nSummary Statistics:")
        print(f"Total vehicles: {len(trip_data)}")
        print(f"Average waiting time: {np.mean(waiting_times):.2f} seconds")
        print(f"Maximum waiting time: {np.max(waiting_times):.2f} seconds")
        
        # Calculate how many vehicles waited at all
        vehicles_with_wait = sum(1 for t in trip_data if t["waiting_time"] > 0)
        wait_percentage = (vehicles_with_wait / len(trip_data)) * 100
        print(f"Vehicles that had to wait: {vehicles_with_wait} ({wait_percentage:.2f}%)")
        
        # Infer traffic control efficiency
        efficiency = 100 - (np.mean(waiting_times) / np.mean([t["duration"] for t in trip_data])) * 100
        print(f"Traffic control efficiency estimate: {efficiency:.2f}%")
        
    except Exception as e:
        print(f"Error analyzing vehicle trajectories: {e}")
        import traceback
        traceback.print_exc()

# Main execution
if __name__ == "__main__":
    analyze_vehicle_trajectories()