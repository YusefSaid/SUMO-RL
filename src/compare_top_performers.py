import os
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import glob  # For finding all tripinfo files

# Find all tripinfo files in the current directory
def find_tripinfo_files():
    # Look for algorithm-specific tripinfo files
    algorithm_files = glob.glob("*_*_tripinfo.xml")
    
    # Also include the default tripinfo.xml if it exists
    if os.path.exists("tripinfo.xml"):
        if "tripinfo.xml" not in algorithm_files:
            algorithm_files.append("tripinfo.xml")
    
    return algorithm_files

# Function to analyze a single tripinfo file
def analyze_tripinfo(file_path, label=None):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found")
        return None
    
    # If no label was provided, extract it from the filename
    if label is None:
        if file_path == "tripinfo.xml":
            label = "RL Model"
        else:
            # Extract label from filename (e.g., a2c_simple_tripinfo.xml -> A2C SIMPLE)
            filename = os.path.basename(file_path)
            label_part = filename.replace("_tripinfo.xml", "").replace("_", " ").upper()
            label = label_part
    
    print(f"\nAnalyzing {label} from {file_path}")
    
    try:
        # Parse the XML file
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Extract trip information
        trips = root.findall("tripinfo")
        
        if not trips:
            print("No trip information found in the file")
            return None
        
        print(f"Found {len(trips)} completed vehicle trips")
        
        # Collect metrics
        durations = []
        waiting_times = []
        time_loss = []
        
        for trip in trips:
            # Extract attributes
            duration = float(trip.get("duration", 0))
            waiting = float(trip.get("waitingTime", 0))
            loss = float(trip.get("timeLoss", 0))
            
            durations.append(duration)
            waiting_times.append(waiting)
            time_loss.append(loss)
        
        # Calculate statistics
        results = {
            "label": label,
            "total_vehicles": len(trips),
            "avg_travel_time": np.mean(durations),
            "avg_waiting_time": np.mean(waiting_times),
            "avg_time_loss": np.mean(time_loss),
            "max_travel_time": np.max(durations),
            "max_waiting_time": np.max(waiting_times),
            "time_loss_percent": 100 * np.mean(time_loss) / np.mean(durations),
            "waiting_distribution": {
                "0-30": sum(1 for w in waiting_times if 0 <= w < 30),
                "30-60": sum(1 for w in waiting_times if 30 <= w < 60),
                "60-120": sum(1 for w in waiting_times if 60 <= w < 120),
                ">120": sum(1 for w in waiting_times if w >= 120)
            }
        }
        
        # Print results
        print(f"Traffic Performance for {label}:")
        print(f"Total vehicles: {results['total_vehicles']}")
        print(f"Average travel time: {results['avg_travel_time']:.2f} seconds")
        print(f"Average waiting time: {results['avg_waiting_time']:.2f} seconds")
        print(f"Average time loss: {results['avg_time_loss']:.2f} seconds")
        print(f"Maximum travel time: {results['max_travel_time']:.2f} seconds")
        print(f"Maximum waiting time: {results['max_waiting_time']:.2f} seconds")
        print(f"Average time loss percentage: {results['time_loss_percent']:.2f}%")
        
        distribution = results['waiting_distribution']
        total = sum(distribution.values())
        
        print("Waiting Time Distribution:")
        for range_label, count in distribution.items():
            if total > 0:
                percentage = 100 * count / total
                print(f"  {range_label} seconds: {count} vehicles ({percentage:.2f}%)")
        
        return results
    
    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")
        return None

# Analyze all files
def main():
    tripinfo_files = find_tripinfo_files()
    
    if not tripinfo_files:
        print("No tripinfo.xml files found to analyze")
        return
    
    print(f"Found {len(tripinfo_files)} tripinfo files to analyze")
    
    results = {}
    for file_path in tripinfo_files:
        result = analyze_tripinfo(file_path)
        if result:
            results[result['label']] = result
    
    # If we have at least two results, compare them
    if len(results) >= 2:
        print("\n=== Comparison of All Traffic Light Controllers ===")
        print("{:<20} {:<12} {:<12} {:<8} {:<12}".format(
            "Controller", "Avg Wait(s)", "Avg Loss(s)", "Loss %", "% Under 30s"))
        print("-" * 70)
        
        # Sort results by average waiting time (best first)
        sorted_results = sorted(results.values(), key=lambda x: x['avg_waiting_time'])
        
        for result in sorted_results:
            label = result['label']
            wait = result['avg_waiting_time']
            loss = result['avg_time_loss']
            loss_pct = result['time_loss_percent']
            under_30 = result['waiting_distribution']['0-30']
            total = result['total_vehicles']
            pct_under_30 = 100 * under_30 / total if total > 0 else 0
            
            print("{:<20} {:<12.2f} {:<12.2f} {:<8.2f} {:<12.2f}".format(
                label, wait, loss, loss_pct, pct_under_30))
        
        # Create comparison plot
        labels = [r['label'] for r in sorted_results]
        waiting_times = [r['avg_waiting_time'] for r in sorted_results]
        loss_times = [r['avg_time_loss'] for r in sorted_results]
        
        plt.figure(figsize=(12, 6))
        
        x = np.arange(len(labels))
        width = 0.35
        
        plt.bar(x - width/2, waiting_times, width, label='Avg Waiting Time')
        plt.bar(x + width/2, loss_times, width, label='Avg Time Loss')
        
        plt.xlabel('Controller')
        plt.ylabel('Time (seconds)')
        plt.title('Performance Comparison of Traffic Controllers')
        plt.xticks(x, labels, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        
        plt.savefig('waiting_time_comparison.png')
        print("\nComparison graph saved as 'waiting_time_comparison.png'")
    elif len(results) == 1:
        print("\nOnly one traffic controller found. No comparison needed.")
    else:
        print("\nNo valid results found for comparison.")

if __name__ == "__main__":
    main()