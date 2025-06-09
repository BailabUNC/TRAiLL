import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def detect_action_phases(channel_data, channel_name):
    """
    Detects start and end points of trigger (rise) and reset (drop) phases
    for a single channel's data using its second derivative.

    Args:
        channel_data (pd.Series): Time series data for the channel.
        channel_name (str): Name of the channel (for context in messages).

    Returns:
        dict: A dictionary with keys "trigger_start", "trigger_end", 
              "reset_start", "reset_end", containing the indices (or timestamps)
              of these points. Values can be NaN if a point is not detected.
    """
    if channel_data.empty or len(channel_data) < 3: # Need at least 3 points for 2nd derivative
        print(f"Channel {channel_name} data is too short to process.")
        return {"trigger_start": np.nan, "trigger_end": np.nan,
                "reset_start": np.nan, "reset_end": np.nan}

    first_derivative = np.gradient(channel_data.to_numpy())
    second_derivative = np.gradient(first_derivative)
    s_d2 = pd.Series(second_derivative, index=channel_data.index)

    std_d2 = s_d2.std()
    if std_d2 == 0 or pd.isna(std_d2): # Flat or problematic second derivative
        print(f"Second derivative for channel {channel_name} is flat or invalid. Cannot detect points.")
        return {"trigger_start": np.nan, "trigger_end": np.nan,
                "reset_start": np.nan, "reset_end": np.nan}

    # Thresholds - these might need tuning
    positive_thresh = std_d2 * 1.0 
    negative_thresh = -std_d2 * 1.0
    near_zero_thresh_pos = std_d2 * 0.2
    near_zero_thresh_neg = -std_d2 * 0.2
    
    key_points = {
        "trigger_start": np.nan, "trigger_end": np.nan,
        "reset_start": np.nan, "reset_end": np.nan
    }

    # --- Trigger (Rise) Detection ---
    try:
        potential_trigger_starts = s_d2[s_d2 > positive_thresh].index
        if not potential_trigger_starts.empty:
            trigger_start_idx = potential_trigger_starts[0]
            key_points["trigger_start"] = trigger_start_idx

            s_d2_after_trigger_start = s_d2[s_d2.index >= trigger_start_idx]
            current_peak_s_d2 = -np.inf
            peak_s_d2_idx = np.nan
            
            for idx_val in s_d2_after_trigger_start.index: # Renamed idx to idx_val to avoid conflict
                if s_d2[idx_val] > current_peak_s_d2:
                    current_peak_s_d2 = s_d2[idx_val]
                    peak_s_d2_idx = idx_val
                if pd.notna(peak_s_d2_idx) and s_d2[idx_val] < near_zero_thresh_pos and idx_val > peak_s_d2_idx :
                    key_points["trigger_end"] = idx_val
                    break
            
            if pd.notna(peak_s_d2_idx) and pd.isna(key_points["trigger_end"]):
                # If it never dropped sufficiently, check if it started falling after peak
                falling_after_peak = s_d2_after_trigger_start[s_d2_after_trigger_start.index > peak_s_d2_idx]
                if not falling_after_peak.empty:
                    potential_ends = falling_after_peak[falling_after_peak < near_zero_thresh_pos].index
                    if not potential_ends.empty:
                        key_points["trigger_end"] = potential_ends[0]
                    else: # Stays high or doesn't fall enough to near_zero_thresh_pos
                        key_points["trigger_end"] = s_d2.index[-1] 
                else: # No data points after peak or stays flat
                     key_points["trigger_end"] = s_d2.index[-1]

    except Exception as e:
        print(f"Error during trigger (rise) detection for {channel_name}: {e}")

    # --- Reset (Drop) Detection ---
    search_from_idx_for_drop = key_points["trigger_end"]
    if pd.isna(search_from_idx_for_drop):
        search_from_idx_for_drop = key_points["trigger_start"]
    if pd.isna(search_from_idx_for_drop) and not s_d2.empty:
        search_from_idx_for_drop = s_d2.index[0]
    elif s_d2.empty:
        print(f"Cannot perform reset (drop) detection for {channel_name} as s_d2 is empty.")
        return key_points # Return already found (or NaN) trigger points


    try:
        if pd.notna(search_from_idx_for_drop):
            s_d2_for_drop_search = s_d2[s_d2.index >= search_from_idx_for_drop]
            if not s_d2_for_drop_search.empty:
                potential_reset_starts = s_d2_for_drop_search[s_d2_for_drop_search < negative_thresh].index
                if not potential_reset_starts.empty:
                    reset_start_idx = potential_reset_starts[0]
                    key_points["reset_start"] = reset_start_idx

                    s_d2_after_reset_start = s_d2[s_d2.index >= reset_start_idx]
                    current_trough_s_d2 = np.inf
                    trough_s_d2_idx = np.nan

                    for idx_val in s_d2_after_reset_start.index: # Renamed idx to idx_val
                        if s_d2[idx_val] < current_trough_s_d2:
                            current_trough_s_d2 = s_d2[idx_val]
                            trough_s_d2_idx = idx_val
                        if pd.notna(trough_s_d2_idx) and s_d2[idx_val] > near_zero_thresh_neg and idx_val > trough_s_d2_idx:
                            key_points["reset_end"] = idx_val
                            break
                    
                    if pd.notna(trough_s_d2_idx) and pd.isna(key_points["reset_end"]):
                        rising_after_trough = s_d2_after_reset_start[s_d2_after_reset_start.index > trough_s_d2_idx]
                        if not rising_after_trough.empty:
                            potential_ends = rising_after_trough[rising_after_trough > near_zero_thresh_neg].index
                            if not potential_ends.empty:
                                key_points["reset_end"] = potential_ends[0]
                            else: # Stays low or doesn't rise enough
                                key_points["reset_end"] = s_d2.index[-1]
                        else: # No data points after trough or stays flat
                            key_points["reset_end"] = s_d2.index[-1]
    except Exception as e:
        print(f"Error during reset (drop) detection for {channel_name}: {e}")
        
    return key_points


def process_csv_for_action_key_points(csv_path, instance_column_name=None):
    """
    Reads a CSV file, identifies instances, finds the most significant channel for each,
    and determines key points (trigger_start, trigger_end, reset_start, reset_end)
    based on the second derivative of that channel.

    Args:
        csv_path (str): Path to the CSV file.
        instance_column_name (str, optional): Name of the column that identifies instances.
                                             If None, the entire CSV is treated as one instance.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return []
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return []

    if df.empty:
        print("CSV file is empty.")
        return []

    instances = []
    if instance_column_name and instance_column_name in df.columns:
        for instance_id, group in df.groupby(instance_column_name):
            instances.append((instance_id, group.copy()))
    elif instance_column_name:
        print(f"Warning: Instance column '{instance_column_name}' not found. Treating entire CSV as one instance.")
        instances.append(("full_dataset", df.copy()))
    else:
        instances.append(("full_dataset", df.copy()))

    results = []

    for instance_id, instance_df in instances:
        print(f"\nProcessing instance: {instance_id}")

        msc_column_name_specifier = 'most_significant_channel_idx'  # Column containing the name of the significant channel

        if msc_column_name_specifier not in instance_df.columns:
            error_msg = f"Column '{msc_column_name_specifier}' not found"
            print(f"Error for instance {instance_id}: {error_msg}. Skipping.")
            results.append({
                "instance_id": instance_id,
                "most_significant_channel": f"N/A - {error_msg}",
                "trigger_start": np.nan, "trigger_end": np.nan,
                "reset_start": np.nan, "reset_end": np.nan
            })
            continue

        try:
            # Assuming the channel name is consistent for all rows of an instance, take the first.
            most_significant_channel_name = str(instance_df[msc_column_name_specifier].iloc[0])
            # this only gives the channel number, add a prefix to get the actual channel name
            prefix = 'original_channel_'
            most_significant_channel_name = prefix + most_significant_channel_name
            print(f"Most significant channel for instance {instance_id}: {most_significant_channel_name}")
            if pd.isna(most_significant_channel_name):
                error_msg = f"Value in '{msc_column_name_specifier}' is NaN"
                print(f"Error for instance {instance_id}: {error_msg}. Skipping.")
                results.append({
                    "instance_id": instance_id,
                    "most_significant_channel": f"N/A - {error_msg}",
                    "trigger_start": np.nan, "trigger_end": np.nan,
                    "reset_start": np.nan, "reset_end": np.nan
                })
                continue
            if not isinstance(most_significant_channel_name, str):
                most_significant_channel_name = str(most_significant_channel_name) # Ensure it's a string
        except IndexError:
            error_msg = f"Instance is empty, cannot get value from '{msc_column_name_specifier}'"
            print(f"Error for instance {instance_id}: {error_msg}. Skipping.")
            results.append({
                "instance_id": instance_id,
                "most_significant_channel": f"N/A - {error_msg}",
                "trigger_start": np.nan, "trigger_end": np.nan,
                "reset_start": np.nan, "reset_end": np.nan
            })
            continue
        except Exception as e:
            error_msg = f"Error accessing '{msc_column_name_specifier}': {e}"
            print(f"Error for instance {instance_id}: {error_msg}. Skipping.")
            results.append({
                "instance_id": instance_id,
                "most_significant_channel": f"N/A - {error_msg}",
                "trigger_start": np.nan, "trigger_end": np.nan,
                "reset_start": np.nan, "reset_end": np.nan
            })
            continue

        if most_significant_channel_name not in instance_df.columns:
            error_msg = f"Channel '{most_significant_channel_name}' (from '{msc_column_name_specifier}') not found as a data column"
            print(f"Error for instance {instance_id}: {error_msg}. Skipping.")
            results.append({
                "instance_id": instance_id,
                "most_significant_channel": f"N/A - {error_msg}",
                "trigger_start": np.nan, "trigger_end": np.nan,
                "reset_start": np.nan, "reset_end": np.nan
            })
            continue
        
        msc_data = instance_df[most_significant_channel_name].dropna()
        
        print(f"Using specified most significant channel for instance {instance_id}: {most_significant_channel_name} (Length: {len(msc_data)})")

        if len(msc_data) < 3 : # Check after potential dropna
             print(f"Not enough data points in channel {most_significant_channel_name} after NaN removal for instance {instance_id}.")
             key_points = {"trigger_start": np.nan, "trigger_end": np.nan, "reset_start": np.nan, "reset_end": np.nan}
        else:
            key_points = detect_action_phases(msc_data, most_significant_channel_name)
        
        print(f"Key points for instance {instance_id} (channel: {most_significant_channel_name}):")
        for point_name, point_value in key_points.items():
            print(f"  {point_name}: {point_value}")
        
        results.append({
            "instance_id": instance_id,
            "most_significant_channel": most_significant_channel_name,
            **key_points
        })
        
    return results


if __name__ == '__main__':
    csv_path = './figures/fig2/concatenated_finger_sequence/data/concatenated_all_channels_sequences_10.csv'

    # Name of the column that identifies different instances/sequences in your CSV.
    # If your CSV doesn't have such a column and all data belongs to one sequence,
    # or if sequences are just concatenated without an ID, set this to None.
    # You might need to adapt instance handling if sequences are implicitly defined.
    instance_id_column = 'instance_index' # Or, e.g., 'sequence_id', 'trial_id'

    print(f"Starting CSV processing for: {csv_path}")
    all_found_key_points = process_csv_for_action_key_points(
        csv_path=csv_path,
        instance_column_name=instance_id_column
    )

    if all_found_key_points:
        print("\\n--- Summary of All Key Points ---")
        for result_item in all_found_key_points:
            print(result_item)
    else:
        print("\\nNo key points were processed or found.")

plt.show()