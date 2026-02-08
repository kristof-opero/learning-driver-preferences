import json
import os
import pandas as pd


def create_df_of_requests(path_to_parent_folder):
    dirs = os.listdir(path_to_parent_folder)
    if "README.txt" in dirs:
        dirs.pop(dirs.index("README.txt")) # Remove README.txt folder

    rows = []

    for dir in dirs:
        path = os.path.join(path_to_parent_folder, dir)
        if os.path.isdir(path):
            files = os.listdir(path)
            for file in files:
                route_id = file.split('-')[0].split('_')[1]
                date = pd.to_datetime(file.split('-')[1], format="%Y%m%d")
                time = pd.to_datetime(file.split('-')[2], format="%H%M%S").time()
                with open(os.path.join(path, file), "r") as f:                    
                    data = json.load(f)
                    request_type = data["configurationName"]
                    request_id = data["id"]
                    tasks = data["tasks"]                    
                    fixed_tasks = [task["taskId"] for task in data["fixedTasks"]]
                    for task in tasks:
                        row = {}
                        row_id = file.split('.')[-2] + '-' + task["id"]                        
                        row["row_id"] = row_id
                        row["route_id"] = route_id
                        row["date"] = date
                        row["time"] = time
                        row["request_id"] = request_id
                        row["request_type"] = request_type
                        row["task_id"] = task["id"]
                        loc = task["address"]
                        lat = loc["latitude"]
                        lon = loc["longitude"]
                        row["lat"] = lat
                        row["lon"] = lon
                        row["location_id"] = str(int(lat * 10 ** 8)) + str(int(lon * 10 ** 8))
                        row["fixed"] = True if row["task_id"] in fixed_tasks else False
                        row["position_fixed"] = (
                            fixed_tasks.index(row["task_id"]) if row["fixed"] else None
                        )
                        rows.append(row)

    return pd.DataFrame(rows)


def create_df_of_responses(path_to_parent_folder):
    dirs = os.listdir(path_to_parent_folder)

    rows = []

    for dir in dirs:
        path = os.path.join(path_to_parent_folder, dir)
        if os.path.isdir(path):
            files = os.listdir(path)
            for file in files:
                route_id = file.split('-')[0].split('_')[1]
                date = pd.to_datetime(file.split('-')[1], format="%Y%m%d")
                time = pd.to_datetime(file.split('-')[2], format="%H%M%S").time()
                with open(os.path.join(path, file), "r") as f:
                       tasks = f.readlines()
                       for task in tasks:
                        row = {}
                        row_id = file.split('.')[-2] + '-' + task.strip()
                        row["row_id"] = row_id
                        row['route_id'] = route_id
                        row['date'] = date
                        row['time'] = time                        
                        row['task_id'] = task.strip()
                        row['task_sequence_number'] = tasks.index(task) + 1
                        rows.append(row)

    return pd.DataFrame(rows)


def join_requests_and_responses(requests_df, responses_df):
    return requests_df.merge(
        responses_df[["row_id", "task_sequence_number"]],
        on='row_id',
        how="inner"
    )


def count_tasks_per_sequence(df):
    counted_df = (df.groupby(["route_id", "date", "time", "request_type"])["location_id"]
        .nunique()
        .reset_index(name="count")
        )
    counted_df["date"] = counted_df["date"].dt.strftime("%Y-%m-%d")
    return counted_df


def select_compare_start_and_end(df):
    rows = []

    for (r, d), g in df.groupby(["route_id", "date"]):
        g_sorted = g.sort_values(by="time", ascending=True)
        types = g_sorted["request_type"].unique().tolist()
        create_sequence = types[::-1].index("CreateSequence") if "CreateSequence" in types else (len(types) - 1)
        idx = len(types) - create_sequence - 1
        counts = g_sorted["count"].to_list()
        row = {
            "route_id": r,
            "date": d,
            "count_start": counts[idx],
            "count_end": counts[-1],
            "abs_diff_end_start": counts[-1] - counts[idx],
            "pct_diff_end_start": (counts[-1] - counts[idx]) / counts[idx] * 100
        }
        rows.append(row)

    return pd.DataFrame(rows)