
from datetime import datetime, time
from pathlib import Path
import pandas as pd
import numpy as np 
import re 
import json

# Directory stuff to refer to
HERE = Path(__file__).resolve().parent  #this file, index_files.py in dir scripts
BASE_DIR = HERE.parent           #1 parent up from HERE, in dir learning-driver-preferences
DATA_DIR = BASE_DIR / "data"  
REQUESTS_DIR = DATA_DIR / "requests"
RESPONSES_DIR = DATA_DIR / "responses"      
OUTPUT = DATA_DIR / "output"          

OUTPUT.mkdir(parents=True, exist_ok=True)

# Set up timeframes 
morning_window   = (time(5, 0),  time(12, 0))
afternoon_window = (time(12, 1), time(17, 0))
not_relevant_window = (time(17, 1), time(23, 59))

# Set up regex compilations
time_pattern_in_req_filename = re.compile(r"^(?:[^-]*-){2}(\d{6})-")

####################################
# HELPER FUNCTIONS
####################################

# To define if a time is within a timeframe 
def in_timeframe(t: time, timeframe: tuple[time,time]) -> bool:
    return t is not None and timeframe[0] <= t < timeframe[1]

# To classify the timeframe of the request-file (request was made when?)
def classify_request_timeframe(time_in_isoformat: str) -> str:
    if pd.isna(time_in_isoformat): return "unknown"
    time = datetime.strptime(time_in_isoformat, "%H:%M:%S").time()
    if in_timeframe(time, morning_window): return "morning"
    if in_timeframe(time, afternoon_window): return "afternoon"
    if in_timeframe(time, not_relevant_window): return "not-relevant"
    return "other"

# To parse time from the filename 
def parse_time_from_filename(name: str) -> time:
    timestring_from_filename = time_pattern_in_req_filename .match(name)
    if not  timestring_from_filename:
        return None
    hh, mm, ss = timestring_from_filename.group(1)[0:2], timestring_from_filename.group(1)[2:4], timestring_from_filename.group(1)[4:6]
    return time(int(hh), int(mm), int(ss))

# To reqd json-files
def try_read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None

# Convert time to a number of seconds, counted from midnight
def to_sec(t): return t.hour*3600 + t.minute*60 + t.second if t else None

# Try to set task_id in request files to integer, if not possible, set to string
def try_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value) if value is not None else None

# Parse the elements in the request file into a dict 
def parse_request_file(path: Path):
    json_data = try_read_json(path)
    dict_with_json_data = {"configurationName": None, "tasks": [], "fixedTasks": set()}
    if json_data:
        dict_with_json_data["configurationName"] = json_data.get("configurationName")
        for task in json_data.get("tasks", []):
            address = task.get("address", {})
            dict_with_json_data["tasks"].append({
                "id": try_int(task.get("id")),
                "lat": float(address.get("latitude", np.nan)),
                "lon": float(address.get("longitude", np.nan)),
                "from": task.get("timeWindow", {}).get("from"),
                "till": task.get("timeWindow", {}).get("till"),
            })
        dict_with_json_data["fixedTasks"] = {try_int(fixed_task.get("taskId")) for fixed_task in json_data.get("fixedTasks", []) if "taskId" in fixed_task}
        return dict_with_json_data

####################################
# BASE FUNCTION: INDEX FILES
####################################

# Index input request- and response-files into 1 dataframe 
def index_files(requests_dir: Path, responses_dir: Path, depot_prefix: str) -> pd.DataFrame:
    rows = []
    for req_depot_region_datestring_dir in sorted(requests_dir.glob(f"{depot_prefix}*")):
        if not req_depot_region_datestring_dir.is_dir(): continue

        folder_name = req_depot_region_datestring_dir.name  # e.g. 0521_300-20220617
        
        if "-" not in folder_name: continue
        
        depot_region, datestring = folder_name.split("-")
        
        if not depot_region.startswith(depot_prefix): continue
        
        resp_depot_region_datestring_dir = responses_dir / folder_name

        for req_file in sorted(req_depot_region_datestring_dir.glob("*.json")):
            req_time = parse_time_from_filename(req_file.name)   # returns datetime.time or None
            req_time_isoformat = req_time.isoformat() if req_time else None

            parsed_req_file = parse_request_file(req_file)
            configuration_type = parsed_req_file.get("configurationName")
            num_tasks = len(parsed_req_file.get("tasks", []))
            num_fixed = len(parsed_req_file.get("fixedTasks", []))

            # try to match response file by exact filename; else nearest in time
            candidate = None

            exact_path = resp_depot_region_datestring_dir / req_file.name.replace(".json", ".txt")
            if exact_path.exists():
                candidate = exact_path
            else:
                # nearest in time among response files with same prefix
      
                req_time_to_seconds = to_sec(req_time)
                resp_file_closest_to_req_file, smallest_delta = None, None
                for resp_file in sorted(resp_depot_region_datestring_dir.glob(f"{depot_region}-{datestring}-*.txt")):
                    resp_time = parse_time_from_filename(resp_file.name)
                    resp_time_to_seconds = to_sec(resp_time)
                    if req_time_to_seconds is not None and resp_time_to_seconds is not None:
                        delta = abs(req_time_to_seconds - resp_time_to_seconds)
                        if resp_file_closest_to_req_file is None or delta < smallest_delta:
                           resp_file_closest_to_req_file, smallest_delta = resp_file, delta
                candidate = resp_file_closest_to_req_file

            rows.append({
                "depot": depot_region.split("_")[0],
                "route_id": depot_region,
                "date": datestring,
                "request_time": req_time_isoformat,
                "time_bucket": classify_request_timeframe(req_time_isoformat),
                "config_name": configuration_type,
                "num_tasks": num_tasks,
                "num_fixed": num_fixed,
                "request_path": str(req_file),
                "response_path": str(candidate) if candidate else None,
            })

    df_files = pd.DataFrame(rows).sort_values(["route_id","date","request_time"], na_position="last")
    df_files.to_csv(OUTPUT / "index_files.csv", index=False)

####################################
# RUN THE INDEX_FILES
####################################
 
if __name__ == "__main__": 
    df = index_files(REQUESTS_DIR, RESPONSES_DIR, "0521")