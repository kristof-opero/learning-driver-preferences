import pandas as pd
from sklearn.cluster import DBSCAN
import numpy as np

def group_locations_DBSCAN(
    df: pd.DataFrame,
    eps_meters: float = 10.0
) -> pd.DataFrame:
    """
    Groups locations in a DataFrame based on proximity using DBSCAN.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing:
        - 'lat' (latitude)
        - 'lon' (longitude)
        - 'loc_id' (existing column to store group IDs)
    eps_meters : float, optional
        Maximum distance (in meters) between points to be considered
        in the same group. Default is 10 meters.

    Returns
    -------
    pd.DataFrame
        DataFrame with updated 'loc_id' values representing proximity groups.
    """

    # Convert meters to degrees (approximation valid for small distances)
    eps_degrees = eps_meters / 111_000.0

     # Work on a copy to avoid side effects
    df = df.copy()

    # Clear existing loc_id values
    df['loc_id'] = np.nan

    # Extract coordinates
    coords = df[['lat', 'lon']].to_numpy()

    # Run DBSCAN clustering
    db = DBSCAN(
        eps=eps_degrees,
        min_samples=1,
        algorithm='kd_tree',
        metric='euclidean',
        n_jobs=-1
    ).fit(coords)

    # Assign cluster labels to existing loc_id column
    df = df.copy()
    df['loc_id'] = db.labels_

    return df


def group_locations_cell_bucketing(
    df: pd.DataFrame,
    eps_meters: float = 10.0,
    lat_col: str = "lat",
    lon_col: str = "lon",
    loc_id_col: str = "loc_id"
) -> pd.DataFrame:
    """
    Groups locations in a DataFrame so that all points in the same group
    are within eps_meters of each other (max distance ≤ eps_meters).

    Uses grid-based bucketing for speed and guaranteed maximum distance.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with latitude and longitude columns.
    eps_meters : float
        Maximum allowed distance in meters between any two points in the same group.
    lat_col : str
        Latitude column name.
    lon_col : str
        Longitude column name.
    loc_id_col : str
        Column name to store location group IDs.

    Returns
    -------
    pd.DataFrame
        DataFrame with updated loc_id column.
    """

    df = df.copy()

    # Clear existing loc_id
    df[loc_id_col] = np.nan

    # Convert meters to degrees (~valid for small distances)
    meters_per_degree = 111_000.0
    cell_size_deg = eps_meters / meters_per_degree

    # Compute grid cell indices
    df["_cell_x"] = (df[lat_col] / cell_size_deg).astype(int)
    df["_cell_y"] = (df[lon_col] / cell_size_deg).astype(int)

    # Assign loc_id based on unique cell combination
    df[loc_id_col] = df.groupby(["_cell_x", "_cell_y"]).ngroup()

    # Clean up temporary columns
    df.drop(columns=["_cell_x", "_cell_y"], inplace=True)

    return df

def add_max_distance_per_loc_id(
    df: pd.DataFrame,
    lat_col: str = "lat",
    lon_col: str = "lon",
    loc_id_col: str = "loc_id",
    output_col: str = "max_group_distance_m"
) -> pd.DataFrame:
    """
    Adds a column containing the maximum distance (in meters) between
    any two points sharing the same loc_id.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing latitude, longitude, and loc_id columns.
    lat_col : str
        Latitude column name.
    lon_col : str
        Longitude column name.
    loc_id_col : str
        Location group ID column name.
    output_col : str
        Name of the output column.

    Returns
    -------
    pd.DataFrame
        DataFrame with an added column containing max intra-group distance in meters.
    """

    df = df.copy()

    # Approx conversion: degrees → meters
    METERS_PER_DEGREE = 111_000.0

    max_distances = {}

    for loc_id, group in df.groupby(loc_id_col):
        if len(group) < 2:
            max_distances[loc_id] = 0.0
            continue

        coords = group[[lat_col, lon_col]].to_numpy()

        # Compute bounding box max distance (fast and sufficient)
        lat_range = coords[:, 0].max() - coords[:, 0].min()
        lon_range = coords[:, 1].max() - coords[:, 1].min()

        max_distances[loc_id] = np.sqrt(
            lat_range ** 2 + lon_range ** 2
        ) * METERS_PER_DEGREE

    df[output_col] = df[loc_id_col].map(max_distances)

    return df

def add_location_count_changed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bepaal of het aantal unieke locaties is gewijzigd over verschillende requests
    voor elke route op elke dag.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe met kolommen: route_id, date, request_id, loc_id
    
    Returns:
    --------
    pd.DataFrame
        Originele dataframe met nieuwe boolean kolom 'location_count_changed'
        - True: aantal locaties varieerde over requests
        - False: aantal locaties bleef constant
    """
    df = df.copy()
    
    # Stap 1: Tel unieke loc_id per route_id + date + request_id
    counts = (
        df.groupby(["route_id", "date", "request_id"])["loc_id"]
        .nunique()
        .reset_index(name="n_unique_loc")
    )
    
    # Stap 2: Controleer of aantallen veranderen over request_id voor elke route_id + date
    # Als er meer dan één unieke telwaarde is, is het aantal gewijzigd
    changed = (
        counts.groupby(["route_id", "date"])["n_unique_loc"]
        .nunique()
        .gt(1)  # Meer dan één unieke telling → aantal gewijzigd
        .reset_index(name="location_count_changed")
    )
    
    # Stap 3: Merge terug naar originele dataframe
    df = df.merge(changed, on=["route_id", "date"], how="left")
    
    return df

def add_location_change_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bereken hoeveel locaties zijn toegevoegd, verwijderd en netto verandering
    tussen eerste en laatste request voor elke route per dag.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe met kolommen: route_id, date, request_id, loc_id
    
    Returns:
    --------
    pd.DataFrame
        Originele dataframe met nieuwe kolommen:
        - locations_added: aantal loc_ids in laatste request maar niet in eerste
        - locations_removed: aantal loc_ids in eerste request maar niet in laatste
        - locations_net_change: netto verandering (toegevoegd - verwijderd)
        - locations_change_pct: percentage verandering t.o.v. eerste request
    """
    df = df.copy()
    
    # Stap 1: Haal vroegste en laatste request_id per route_id + date
    req_order = (
        df.groupby(["route_id", "date"])["request_id"]
        .agg(["min", "max"])
        .reset_index()
        .rename(columns={"min": "first_request", "max": "last_request"})
    )
    
    # Stap 2: Verzamel set van loc_ids in eerste request
    first_locs = (
        df.merge(req_order, on=["route_id", "date"])
        .query("request_id == first_request")
        .groupby(["route_id", "date"])["loc_id"]
        .apply(set)
        .reset_index(name="first_locs")
    )
    
    # Stap 3: Verzamel set van loc_ids in laatste request
    last_locs = (
        df.merge(req_order, on=["route_id", "date"])
        .query("request_id == last_request")
        .groupby(["route_id", "date"])["loc_id"]
        .apply(set)
        .reset_index(name="last_locs")
    )
    
    # Stap 4: Bereken toevoegingen, verwijderingen en netto verandering
    diff = (
        first_locs.merge(last_locs, on=["route_id", "date"])
        .assign(
            # Locaties in laatste maar niet in eerste = toegevoegd
            locations_added=lambda x: x.apply(
                lambda row: len(row["last_locs"] - row["first_locs"]), axis=1
            ),
            # Locaties in eerste maar niet in laatste = verwijderd
            locations_removed=lambda x: x.apply(
                lambda row: len(row["first_locs"] - row["last_locs"]), axis=1
            ),
            # Netto verandering (kan negatief zijn als meer verwijderd dan toegevoegd)
            locations_net_change=lambda x: x.apply(
                lambda row: len(row["last_locs"]) - len(row["first_locs"]), axis=1
            ),
            # Percentage verandering relatief t.o.v. eerste request aantal
            locations_change_pct=lambda x: x.apply(
                lambda row: (len(row["last_locs"]) - len(row["first_locs"])) / 
                           len(row["first_locs"]) * 100 if len(row["first_locs"]) > 0 else 0,
                axis=1
            )
        )[["route_id", "date", "locations_added", "locations_removed", 
           "locations_net_change", "locations_change_pct"]]
    )
    
    # Stap 5: Merge terug naar originele dataframe
    df = df.merge(diff, on=["route_id", "date"], how="left")
    
    return df

def add_order_changed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bepaal of de volgorde van locaties is gewijzigd tussen opeenvolgende requests
    voor elke route op elke dag.
    
    Beschouwt alleen GEMEENSCHAPPELIJKE locaties (aanwezig in beide requests) om
    valse positieven door toevoegingen/verwijderingen te vermijden.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe met kolommen: route_id, date, request_id, loc_id, time, position_fixed
    
    Returns:
    --------
    pd.DataFrame
        Originele dataframe met nieuwe boolean kolom 'order_changed'
        - True: de relatieve volgorde van gemeenschappelijke locaties is gewijzigd
        - False: volgorde bleef hetzelfde (of eerste request, of geen gemeenschappelijke locaties)
    """
    df = df.copy()
    
    # Stap 1: Bouw sorteerbare datetime voor juiste ordening
    df["datetime"] = pd.to_datetime(
        df["date"].astype(str) + " " + df["time"].astype(str),
        errors='coerce'
    )
    
    # Stap 2: Bouw geordende loc_id sequenties per request
    # Sorteer op datetime en position_fixed om de werkelijke bezoekorder te krijgen
    request_sequences = (
        df.sort_values(["route_id", "date", "request_id", "datetime", "position_fixed"])
        .groupby(["route_id", "date", "request_id"])
        .agg(
            loc_sequence=("loc_id", tuple),  # Geordende tuple van locaties
            request_time=("datetime", "min")  # Tijdstempel van request
        )
        .reset_index()
    )
    
    # Stap 3: Sorteer requests chronologisch binnen elke route+datum
    request_sequences = request_sequences.sort_values(
        ["route_id", "date", "request_time"]
    )
    
    # Stap 4: Vergelijk elk request met het vorige
    def compare_order(group):
        """
        Vergelijk opeenvolgende requests binnen een route+datum groep.
        Vergelijkt alleen gemeenschappelijke locaties om valse positieven te vermijden.
        """
        group = group.sort_values("request_time").reset_index(drop=True)
        order_changed = []
        
        for i in range(len(group)):
            if i == 0:
                # Eerste request: geen vorig request om mee te vergelijken
                order_changed.append(False)
            else:
                prev_seq = group.loc[i-1, "loc_sequence"]
                curr_seq = group.loc[i, "loc_sequence"]
                
                # Vind gemeenschappelijke locaties
                prev_set = set(prev_seq)
                curr_set = set(curr_seq)
                common = prev_set & curr_set
                
                if len(common) == 0:
                    # Geen gemeenschappelijke locaties: kan volgorde niet vergelijken
                    order_changed.append(False)
                else:
                    # Extraheer subsequentie van gemeenschappelijke locaties in hun originele volgorde
                    prev_common = tuple(loc for loc in prev_seq if loc in common)
                    curr_common = tuple(loc for loc in curr_seq if loc in common)
                    
                    # Vergelijk: is de relatieve volgorde gewijzigd?
                    order_changed.append(prev_common != curr_common)
        
        group["order_changed"] = order_changed
        return group
    
    request_sequences = (
        request_sequences
        .groupby(["route_id", "date"], group_keys=False)
        .apply(compare_order)
        .reset_index(drop=True)
    )
    
    # Stap 5: Merge terug naar originele dataframe
    df = df.merge(
        request_sequences[["route_id", "date", "request_id", "order_changed"]],
        on=["route_id", "date", "request_id"],
        how="left"
    )
    
    # Stap 6: Opruimen van tijdelijke kolom
    df = df.drop(columns=["datetime"])
    
    return df

def add_order_change_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tel hoeveel locaties van positie zijn veranderd tussen opeenvolgende requests.
    
    Gebruikt Kendall tau afstand (aantal paargewijze verwisselingen nodig) om
    de omvang van volgorde-wijzigingen te kwantificeren.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe met kolommen: route_id, date, request_id, loc_id, time, position_fixed
    
    Returns:
    --------
    pd.DataFrame
        Originele dataframe met nieuwe kolommen:
        - order_changes_count: aantal paargewijze positie-verwisselingen (inversies)
        - order_changes_pct: percentage van mogelijke verwisselingen die plaatsvonden
    """
    df = df.copy()
    
    # Stap 1: Bouw sorteerbare datetime
    df["datetime"] = pd.to_datetime(
        df["date"].astype(str) + " " + df["time"].astype(str),
        errors='coerce'
    )
    
    # Stap 2: Bouw geordende loc_id sequenties per request
    request_sequences = (
        df.sort_values(["route_id", "date", "request_id", "datetime", "position_fixed"])
        .groupby(["route_id", "date", "request_id"])
        .agg(
            loc_sequence=("loc_id", tuple),
            request_time=("datetime", "min")
        )
        .reset_index()
    )
    
    # Stap 3: Sorteer requests chronologisch
    request_sequences = request_sequences.sort_values(
        ["route_id", "date", "request_time"]
    )
    
    # Stap 4: Bereken Kendall tau afstand (aantal paargewijze verwisselingen)
    def count_order_changes(group):
        """
        Tel paargewijze inversies tussen opeenvolgende requests.
        
        Een inversie vindt plaats wanneer twee locaties in tegengestelde volgorde
        verschijnen in het huidige request vergeleken met het vorige request.
        """
        group = group.sort_values("request_time").reset_index(drop=True)
        change_counts = []
        change_pcts = []
        
        for i in range(len(group)):
            if i == 0:
                # Eerste request: geen vergelijking
                change_counts.append(0)
                change_pcts.append(0.0)
            else:
                prev_seq = list(group.loc[i-1, "loc_sequence"])
                curr_seq = list(group.loc[i, "loc_sequence"])
                
                # Vind gemeenschappelijke locaties
                common = set(prev_seq) & set(curr_seq)
                
                if len(common) < 2:
                    # Minstens 2 gemeenschappelijke items nodig om volgorde te hebben
                    change_counts.append(0)
                    change_pcts.append(0.0)
                else:
                    # Extraheer gemeenschappelijke subsequenties
                    prev_common = [loc for loc in prev_seq if loc in common]
                    curr_common = [loc for loc in curr_seq if loc in common]
                    
                    # Maak positie mapping voor vorige sequentie
                    prev_positions = {loc: idx for idx, loc in enumerate(prev_common)}
                    
                    # Tel inversies: hoeveel paren zijn uit volgorde?
                    inversions = 0
                    n = len(curr_common)
                    for j in range(n):
                        for k in range(j + 1, n):
                            # Als curr_common[j] na curr_common[k] kwam in prev_common
                            if prev_positions[curr_common[j]] > prev_positions[curr_common[k]]:
                                inversions += 1
                    
                    # Maximum mogelijke inversies voor n items: n*(n-1)/2
                    max_inversions = n * (n - 1) / 2
                    pct = (inversions / max_inversions * 100) if max_inversions > 0 else 0
                    
                    change_counts.append(inversions)
                    change_pcts.append(pct)
        
        group["order_changes_count"] = change_counts
        group["order_changes_pct"] = change_pcts
        return group
    
    request_sequences = (
        request_sequences
        .groupby(["route_id", "date"], group_keys=False)
        .apply(count_order_changes)
        .reset_index(drop=True)
    )
    
    # Stap 5: Merge terug
    df = df.merge(
        request_sequences[["route_id", "date", "request_id", 
                          "order_changes_count", "order_changes_pct"]],
        on=["route_id", "date", "request_id"],
        how="left"
    )
    
    # Stap 6: Opruimen
    df = df.drop(columns=["datetime"])
    
    return df