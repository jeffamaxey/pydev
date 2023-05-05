import glob
import math
import multiprocessing as mp
import os
import smtplib
import warnings
from datetime import datetime, timedelta
from decimal import Decimal
from os.path import dirname

import numpy as np
import pandas as pd
import sqlalchemy as sa
import yaml
from bcpandas import SqlCreds
from importlib.machinery import SourceFileLoader

from utils import (bcp, df_replace_nan_and_inf, generate_run_id, make_table,
                   send_text_notif, loop_queries)

# Supresses the SettingWithCopyWarning and FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None  # default='warn'

# Open the config files and set global config variables.
with open(f'{dirname(dirname(__file__))}/config.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.Loader)

with open(f'{dirname(dirname(__file__))}/table_config.yaml', 'r') as f:
    table_config = yaml.load(f, Loader=yaml.Loader)


DEBUG = config["debug"]

# General Parameters
SCHEMA = config["general_params"]["schema"]
FILTER_TABLE = config["general_params"]["filter_table"]
RUN_ID = generate_run_id()
VERSION = config["version"]
TABLE_SUFFIX = f'v{config["version"].split(".")[-1]}'
TEXT_NOTIFICATIONS = config["general_params"]["text_notifications"]

# File Paths
ERROR_LOG_PATH = config["file_paths"]["error_log_path"]
RESULT_PATH = config["file_paths"]["result_path"]

# Projection Parameteres
RUN_BASELINE_READIN = config["projection_params"]["run_baseline_readin"]
RUN_BASELINE_LIVE = config["projection_params"]["run_baseline_live"]
RUN_BASELINE_DRIFT = config["projection_params"]["run_baseline_drift"]
RUN_BASELINE_DYNAMIC = config["projection_params"]["run_baseline_dynamic"]
RUN_CORRECTED = config["projection_params"]["run_corrected"]
USE_MULTIPROCESS = config["projection_params"]["use_multiprocess"]
N_CORES = config["projection_params"]["n_cores"]
N_BATCHES = config["projection_params"]["n_batches"]
TO_SQL = config["projection_params"]["to_sql"]
FILTER_SCENARIOS = config["projection_params"]["filter_scenarios"]
FILTER_PINS = config["projection_params"]["filter_pins"]

if config["base_contract_proj_params"]["use_most_recent_version"]:
    BASE_CONTRACT_BLA_VERSION = TABLE_SUFFIX
else:
    BASE_CONTRACT_BLA_VERSION = config["base_contract_proj_params"]["base_contract_bla_version"]

# Forcing certain parameters False as a safeguard.
if DEBUG:
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    TEXT_NOTIFICATIONS = False

TO_CSV = True if TO_SQL else config["projection_params"]["to_csv"]
# Notification Credentials
SENDER_NAME = config["notification_credentials"]["username"]
SENDER_PASS = config["notification_credentials"]["app_password"]
RECEIVERS = config["notification_credentials"]["receivers"]

# Setting SQL credentials.
DRIVER = config["credentials"]["driver"]
SERVER = config["credentials"]["server"]
DATABASE = config["credentials"]["database"]
USERNAME = config["credentials"]["username"]
PASSWORD = config["credentials"]["password"]
CONN_STR = f'''
    DRIVER={DRIVER};
    SERVER={SERVER};
    DATABASE={DATABASE};
    UID={USERNAME};
    PWD={PASSWORD};
'''
CONN_URL = sa.engine.URL.create(
    drivername="mssql+pyodbc",
    username=USERNAME,
    password=PASSWORD,
    host=SERVER,
    database=DATABASE,
    query={
        "driver": "ODBC Driver 17 for SQL Server",
    },
)
CREDS = SqlCreds(
    server=SERVER,
    database=DATABASE,
    username=USERNAME,
    password=PASSWORD,
)

GATE_NAMES = {
    "gate_1": "Gate 1: TEMP Trad Over Allocated and Legacy Balance = $0",
    "gate_2": "Gate 2: No Over Allocation and Legacy Balance = $0",
    "gate_3": "Gate 3: TEMP Trad Over Allocated and Legacy Balance > $0",
    "gate_4": "Gate 4: No TEMP Trad Over Allocation and Legacy Over Allocated",
    "gate_5": "Gate 5: No Over Allocation and Legacy Balance > $0",
    "gate_6": "Gate 6: TEMP Trad Over Allocated and Legacy Over Allocated",
    "gate_7": "Gate 7: X-Case TEMP Trad Not Over Allocated",
    "gate_8": "Gate 8: X-Case TEMP Trad Over Allocated and Legacy Balance = $0",
}
TRAD_TICKERS = [
    'TICP1',
    'TIAVR',
    'TIAFX',
    'TIARA',
    'TIAGR',
    'TIASR',
    'TIAGS',
    'TIAIR',
    'TIRC1',
    'TICP2',
    'TICP3',
    'TIATP',
    'TIAIP',
    'TC1TP',
    'TC1IO',
    'TIAIX',
]


def get_filter_pins_suffix():
    """Returns a string that indicates if certain PINs were filtered in the run.

    This function checks if the FILTER_PINS and FILTER_SCENARIOS global
    variables are empty. If they are, then the suffix string will remain empty.
    If they aren't, then the suffix will contain the number of pins included in
    the run and/or the number of scenarios included.

    E.g., end = '_1pins_64scenarios'; end = '_12pins'

    Args:
        None
    
    Returns:
        A string that is appended to file names and table names as a suffix.
        This string may be empty.
    """
    end = ''
    if FILTER_PINS:
        end += f'_{len(FILTER_PINS)}pins'
    if FILTER_SCENARIOS:
        end += f'_{len(FILTER_SCENARIOS)}scenarios'
    
    return end

def identify_overalloc(df, implicit_ind, illiquid_ind, date_tm_trad_legacy_bal, ca_ticker_map, model_weight):
    """Identify if there is an overallocation and assign the proper logic gate.

    This function will identify which rebalancing logic gate to apply. First, 
    the intermediate input dfs are joined such that we have a reference
    between pin-plan-subplan and each ticker's actual and target %s. Moreover,
    this function also identifies legacy vs active balances, and traditional
    vs non-traditional tickers forming the baseline for assessing the logic to
    determine the proper rebalancing gate.

    Args:
        df: Input DataFrame containing the ticker data for one pin-plan-subplan
          on the rebalancing date.
        implicit_ind: Indicator for checking implicit consideration. For
          determining gates 7 and 8.
        illiquid_ind: Indicates if a certain ticker is illiquid. Used for
          determining trad tickers for consideration.
        date_tm_trad_legacy_bal: Trad legacy balance value used for calculating
          x_case_trad_pct which is used for determining gates 7 and 8.
        ca_ticker_map: Identifies the legacy vs active assets for each
          participant.
        model_weight: Identifies the target model weights.
    
    Returns:
        A tuple specifying which rebalancing logic gate will be applied,
        the input dataframe which has been modified with the addition of
        several columns following the logic outlined in the excel model, and
        both the active and target percentages for TEMP traditional assets.
    """
    df['prev_day_proj'] = round(df['prev_day_proj'], 8)

    # Join the input df with the considered assets ticker map.
    df = pd.merge(
        df,
        ca_ticker_map,
        how="left",
        left_on=["ic220_priceid"],
        right_on =["mf_ticker_new"]
    )
    # Fill NaN legacy balance values as 0.
    if ca_ticker_map.empty:
        df["raw_legacy_balance"] = 0
    else:
        df["raw_legacy_balance"] = df["raw_legacy_balance"].fillna(0)

    # Create an identifier for traditional assets by filtering for TICP1.
    if illiquid_ind == 1:
        df["is_aait_trad"] = np.where(
            df["ic220_priceid"].isin(TRAD_TICKERS),
            1,
            0
        )
    else:
        df["is_aait_trad"] = 0

    # Join the resulting df with model weights df.
    model_weight_cols = [
        "pin",
        "plan",
        "sub_plan",
        "calendar_dates",
        "fund_ticker",
        "fund_weight",
    ]
    df = pd.merge(
        df,
        model_weight[model_weight_cols],
        how='left', 
        left_on=["pin","plan","sub_plan","calendar_dates","ic220_priceid"], 
        right_on=["pin","plan","sub_plan","calendar_dates","fund_ticker"],
    )
    # Fill NaN fund weights with 0.
    df["fund_weight"] = df["fund_weight"].fillna(0)

    # Trad legacy balance + trad prev proj / total legacy + total prev proj.
    # This logic assumes each participant has only 1 trad ticker. There is
    #   currently one known case where this will break.
    x_case_trad_pct = (
        date_tm_trad_legacy_bal + 
        safe_divide(
            (df['is_aait_trad'] * df['prev_day_proj']).sum(),
            (df['raw_legacy_balance'].sum() + df['prev_day_proj'].sum())
        )
    )
    df['x_case_trad_pct'] = x_case_trad_pct

    # Calculate the legacy balance.
    df["legacy_balance"] = np.where(
        df["fund_weight"] > 0,
        df["raw_legacy_balance"],
        0
    )
    # Calculate target cash.
    df["target_cash"] = (
        (df["prev_day_proj"].sum() + df["legacy_balance"].sum()) * 
        (df["fund_weight"] / 100)
    )
    # Calculate unsellable assets.
    df["unsellable"] = np.where(
        (df["is_aait_trad"] == 1), # & (illiquid_ind == 1),
        df["legacy_balance"] + df["prev_day_proj"],
        df["legacy_balance"]
    )
    # Calculate the difference between target cash and unsellable assets.
    df["difference"] = df["target_cash"] - df["unsellable"]
    # Identify which tickers are overallocated.
    df["over_allocation"] = np.where(df["difference"] < 0, 1, 0)
    # Calculate active %s per ticker
    df["active_pct"] = df["prev_day_proj"] / (df["prev_day_proj"].sum())

    # Calculate trad active and trad target %s.
    trad_active_pct =  (df["active_pct"] * df["is_aait_trad"]).sum()
    trad_target_pct =  (df["fund_weight"] * df["is_aait_trad"]).sum() / 100
    df["trad_active_pct"] = trad_active_pct
    df["trad_target_pct"] = trad_target_pct

    # Determine the rebalancing gate based on over allocation and legacy assets.
    legacy_bal_0 = df["legacy_balance"].sum() == 0
    legacy_overalloc = (
        (df["over_allocation"] == 1) & (df["is_aait_trad"] != 1)
    ).max()
    aait_trad_overalloc = (
        (df["over_allocation"] == 1) & (df["is_aait_trad"] == 1)
    ).max()

    # Gate 7
    if implicit_ind == 1 and x_case_trad_pct <= trad_target_pct:
        rebal_gate = GATE_NAMES["gate_7"]
    elif implicit_ind == 1:
        rebal_gate = GATE_NAMES["gate_8"]
    elif legacy_bal_0 and not aait_trad_overalloc and not legacy_overalloc:
        rebal_gate = GATE_NAMES["gate_2"]
    elif not legacy_bal_0 and not aait_trad_overalloc and not legacy_overalloc:
        rebal_gate = GATE_NAMES["gate_5"]
    elif legacy_bal_0 and aait_trad_overalloc:
        rebal_gate = GATE_NAMES["gate_1"]
    elif not legacy_bal_0 and aait_trad_overalloc and not legacy_overalloc:
        rebal_gate = GATE_NAMES["gate_3"]
    elif not aait_trad_overalloc:
        rebal_gate = GATE_NAMES["gate_4"]
    else:
        rebal_gate = GATE_NAMES["gate_6"]

    return rebal_gate, df, trad_active_pct, trad_target_pct

def apply_rounding(temp, column, neg_t381_ind=0, t815_making_whole_100_pct=False, trad_floor_adj=False):
    """Applies rounding to a particular column to ensure it sums to 100%.

    This function applies a special set of rounding rules. If the sum of
    percentages exceeds 100%, then it will subtract percentage from the ticker
    with the greatest allocation. If the sum of percentages is less than 100%,
    then it will add percentage from the ticker with the least allocation. The
    full scope of this rounding algorithm will be detailed more greatly in the
    excel model and associated documentation.

    Args:
        temp: The main dataframe for which the rebalancing weights are calculated.
          Everything ultimately is an addition to this main DataFrame. 
        column: The column of percentages specified. Default = 0
        neg_t381_ind: Indicates if the calculated t381 value is less than 0.
        t815_making_whole_100_pct: Boolean indicator for certain logical
          conditions set in the rounding function. Default = False
        over_alloc: Optional keyword argument to specify one special case. Only
          one instance in the run needs to use over_alloc=True, and all others
          can default to over_alloc=False.
        trad_floor_adj: Optional keyword argument to specify an override on the
          over_alloc indicator for the trad_floor_adj at the end of gate 4.
    Returns:
        A Series of the updated and rounded percentages for the specified
        column.
    """
    df = temp.copy()
    #df[column]=df[column].apply(lambda x: round(x,10))
    column_total_pct = df[column].sum()
    if t815_making_whole_100_pct:
        over_alloc = df['over_allocation'].sum() >= 1
    elif "prev_over_alloc" in df.columns and neg_t381_ind == 0 and not trad_floor_adj:
        over_alloc = df['prev_over_alloc'].sum() >= 1
    else:
        over_alloc = False

    # If the sum of percents is over 100%
    if column_total_pct > 1:
        if t815_making_whole_100_pct:
            df[f"rank_{column}"] = (
                df[(df[column] != 0) & (df['over_allocation'] == 0)]
                .sort_values([column, 'ic220_priceid'], ascending=[False, True])
                .groupby(['pin'])
                .cumcount() + 1
            )
        elif over_alloc:
            df[f"rank_{column}"] = (
                df[(df[column] != 0) & (df['prev_over_alloc'] == 0)]
                .sort_values([column, 'ic220_priceid'], ascending=[False, True])
                .groupby(['pin'])
                .cumcount() + 1
            )
        else:
            df[f"rank_{column}"] = (
                df[df[column] != 0]
                .sort_values([column, 'ic220_priceid'], ascending=[False, True])
                .groupby(['pin'])
                .cumcount() + 1
            )

        if df["is_aait_trad"].sum() == 0:
            adjusted_rounding = np.where(
                df[f"rank_{column}"] == 1,
                df[column] + (1 - column_total_pct),
                df[column]
            )
        else:
            if df.loc[df["is_aait_trad"] == 1 , f"rank_{column}"].iloc[0] == 1:
                adjusted_rounding = np.where(
                    df[f"rank_{column}"] == 2,
                    df[column] + (1 - column_total_pct),
                    df[column]
                )
            else:
                adjusted_rounding = np.where(
                    df[f"rank_{column}"] == 1,
                    df[column] + (1 - column_total_pct),
                    df[column]
                )

        # Normal heirarchical rounding treatment.
        if round(adjusted_rounding.sum(), 2) > 1:
            df[f"rank_{column}"] = (
                df[df[column] != 0]
                .sort_values([column, 'ic220_priceid'], ascending=[False, True])
                .groupby(['pin'])
                .cumcount() + 1
            )
            if df["is_aait_trad"].sum() == 0:
                adjusted_rounding = np.where(
                    df[f"rank_{column}"] == 1,
                    df[column] + (1 - column_total_pct),
                    df[column]
                )
            else:
                if df.loc[df["is_aait_trad"] == 1 , f"rank_{column}"].iloc[0] == 1:
                    adjusted_rounding = np.where(
                        df[f"rank_{column}"] == 2,
                        df[column] + (1 - column_total_pct),
                        df[column]
                    )
                else:
                    adjusted_rounding = np.where(
                        df[f"rank_{column}"] == 1,
                        df[column] + (1 - column_total_pct),
                        df[column]
                    )

        if round(adjusted_rounding.sum(), 2) > 1:
            adjusted_rounding = df['is_aait_trad']
            
        # Negative t381 heirarchical adjustment logic.
        # CREATE MAKING WHOLE --> CHECK MAKING WHOLE --> CREATE ADJUSTMENT --> LOOP
        if np.where(adjusted_rounding < 0, 1, 0).sum() >= 1:
            # Pull necessary data from main df
            neg_chk = pd.DataFrame()
            neg_chk["ic220_priceid"] = df["ic220_priceid"]
            neg_chk["is_aait_trad"] = df["is_aait_trad"]
            neg_chk["prev_over_alloc"] = df["prev_over_alloc"]
            neg_chk["fund_weight"] = df["fund_weight"]
            neg_chk['pin'] = df['pin']
            neg_chk["not_over_alloc"] = np.where(
                neg_chk["is_aait_trad"] + neg_chk["prev_over_alloc"] >= 1,
                0,
                1
            )
            # Pull the calculated t381s and ceiling negative values to 0.
            neg_chk["making_to_whole_100"] = adjusted_rounding
            neg_chk["making_to_whole_100_adjusted"] = np.where(
                neg_chk["making_to_whole_100"] < 0,
                0,
                neg_chk["making_to_whole_100"],
            )
            # Step 1: Iterate through non-trad, non-overallocated tickers.
            while (
                np.where(neg_chk["making_to_whole_100"] < 0, 1, 0).sum() >= 1 and
                (neg_chk["not_over_alloc"] * neg_chk["making_to_whole_100_adjusted"]).sum() > 0
            ):
                column_total_pct = neg_chk['making_to_whole_100_adjusted'].sum()

                # Rank tickers
                if over_alloc:
                    neg_chk["rank_making_to_whole_100_adjusted"] = (
                        neg_chk[(neg_chk["fund_weight"] != 0) & (neg_chk['prev_over_alloc'] == 0)]
                        .sort_values(['making_to_whole_100_adjusted', 'ic220_priceid'], ascending=[False, True])
                        .groupby(['pin'])
                        .cumcount() + 1
                    )
                else:
                    neg_chk["rank_making_to_whole_100_adjusted"] = (
                        neg_chk[neg_chk["fund_weight"] != 0]
                        .sort_values(['making_to_whole_100_adjusted', 'ic220_priceid'], ascending=[False, True])
                        .groupby(['pin'])
                        .cumcount() + 1
                    )
                
                # Calc new making_to_whole_100
                if neg_chk["is_aait_trad"].sum() == 0:
                    neg_chk['making_to_whole_100'] = np.where(
                        neg_chk["rank_making_to_whole_100_adjusted"] == 1,
                        neg_chk['making_to_whole_100_adjusted'] + (1 - column_total_pct),
                        neg_chk['making_to_whole_100_adjusted']
                    )
                else:
                    if neg_chk.loc[neg_chk["is_aait_trad"] == 1 , "rank_making_to_whole_100_adjusted"].iloc[0] == 1:
                        neg_chk['making_to_whole_100'] = np.where(
                            neg_chk["rank_making_to_whole_100_adjusted"] == 2,
                            neg_chk['making_to_whole_100_adjusted'] + (1 - column_total_pct),
                            neg_chk['making_to_whole_100_adjusted']
                        )
                    else:
                        neg_chk['making_to_whole_100'] = np.where(
                            neg_chk["rank_making_to_whole_100_adjusted"] == 1,
                            neg_chk['making_to_whole_100_adjusted'] + (1 - column_total_pct),
                            neg_chk['making_to_whole_100_adjusted']
                        )
  
                neg_chk["making_to_whole_100_adjusted"] = np.where(
                    neg_chk["making_to_whole_100"] < 0,
                    0,
                    neg_chk["making_to_whole_100"],
                )
            
            # Check if negative t381 is solved.
            if np.where(neg_chk["making_to_whole_100"] < 0, 1, 0).sum() == 0:
                return neg_chk["making_to_whole_100"]
            
            # Step 2: Iterate through non-trad, overallocated tickers.
            while (
                np.where(neg_chk["making_to_whole_100"] < 0, 1, 0).sum() >= 1 and
                (neg_chk["prev_over_alloc"] * neg_chk["making_to_whole_100_adjusted"]).sum() > 0
            ):
                column_total_pct = neg_chk['making_to_whole_100_adjusted'].sum()

                neg_chk["rank_making_to_whole_100_adjusted"] = (
                    neg_chk[neg_chk["fund_weight"] != 0]
                    .sort_values(['making_to_whole_100_adjusted', 'ic220_priceid'], ascending=[False, True])
                    .groupby(['pin'])
                    .cumcount() + 1
                )
                
                if neg_chk["is_aait_trad"].sum() == 0:
                    neg_chk['making_to_whole_100_adjusted'] = np.where(
                        neg_chk["rank_making_to_whole_100_adjusted"] == 1,
                        neg_chk['making_to_whole_100_adjusted'] + (1-column_total_pct),
                        neg_chk['making_to_whole_100_adjusted']
                    )
                else:
                    if neg_chk.loc[neg_chk["is_aait_trad"] == 1 , "rank_making_to_whole_100_adjusted"].iloc[0] == 1:
                        neg_chk['making_to_whole_100_adjusted'] = np.where(
                            neg_chk["rank_making_to_whole_100_adjusted"] == 2,
                            neg_chk['making_to_whole_100_adjusted'] + (1 - column_total_pct),
                            neg_chk['making_to_whole_100_adjusted']
                        )
                    else:
                        neg_chk['making_to_whole_100_adjusted'] = np.where(
                            neg_chk["rank_making_to_whole_100_adjusted"] == 1,
                            neg_chk['making_to_whole_100_adjusted'] + (1 - column_total_pct),
                            neg_chk['making_to_whole_100_adjusted']
                        )
                    
                neg_chk["making_to_whole_100_adjusted"] = np.where(
                    neg_chk["making_to_whole_100"] < 0,
                    0,
                    neg_chk["making_to_whole_100"],
                )
            
            if np.where(neg_chk["making_to_whole_100"] < 0, 1, 0).sum() == 0:
                return neg_chk["making_to_whole_100"]
            else:
                # Step 3: If the t381s are still negative after removing all
                #   non trad tickers, then we allocate 100% to trad.
                return neg_chk['is_aait_trad']
        # END: Negative t381 protection logic.
        
    # If the sum of percents is under 100%
    elif column_total_pct < 1:
        if t815_making_whole_100_pct:
            df[f"rank_{column}"] = (
                df[(df[column] != 0) & (df['over_allocation'] == 0)]
                .sort_values([column, 'ic220_priceid'], ascending=[True, True])
                .groupby(['pin'])
                .cumcount() + 1
            )
        elif over_alloc:
            df[f"rank_{column}"] = (
                df[(df[column] != 0) & (df['prev_over_alloc'] == 0)]
                .sort_values([column, 'ic220_priceid'], ascending=[True, True])
                .groupby(['pin'])
                .cumcount() + 1
            )
        else:
            df[f"rank_{column}"] = (
                df[df[column]!=0]
                .sort_values([column, 'ic220_priceid'], ascending=[True, True])
                .groupby(['pin'])
                .cumcount() + 1
            )
        if df["is_aait_trad"].sum() == 0:
            adjusted_rounding = np.where(
                df[f"rank_{column}"] == 1,
                df[column] + (1 - column_total_pct),
                df[column]
            )
        else:
            if df.loc[df["is_aait_trad"] == 1 , f"rank_{column}"].iloc[0] == 1:
                adjusted_rounding = np.where(
                    df[f"rank_{column}"] == 2,
                    df[column] + (1 - column_total_pct),
                    df[column]
                )
            else:
                adjusted_rounding = np.where(
                    df[f"rank_{column}"] == 1,
                    df[column] + (1 - column_total_pct),
                    df[column]
                )
        
        if round(adjusted_rounding.sum(), 2) < 1:
            if t815_making_whole_100_pct:
                df[f"rank_{column}"] = (
                    df[(df["fund_weight"] != 0) & (df['over_allocation'] == 0)]
                    .sort_values([column, 'ic220_priceid'], ascending=[True, True])
                    .groupby(['pin'])
                    .cumcount() + 1
                )
            elif over_alloc:
                df[f"rank_{column}"] = (
                    df[(df["fund_weight"] != 0) & (df['prev_over_alloc'] == 0)]
                    .sort_values([column, 'ic220_priceid'], ascending=[True, True])
                    .groupby(['pin'])
                    .cumcount() + 1
                )
            else:
                df[f"rank_{column}"] = (
                    df[df["fund_weight"]!=0]
                    .sort_values([column, 'ic220_priceid'], ascending=[True, True])
                    .groupby(['pin'])
                    .cumcount() + 1
                )
            if df["is_aait_trad"].sum() == 0:
                adjusted_rounding = np.where(
                    df[f"rank_{column}"] == 1,
                    df[column] + (1 - column_total_pct),
                    df[column]
                )
            else:
                if df.loc[df["is_aait_trad"] == 1 , f"rank_{column}"].iloc[0] == 1:
                    adjusted_rounding = np.where(
                        df[f"rank_{column}"] == 2,
                        df[column] + (1 - column_total_pct),
                        df[column]
                    )
                else:
                    adjusted_rounding = np.where(
                        df[f"rank_{column}"] == 1,
                        df[column] + (1 - column_total_pct),
                        df[column]
                    )
    
    else:
        df[f"rank_{column}"] = 0
        adjusted_rounding = df[column]

    return adjusted_rounding

def round_down(x, decimal_pnts):
    """Rounds a number down at the specified number of decimal places.

    Due to the strangeness of floating point numbers, we round the multiplcation
    (x * factor) by decimal_pnts + 1. Python ouput to a muplitiplication like
    0.14 **2 is 14.000000000000002 which leads to an error in the math.ceil
    function.

    Args:
        x: The number to be rounded down.
        decimal_pnts: the number of dicimal places to be rounded at.
    
    Returns:
        A number rounded down at the specified number of decimal places.
    """
    factor = 10 ** decimal_pnts
    return math.floor(round((x * factor), decimal_pnts + 1)) / factor

def round_up(x, decimal_pnts):
    """Rounds a number up at the specified number of decimal places.

    Due to the strangeness of floating point numbers, we round the multiplcation
    (x * factor) by decimal_pnts + 1. Python ouput to a muplitiplication like
    0.14 **2 is 14.000000000000002 which leads to an error in the math.ceil
    function. 

    Args:
        x: The number to be rounded up.
        decimal_pnts: the number of dicimal places to be rounded at.
    
    Returns:
        A number rounded down at the specified number of decimal places.
    """
    if np.isnan(x):
        x = 0
    
    factor = 10 ** decimal_pnts
    return math.ceil(round((x * factor), decimal_pnts + 1)) / factor 

def check_greater_than_101(df, column):
    """Creates an array of 1s and 0s denoting if a columns value is > 101%.

    This function creates a temporary dataframe based on an imput df and
    column. If the value from the input column contains NaN or inf, then the
    output value for that entry is 0. Otherwise, it checks if "check_101_pct" is
    greater than 1.01. If yes, then the return value is 1, otherwise it is 0.

    Args:
        df: A dataframe containing the data being assessed
        column: A string name denoting which column of the dataframe to assess.
    
    Returns:
        A numpy array of 1s and 0s, the 1s denote the values of that column that
        are not inf or null and are greater than 1.01.
    """
    temp = pd.DataFrame()
    temp["check_101_pct"] = df[column]
    temp["is_inf_check"] = [math.isinf(x) for x in df[column]]
    temp["is_inf_check"] = temp["is_inf_check"].astype(int)
    temp["is_nan_check"] = [math.isnan(x) for x in df[column]]
    temp["is_nan_check"] = temp["is_nan_check"].astype(int)
    temp["ind"] = temp["is_inf_check"] + temp["is_nan_check"]

    return np.where(
        temp["ind"] == 1, 0, np.where(temp["check_101_pct"] > 1.01, 1, 0)
    )

def check_less_than_101(df, column):
    """Creates an array of 1s and 0s denoting if a columns value is < 101%.

    This function creates a temporary dataframe based on an imput df and
    column. If the value from the input column contains NaN or inf, then the
    output value for that entry is 0. Otherwise, it checks if "check_101_pct" is
    less than 1.01. If yes, then the return value is 1, otherwise it is 0.

    Args:
        df: A dataframe containing the data being assessed
        column: A string name denoting which column of the dataframe to assess.
    
    Returns:
        A numpy array of 1s and 0s, the 1s denote the values of that column that
        are not inf or null and are less than 1.01.
    """
    temp = pd.DataFrame()
    temp["check_101_pct"] = df[column]
    temp["is_inf_check"] = [math.isinf(x) for x in df[column]]
    temp["is_inf_check"] = temp["is_inf_check"].astype(int)
    temp["is_nan_check"] = [math.isnan(x) for x in df[column]]
    temp["is_nan_check"] = temp["is_nan_check"].astype(int)
    temp["ind"] = temp["is_inf_check"] + temp["is_nan_check"]

    return np.where(
        temp["ind"] == 1, 0, np.where(temp["check_101_pct"] < 1.01, 1, 0)
    )

def calc_gate_1(df, trad_active_pct, trad_target_pct):
    """Rebalance when TEMP Trad is overallocated and legacy Balance is Zero.

    This function represents Gate 1 - the first of eight logic gates designed to
    handle different combinations of asset allocation and legacy balances. Gate
    1 is triggered when the pin-plan-subplan has TEMP Traditional assets
    overallocated, but Zero legacy assets.

    The purpose of each logic gate is to return a live calculation of the
    pin-plan-subplan's t381 and t815 weight distributions. These functions are
    direct implementations from the models built in excel by the LEC Actuary
    team.

    Args:
        df: A table representing the target pin-plan-subplan's asset allocations
          on the day of rebalancing.
        trad_active_pct: The current percentage of traditional assets
          representing the target pin-plan-subplan's full portfolio.
        trad_target_pct: The desired percentage of traditional assets
          representing the target pin-plan-subplan's full portfolio after the
          rebalancing has occured.
    
    Returns:
        A tuple of three distinct results to be unpacked. The first is a
        modification of the input DataFrame containing staging values that were
        used for calculating the new t381 and t815 values, and the new t381 and
        t815 values. The second output in the tuple is the t381_calc value, and
        the third is the t815_calc value.
    """
    gate_1_history = pd.DataFrame()

    # Round the trad % for active assets with a manual decimal rounding system.
    # NOTE: This can be replaced with the decimal_round function.
    if round(trad_active_pct, 4) > round_down(trad_active_pct, 2):
        trad_new_pct = round_up(trad_active_pct, 2)
    else:
        trad_new_pct = round_down(trad_active_pct, 4)

    # Sets all trad tickers to 0% and adjusts all other tickers on target %s.
    df["alloc_non_trad"] = np.where(
        df["is_aait_trad"] == 1,
        0,
        (df["fund_weight"] / 100) / (1 - trad_target_pct)
    )
    # Replace trad ticker allocations with the rounded %s.
    df["replace_aait_trad_new_pct"] = np.where(
        df["is_aait_trad"]==1,
        trad_new_pct,
        round(df["alloc_non_trad"] * (1 - trad_new_pct), 2)
    )
    # Get t381 values by applying rounding logic.
    df["replace_aait_trad_new_pct"]=df["replace_aait_trad_new_pct"].apply(lambda x : round(x,10))
    df["t381_calc"] = apply_rounding(df, "replace_aait_trad_new_pct")
    df["t381_calc"] = df["t381_calc"].apply(lambda x: round(x, 2))
    # Sum t381 %s for trad tickers
    trad_t381 = (df["t381_calc"] * df["is_aait_trad"]).sum()
    df["trad_t381"] = trad_t381

    df["remove_trad"] = round(df["alloc_non_trad"], 2)
    df["t815_calc"] = apply_rounding(df, "remove_trad")
    df["t815_calc"] = df["t815_calc"].apply(lambda x: round(x, 2))

    df = df_replace_nan_and_inf(df)
    df = df_attach_metadata(df, run_id=RUN_ID, version=VERSION)
    gate_1_history = pd.concat([gate_1_history, df])

    return gate_1_history, df["t381_calc"].values, df["t815_calc"].values

def calc_gate_2(df, trad_active_pct, trad_target_pct):
    """Rebalance when TEMP Trad is NOT overallocated and legacy balance is Zero.
    
    This function represents Gate 2 - the second of eight logic gates designed to
    handle different combinations of asset allocation and legacy balances. Gate
    2 is triggered when the pin-plan-subplan has TEMP Traditional assets NOT
    overallocated, and Zero legacy assets.

    The purpose of each logic gate is to return a live calculation of the
    pin-plan-subplan's t381 and t815 weight distributions. These functions are
    direct implementations from the models built in excel by the LEC Actuary
    team.

    Args:
        df: A table representing the target pin-plan-subplan's asset allocations
          on the day of rebalancing.
        trad_active_pct: The current percentage of traditional assets
          representing the target pin-plan-subplan's full portfolio.
        trad_target_pct: The desired percentage of traditional assets
          representing the target pin-plan-subplan's full portfolio after the
          rebalancing has occured.
    
    Returns:
        A tuple of three distinct results to be unpacked. The first is a
        modification of the input DataFrame containing staging values that were
        used for calculating the new t381 and t815 values, and the new t381 and
        t815 values. The second output in the tuple is the t381_calc value, and
        the third is the t815_calc value.
    """
    gate_2_history = pd.DataFrame()
    
    # Round the trad % for active assets with a manual decimal rounding system.
    if round(trad_active_pct, 4) > round_down(trad_active_pct, 2):
        trad_new_pct = round_up(trad_active_pct, 2)
    else:
        trad_new_pct = round_down(trad_active_pct, 4)

    df["alloc_non_trad"] = np.where(
        df["is_aait_trad"] == 1, 0, (df["fund_weight"] / 100) / (1 - trad_target_pct)
    )

    trad_replace_pct = max(trad_new_pct, trad_target_pct)
    df["trad_replace_pct"] = trad_replace_pct

    # Calculate t381 and t815 %s.
    df["replace_aait_trad_new_pct"] = np.where(
        df["is_aait_trad"] == 1,
        trad_replace_pct,
        round(df["alloc_non_trad"] * (1 - trad_replace_pct), 2),
    )
    df["replace_aait_trad_new_pct"] =df["replace_aait_trad_new_pct"].apply(lambda x: round(x,10))
    df["t381_calc"] = apply_rounding(df, "replace_aait_trad_new_pct")
    df["t381_calc"] = df["t381_calc"].apply(lambda x: round(x, 2))
    df["t815_calc"] = df["t381_calc"]

    df = df_replace_nan_and_inf(df)
    df = df_attach_metadata(df, run_id=RUN_ID, version=VERSION)
    gate_2_history = pd.concat([gate_2_history, df])

    return gate_2_history, df["t381_calc"].values, df["t815_calc"].values

def calc_gate_3(df, trad_active_pct, trad_target_pct):
    """Rebalance when TEMP Trad is Overallocated AND legacy balance > 0.

    This function represents Gate 3 - the third of eight logic gates designed to
    handle different combinations of asset allocation and legacy balances. Gate
    2 is triggered when the pin-plan-subplan has TEMP Traditional assets
    overallocated, and legacy assets greater than zero.

    The purpose of each logic gate is to return a live calculation of the
    pin-plan-subplan's t381 and t815 weight distributions. These functions are
    direct implementations from the models built in excel by the LEC Actuary
    team.

    Args:
        df: A table representing the target pin-plan-subplan's asset allocations
          on the day of rebalancing.
        trad_active_pct: The current percentage of traditional assets
          representing the target pin-plan-subplan's full portfolio.
        trad_target_pct: The desired percentage of traditional assets
          representing the target pin-plan-subplan's full portfolio after the
          rebalancing has occured.
    
    Returns:
        A tuple of three distinct results to be unpacked. The first is a
        modification of the input DataFrame containing staging values that were
        used for calculating the new t381 and t815 values, and the new t381 and
        t815 values. The second output in the tuple is the t381_calc value, and
        the third is the t815_calc value.
    """
    gate_3_history = pd.DataFrame()
    df = df_attach_metadata(df, run_id=RUN_ID, version=VERSION)

    df["alloc_non_trad"] = np.where(
        df["is_aait_trad"] == 1,
        0,
        (df["fund_weight"] / 100) / (1 - trad_target_pct)
    )

    prev_day_proj_trad = (df["is_aait_trad"] * df["prev_day_proj"]).sum()
    df["alloc_non_trad_active"] = (
        df["alloc_non_trad"] *
        (df["prev_day_proj"].sum() - prev_day_proj_trad)
    )

    df["total_non_trad_alloc_unsellable"] = np.where(
        df["is_aait_trad"] == 1,
        0,
        df["alloc_non_trad_active"] + df["unsellable"]
    )

    df["check_101_pct"] = np.where(
        df["is_aait_trad"] == 1,
        0,
        (df["total_non_trad_alloc_unsellable"] / df["target_cash"])
    )
    df["check_101_pct"] = df["check_101_pct"].fillna(0)
    df["check_101"] = check_greater_than_101(df, "check_101_pct")

    if round(trad_active_pct, 4) > round_down(trad_active_pct, 2):
        trad_new_pct = round_up(trad_active_pct, 2)
    else:
        trad_new_pct = round_down(trad_active_pct, 4)

    df["trad_new_pct"] = trad_new_pct

    df["gross_down_alloc_non_trad"] = np.where(
        df["is_aait_trad"] == 1,
        0,
        df["alloc_non_trad"] * (1 - trad_new_pct)
    )
    df["gross_down_alloc_non_trad"] = df["gross_down_alloc_non_trad"].fillna(0)
    df["replace_new_trad_pct"] = np.where(
        df["is_aait_trad"] == 1,
        trad_new_pct,
        df["gross_down_alloc_non_trad"]
    )
    df["replace_new_trad_pct"] = df["replace_new_trad_pct"].fillna(0)
    df["apply_rounding_all_n"] = np.where(
        df["is_aait_trad"] == 1,
        trad_new_pct,
        round(df["replace_new_trad_pct"], 2)
    )
    df["apply_rounding_all_n"] = df["apply_rounding_all_n"].fillna(0)
    df["apply_rounding_all_n"]=df["apply_rounding_all_n"].apply(lambda x: round(x,10))

    df["making_whole_100"] = apply_rounding(df, "apply_rounding_all_n")
    df["iteration"] = 0
    df["unsellable_target_cash"] = 0
    df["refreshed_alloc_after_test"] = 0
    df["check_101_fund_weight"] = 0
    df["check_101_less_than"] = 0
    df["check_prev_iter_not_considered"] = 0
    df["prev_iter_not_considered_fund_weight"] = 0
    df["prorata_weights_less_101"] = 0
    df["inflation_factor"] = 0
    df["refresh_alloc_inflation_factor"] = 0
    df["apply_rounding_found_y"] = 0
    df["prev_over_alloc"] = df['over_allocation']
    df["t381_calc"] = np.where(
        df["fund_weight"] > 0,
        df["making_whole_100"],
        0
    )
    df["t381_calc"] = df["t381_calc"].apply(lambda x: round(x, 2))
    df["remove_trad"] = np.where(
        df["is_aait_trad"] == 1,
        0,
        round(((df["fund_weight"] / 100) / (1 - trad_target_pct)), 2)
    )
    df["t815_calc"] = apply_rounding(df, "remove_trad", t815_making_whole_100_pct=True)
    df["t815_calc"] = df["t815_calc"].apply(lambda x: round(x, 2))

    model_weight_cnt = (np.where(df['fund_weight'] > 0, 1, 0)).sum()
    # if (np.where(df['fund_weight'] > 0, 1, 0) * df['is_aait_trad']).sum():
    #     if (df['over_allocation'] * df['is_aait_trad']).sum():
    #         prev_overalloc_cnt = (df['over_allocation']).sum()
    #     else:
    #         prev_overalloc_cnt = (df['over_allocation'] + df['is_aait_trad']).sum()
    # else:
    prev_overalloc_cnt = (df['over_allocation'] + df['check_101']).sum()

    # if prev_overalloc_cnt >= model_weight_cnt and df['check_101'].sum() != 0:
    #     df['t381_from_active_pct'] = np.where(
    #         df['is_aait_trad'] == 1, round_up(trad_active_pct, 2), round(df['active_pct'], 2)
    #     )
    #     df['t381_calc_updated'] = apply_rounding(df, "t381_from_active_pct")
    #     df['overalloc_excedes_active_ticker'] = 1
    #     enter_loop = False
    # else:
    #     df['t381_from_active_pct'] = 0
    #     df['t381_calc_updated'] = df["t381_calc"]
    #     df['overalloc_excedes_active_ticker'] = 0
    #     enter_loop = True
    
    df['t381_calc_updated'] = df["t381_calc"]
    df['overalloc_excedes_active_ticker'] = 0

    df["active_bal_calc"] = 0
    df["legacy_bal_calc"] = 0
    df["total_bal_calc"] = 0
    df["funds_consider_next_iter"] = 0
    df["next_iter_check_101_pct"] = 0
    df["next_iter_check_fund_101_pct"] = 0

    df = df_replace_nan_and_inf(df)
    gate_3_history = pd.concat([gate_3_history, df])

    if df["check_101"].sum() != 0:
        i = 1
        while df["check_101"].sum() != 0:
                        
            df["gross_down_alloc_non_trad"] = 0
            df["replace_new_trad_pct"] = 0
            df["apply_rounding_all_n"] = 0
            df["making_whole_100"] = 0       
                        
            df["iteration"] = i
            df["unsellable_target_cash"] = np.where(
                df["check_101"] == 0, 0, df["target_cash"] - df["unsellable"]
            )
            
            if i == 1:
                df["refreshed_alloc_after_test"] = np.where(
                    df["is_aait_trad"] == 1,
                    trad_new_pct,
                    np.where(
                        df["check_101"] == 1,
                        df["unsellable_target_cash"] / df["prev_day_proj"].sum(),
                        df["alloc_non_trad"],
                    ),
                )
            else:
                df["refreshed_alloc_after_test"] = np.where(
                    df["is_aait_trad"] == 1,
                    trad_new_pct,
                    np.where(
                        df["check_101"] == 1,
                        df["unsellable_target_cash"] / df["prev_day_proj"].sum(),
                        df["t381_calc"]
                    )
                )

            check_101_fund_weight = (
                df["check_101"] * df["fund_weight"] / 100
            ).sum()
            df["check_101_fund_weight"] = check_101_fund_weight

            if i == 1:
                df["check_101_less_than"] = check_less_than_101(df, "check_101_pct")
                df["check_prev_iter_not_considered"] = 0
                df["prev_iter_not_considered_fund_weight"] = 0
                df["prorata_weights_less_101"] = np.where(
                    df["is_aait_trad"] == 1,
                    0,
                    np.where(
                        df["check_101_less_than"] == 1,
                        (
                            (df["fund_weight"] / 100) /
                            (1 - trad_target_pct - check_101_fund_weight)
                        ),
                        0,
                    ),
                )
            else:
                df["check_101_less_than"] = check_less_than_101(df, "next_iter_check_101_pct")
                df["check_prev_iter_not_considered"] = np.where(
                    (
                        (df["is_aait_trad"] != 1) &
                        (df["funds_consider_next_iter"] == 0)
                    ),
                    1,
                    0,
                )
                prev_iter_not_considered_fund_weight = (
                    df["check_prev_iter_not_considered"] *
                    df["fund_weight"] /
                    100
                ).sum()
                df["prev_iter_not_considered_fund_weight"] = prev_iter_not_considered_fund_weight

                df["prorata_weights_less_101"] = np.where(
                    df["is_aait_trad"] == 1,
                    0,
                    np.where(
                        (
                            (df["check_101_less_than"] == 1) &
                            (df["funds_consider_next_iter"] == 1) &
                            (df["fund_weight"] > 0)
                        ),
                        (
                            (df["fund_weight"] / 100) /
                            (
                                1 -
                                trad_target_pct -
                                check_101_fund_weight -
                                prev_iter_not_considered_fund_weight
                            )
                        ),
                        0,
                    ),
                )

            df["inflation_factor"] = (
                df["prorata_weights_less_101"] *
                (1 - df["refreshed_alloc_after_test"].sum())
            )
            df["refresh_alloc_inflation_factor"] = np.where(
                df["inflation_factor"] == 0,
                df["refreshed_alloc_after_test"],
                df["refreshed_alloc_after_test"] + df["inflation_factor"]
            )

            neg_t381_ind = np.where(round(df['refresh_alloc_inflation_factor'], 8) < 0, 1, 0).sum()
            if neg_t381_ind > 0:
                df["apply_rounding_found_y"] = np.where(
                    df["is_aait_trad"] == 1,
                    trad_new_pct,
                    np.where(
                        df['refresh_alloc_inflation_factor'] < 0,
                        0,
                        round(df["refresh_alloc_inflation_factor"], 2)
                    )     
                )
            else:
                df["apply_rounding_found_y"] = np.where(
                    df["is_aait_trad"] == 1,
                    trad_new_pct,
                    round(df["refresh_alloc_inflation_factor"], 2)
                )

            if i == 1:
                df["prev_over_alloc"] = df["check_101"] + df['over_allocation']
            else:
                df["prev_over_alloc"] = (
                    df["check_101"] +
                    gate_3_history.loc[
                        gate_3_history["iteration"] == (i-1),
                        "prev_over_alloc"
                    ]
                )
            
            df["apply_rounding_found_y"] = df["apply_rounding_found_y"].apply(lambda x: round(x,10))
            df["t381_calc"] = apply_rounding(df, "apply_rounding_found_y", neg_t381_ind=neg_t381_ind)
            df["t381_calc"] = df["t381_calc"].apply(lambda x: round(x, 2))
            df["remove_trad"] = np.where(
                df["is_aait_trad"] == 1,
                0,
                round(((df["fund_weight"] / 100) / (1 - trad_target_pct)), 2)
            )
            df["t815_calc"] = apply_rounding(df, "remove_trad", t815_making_whole_100_pct=True)
            df["t815_calc"] = df["t815_calc"].apply(lambda x: round(x, 2))
            
            model_weight_cnt = (np.where(df['fund_weight'] > 0, 1, 0)).sum()
            if (np.where(df['fund_weight'] > 0, 1, 0) * df['is_aait_trad']).sum():
                if (df['prev_over_alloc'] * df['is_aait_trad']).sum():
                    prev_overalloc_cnt = (df['prev_over_alloc']).sum()
                else:
                    prev_overalloc_cnt = (df['prev_over_alloc'] + df['is_aait_trad']).sum()
            else:
                prev_overalloc_cnt = (df['prev_over_alloc']).sum()
            
            if prev_overalloc_cnt >= model_weight_cnt and i > 1:
                df['t381_calc_updated'] = gate_3_history.loc[
                    gate_3_history["iteration"] == (i-1),
                    "t381_calc"
                ]
                df['overalloc_excedes_active_ticker'] = 1
            else:
                df['t381_calc_updated'] = df['t381_calc']
                df['overalloc_excedes_active_ticker'] = 0

            df["active_bal_calc"] = df["prev_day_proj"].sum() * df["t381_calc_updated"]
            df["legacy_bal_calc"] = np.where(
                df["is_aait_trad"] == 1,
                df["unsellable"] - df["prev_day_proj"],
                df["unsellable"],
            )
            df["total_bal_calc"] = df["legacy_bal_calc"] + df["active_bal_calc"]

            if i == 1:
                df["funds_consider_next_iter"] = np.where(
                    (df["is_aait_trad"] == 1) | (df["check_101"] == 1),
                    0,
                    1,
                )
            else:
                df["funds_consider_next_iter"] = np.where(
                    (
                        (df["is_aait_trad"] == 1) |
                        (df["check_101"] == 1) |
                        (df["funds_consider_next_iter"] == 0)
                    ),
                    0,
                    1
                )
            
            df["next_iter_check_101_pct"] = (
                df["total_bal_calc"] /
                df["target_cash"]
            )
            df["next_iter_check_101_pct"] = df["next_iter_check_101_pct"].fillna(0)
            df["next_iter_check_fund_101_pct"] = (
                df["next_iter_check_101_pct"] *
                df["funds_consider_next_iter"]
            )
            df["next_iter_check_fund_101_pct"] = df["next_iter_check_fund_101_pct"].fillna(0)

            df = df_replace_nan_and_inf(df)
            gate_3_history = pd.concat([gate_3_history, df])
            
            df["check_101"] = check_greater_than_101(df, "next_iter_check_fund_101_pct")
            i += 1

            if prev_overalloc_cnt >= model_weight_cnt or neg_t381_ind > 0:
                break

    return gate_3_history, df["t381_calc_updated"].values, df["t815_calc"].values

def calc_gate_4(df, trad_active_pct, trad_target_pct):
    """Rebalance when TEMP Trad NOT overallocated and Legacy IS overallocated.

    This function represents Gate 4 - the fourth of eight logic gates designed to
    handle different combinations of asset allocation and legacy balances. Gate
    2 is triggered when the pin-plan-subplan has TEMP Traditional assets NOT
    overallocated, but legacy assets ARE overallocated.
    
    The purpose of each logic gate is to return a live calculation of the
    pin-plan-subplan's t381 and t815 weight distributions. These functions are
    direct implementations from the models built in excel by the LEC Actuary
    team.

    Args:
        df: A table representing the target pin-plan-subplan's asset allocations
          on the day of rebalancing.
        trad_active_pct: The current percentage of traditional assets
          representing the target pin-plan-subplan's full portfolio.
        trad_target_pct: The desired percentage of traditional assets
          representing the target pin-plan-subplan's full portfolio after the
          rebalancing has occured.
    
    Returns:
        A tuple of three distinct results to be unpacked. The first is a
        modification of the input DataFrame containing staging values that were
        used for calculating the new t381 and t815 values, and the new t381 and
        t815 values. The second output in the tuple is the t381_calc value, and
        the third is the t815_calc value.
    """
    gate_4_history =  pd.DataFrame()
    df = df_attach_metadata(df, run_id=RUN_ID, version=VERSION)

    # Round the trad % for active assets with a manual decimal rounding system.
    # NOTE: This can be replaced with the decimal_round function.
    if round(trad_active_pct, 4) > round_down(trad_active_pct, 2):
        trad_new_pct = round_up(trad_active_pct, 2)
    else:
        trad_new_pct = round_down(trad_active_pct, 4)

    df["trad_new_pct"] = trad_new_pct
    df["x_case_fund_pct"] = 0

    total_overallocated_fund_weight = (
        df["over_allocation"] * (df["fund_weight"] / 100)
    ).sum()

    df["total_overallocated_fund_weight"] = total_overallocated_fund_weight
    
    df["revised_fund_weight_exclude_overalloc"] = np.where(
        df["over_allocation"] == 1,
        0,
        (df["fund_weight"] / 100) / (1 - total_overallocated_fund_weight),
    )

    trad_revised_target_pct = (
        df["is_aait_trad"] * df["revised_fund_weight_exclude_overalloc"]
    ).sum()
    df["trad_revised_target_pct"] = trad_revised_target_pct

    df["revised_active_target_cash"] = (
        df["prev_day_proj"].sum() *
        df["revised_fund_weight_exclude_overalloc"]
    )
    df["legacy_bal_calc"] = df ['raw_legacy_balance'] 
    df["total_bal_calc"] = df["legacy_bal_calc"] + df["revised_active_target_cash"]
    df["check_101_pct"] = df["total_bal_calc"] / df["target_cash"]
    df["check_101_pct"].fillna(0)
    df["check_101_temp"] = check_greater_than_101(df, "check_101_pct")
    df["check_101"] = np.where(df["over_allocation"] == 1, 0, df["check_101_temp"])
    
    if round(trad_revised_target_pct, 4) > round_down(trad_revised_target_pct, 2):
        new_alloc_trad_pct = round_up(trad_revised_target_pct, 2)
    else:
        new_alloc_trad_pct = round_down(trad_revised_target_pct, 4)
    
    df["new_alloc_round_up_pct"] = 0
    df["new_alloc_trad_pct"] = new_alloc_trad_pct
    df["new_alloc_pct"] = np.where(
        df["is_aait_trad"] == 1,
        df["new_alloc_trad_pct"],
        round(
            (df["revised_fund_weight_exclude_overalloc"] / (1 - trad_revised_target_pct)) * (1 - new_alloc_trad_pct)
            ,2
        )
    )
    df["iteration"] = 0
    df["target_sub_legacy_cash"] = 0
    df["refreshed_alloc"] = 0
    df["prev_over_alloc"] = df['over_allocation']
    df["prev_over_alloc_fund_weight"] = 0
    df["prorata_weights_less_101"] = 0
    df["inflating_factor"] = 0
    df["refresh_alloc_inflating_factor"] = 0
    df["refresh_alloc_inflating_factor_roundup"] = 0
    df["apply_rounding"] = 0
    df["apply_rounding"] = df["apply_rounding"].fillna(0)
    df["new_alloc_pct"]=df["new_alloc_pct"].apply(lambda x :round(x,10))
    df["t381_calc"] = apply_rounding(df, "new_alloc_pct")
    #df["t381_calc"] = df["t381_calc"].apply(lambda x: round(x, 2))
    df["remove_overallocated_fund"] = np.where(
        df["over_allocation"] == 1,
        0,
        round((df["fund_weight"] / 100) / (1 - total_overallocated_fund_weight), 2)
    )
    df["t815_calc"] = apply_rounding(df, "remove_overallocated_fund", t815_making_whole_100_pct=True)
    df["t815_calc"] = df["t815_calc"].apply(lambda x: round(x, 2))
    
    model_weight_cnt = (np.where(df['fund_weight'] > 0, 1, 0)).sum()
    if (np.where(df['fund_weight'] > 0, 1, 0) * df['is_aait_trad']).sum():
        if (df['check_101'] * df['is_aait_trad']).sum():
            prev_overalloc_cnt = (df['over_allocation'] + df['check_101']).sum()
        else:
            prev_overalloc_cnt = (df['over_allocation'] + df['check_101'] + df['is_aait_trad']).sum()
    else:
        prev_overalloc_cnt = (df['over_allocation'] + df['check_101']).sum()

    # if prev_overalloc_cnt >= model_weight_cnt and df['check_101'].sum() != 0:
    #     df['t381_from_active_pct'] = np.where(
    #         df['is_aait_trad'] == 1, round_up(trad_active_pct, 2), round(df['active_pct'], 2)
    #     )
    #     df['t381_calc_updated'] = df["t381_calc"]
    #     df['overalloc_excedes_active_ticker'] = 1
    #     enter_loop = False
    # else:
    #     df['t381_from_active_pct'] = 0
    #     df['t381_calc_updated'] = df["t381_calc"]
    #     df['overalloc_excedes_active_ticker'] = 0
    #     enter_loop = True
    
    df['t381_calc_updated'] = df["t381_calc"]
    df['overalloc_excedes_active_ticker'] = 0

    df["remove_overallocated_fund"] = df["remove_overallocated_fund"].fillna(0)
    df["active_bal_recalc"] = 0
    df["legacy_bal_recalc"] = 0
    df["total_bal_recalc"] = 0
    df["total_bal_recalc"] = df["total_bal_recalc"].fillna(0)
    df["next_iter_check_101_pct"] = 0
    df["next_iter_check_fund_101_pct"] = 0

    df["replace_max_trad_pct_t381_trad_floor_adj"] = 0
    df["prorata_weights_non_trad_trad_floor_adj"] = 0
    df["inflating_factor_trad_floor_adj"] = 0
    df["refreshed_alloc_after_inflating_factor_trad_floor_adj"] = 0
    df["apply_rounding_trad_floor_adj"] = 0
    df["final_t381"] = 0

    df = df_replace_nan_and_inf(df)
    gate_4_history = pd.concat([gate_4_history, df])

    i = 1
    if df["check_101"].sum() != 0:
        while df["check_101"].sum() != 0:
            df["new_alloc_round_up_pct"] = 0
            df["new_alloc_pct"] = 0
            df["iteration"] = i   
            df["target_sub_legacy_cash"] = df["check_101"] * (
                df["target_cash"] - df["legacy_bal_calc"]
            )
            
            if i == 1:
                df["refreshed_alloc"] = np.where(
                    df["check_101"] == 1,
                    df["target_sub_legacy_cash"] / df["prev_day_proj"].sum(),
                    df["revised_fund_weight_exclude_overalloc"]
                )
                df["prev_over_alloc"] = np.where(
                    (df["check_101"] == 1) | (df["over_allocation"] == 1),
                    1,
                    0
                )
            else:
                df["refreshed_alloc"] = np.where(
                    df["check_101"] == 1,
                    df["target_sub_legacy_cash"] / df["prev_day_proj"].sum(),
                    df["t381_calc"]
                )
                df["prev_over_alloc"] = np.where(
                    (df["check_101"] == 1) | 
                    (
                        gate_4_history.loc[
                            gate_4_history["iteration"] == (i-1),
                            "prev_over_alloc"
                        ] == 1
                    ),
                    1,
                    0
                )
            
            prev_over_alloc_fund_weight = (
                df["prev_over_alloc"] * (df["fund_weight"] / 100)
            ).sum()
            df["prev_over_alloc_fund_weight"] = prev_over_alloc_fund_weight

            df["prorata_weights_less_101"] = np.where(
                df["prev_over_alloc"] >= 1,
                0,
                (df["fund_weight"] / 100) / (1 - prev_over_alloc_fund_weight)
            )
            df["inflating_factor"] = (
                df["prorata_weights_less_101"] *
                (1 - df["refreshed_alloc"].sum())
            )
            df["refresh_alloc_inflating_factor"] = (
                df["inflating_factor"] +
                df["refreshed_alloc"]
            )
            df["refresh_alloc_inflating_factor_roundup"] = [
                round_up(x, 2) for x in df["refresh_alloc_inflating_factor"]
            ]

            neg_t381_ind = np.where(round(df['refresh_alloc_inflating_factor'], 8) < 0, 1, 0).sum()
            if neg_t381_ind > 0:
                df["apply_rounding"] = np.where(
                    df["is_aait_trad"] == 1,
                    df["refresh_alloc_inflating_factor_roundup"],
                    np.where(
                        df['refresh_alloc_inflating_factor'] < 0,
                        0,
                        round(df["refresh_alloc_inflating_factor"], 2)
                    )     
                )
            else:
                df["apply_rounding"] = np.where(
                    df["is_aait_trad"] == 1,
                    df["refresh_alloc_inflating_factor_roundup"],
                    round(df["refresh_alloc_inflating_factor"], 2)
                )
            
            df["apply_rounding"]=df["apply_rounding"].apply(lambda x: round(x,10))
            df["t381_calc"] = apply_rounding(df, "apply_rounding", neg_t381_ind=neg_t381_ind)
            df["t381_calc"] = df["t381_calc"].apply(lambda x: round(x, 2))
            
            df["remove_overallocated_fund"] = np.where(
                df["over_allocation"] == 1,
                0,
                round((df["fund_weight"] / 100) / (1 - total_overallocated_fund_weight), 2)
            )
            df["t815_calc"] = apply_rounding(df, "remove_overallocated_fund", t815_making_whole_100_pct=True)
            df["t815_calc"] = df["t815_calc"].apply(lambda x: round(x, 2))

            model_weight_cnt = (np.where(df['fund_weight'] > 0, 1, 0)).sum()
            if (np.where(df['fund_weight'] > 0, 1, 0) * df['is_aait_trad']).sum():
                if (df['prev_over_alloc'] * df['is_aait_trad']).sum():
                    prev_overalloc_cnt = (df['prev_over_alloc']).sum()
                else:
                    prev_overalloc_cnt = (df['prev_over_alloc'] + df['is_aait_trad']).sum()
            else:
                prev_overalloc_cnt = (df['prev_over_alloc']).sum()
            
            if prev_overalloc_cnt >= model_weight_cnt and i > 1:
                df['t381_calc_updated'] = gate_4_history.loc[
                    gate_4_history["iteration"] == (i-1),
                    "t381_calc"
                ]
                df['overalloc_excedes_active_ticker'] = 1
            else:
                df['t381_calc_updated'] = df['t381_calc']
                df['overalloc_excedes_active_ticker'] = 0

            df["active_bal_recalc"] = df["prev_day_proj"].sum() * df["t381_calc_updated"]
            df["legacy_bal_recalc"] = df["legacy_bal_calc"]
            df["total_bal_recalc"] = df["legacy_bal_recalc"] + df["active_bal_recalc"]
            df["next_iter_check_101_pct"] = df["total_bal_recalc"] / df["target_cash"]   
            df["next_iter_check_101_pct"] = [
                0 if math.isinf(x) else x
                for x in df["next_iter_check_101_pct"].fillna(0)
            ]
            df["next_iter_check_fund_101_pct"] = (
                df["next_iter_check_101_pct"] *
                np.where(
                    df["prev_over_alloc"] == 0,
                    1,
                    0
                )
            )

            df = df_replace_nan_and_inf(df)
            gate_4_history = pd.concat([gate_4_history, df])

            df["check_101"] = check_greater_than_101(df, "next_iter_check_fund_101_pct")
            df["check_101"].fillna(0)
            i += 1

            if prev_overalloc_cnt >= model_weight_cnt or neg_t381_ind > 0:
                break
    
    # TRAD FLOOR ADJUSTMENT LOGIC
    df["iteration"] = i
    replace_max_trad_pct_t381 = max(
        round_up(trad_active_pct, 2),
        (df['is_aait_trad'] * df["t381_calc_updated"]).sum()
    )
    df['replace_max_trad_pct_t381_trad_floor_adj'] = np.where(
        df['is_aait_trad'] == 1,
        replace_max_trad_pct_t381,
        df["t381_calc_updated"]
    )
    df['prorata_weights_non_trad_trad_floor_adj'] = np.where(
        df['is_aait_trad'] == 1,
        0,
        safe_divide(df['replace_max_trad_pct_t381_trad_floor_adj'], (1 - (df['is_aait_trad'] * df["t381_calc_updated"]).sum()))
    )
    df['inflating_factor_trad_floor_adj'] = df['prorata_weights_non_trad_trad_floor_adj'] * (1 - df['replace_max_trad_pct_t381_trad_floor_adj'].sum())
    df['refreshed_alloc_after_inflating_factor_trad_floor_adj'] = df['replace_max_trad_pct_t381_trad_floor_adj'] + df['inflating_factor_trad_floor_adj']
    df['apply_rounding_trad_floor_adj'] = np.where(
        df['is_aait_trad'] == 1,
        df['refreshed_alloc_after_inflating_factor_trad_floor_adj'].apply(lambda x:round_up(x, 2)),
        round(df['refreshed_alloc_after_inflating_factor_trad_floor_adj'], 2)
    )
    df['final_t381'] = apply_rounding(df, 'apply_rounding_trad_floor_adj', trad_floor_adj=True)

    df = df_replace_nan_and_inf(df)
    gate_4_history = pd.concat([gate_4_history, df])

    return gate_4_history, df["final_t381"].values, df["t815_calc"].values

def calc_gate_5(df):
    """Rebalance when TEMP Trad is NOT overallocated and legacy balance > 0.

    This function represents Gate 5 - the fifth of eight logic gates designed to
    handle different combinations of asset allocation and legacy balances. Gate
    2 is triggered when the pin-plan-subplan has TEMP Traditional assets NOT
    overallocated, but legacy assets are greater than zero.
    
    The purpose of each logic gate is to return a live calculation of the
    pin-plan-subplan's t381 and t815 weight distributions. These functions are
    direct implementations from the models built in excel by the LEC Actuary
    team.

    Args:
        df: A table representing the target pin-plan-subplan's asset allocations
          on the day of rebalancing.
    
    Returns:
        A tuple of three distinct results to be unpacked. The first is a
        modification of the input DataFrame containing staging values that were
        used for calculating the new t381 and t815 values, and the new t381 and
        t815 values. The second output in the tuple is the t381_calc value, and
        the third is the t815_calc value.
    """
    gate_5_history = pd.DataFrame()
    df = df_attach_metadata(df, run_id=RUN_ID, version=VERSION)

    df["target_sub_legacy"] = np.where(
        df["fund_weight"] > 0, df["target_cash"] - df["legacy_balance"], 0
    )
    df["alloc_adjust_legacy"] = df["target_sub_legacy"] / df["prev_day_proj"].sum()

    aait_trad_adjust_pct = (df["alloc_adjust_legacy"] * df["is_aait_trad"]).sum()
    df["aait_trad_adjust_pct"] = aait_trad_adjust_pct

    if round(aait_trad_adjust_pct, 4) > round_down(aait_trad_adjust_pct, 2):
            trad_new_pct = round_up(aait_trad_adjust_pct, 2)
    else:
        trad_new_pct = round_down(aait_trad_adjust_pct, 4)

    df["trad_new_pct"] = trad_new_pct

    df["replace_aait_trad_round"] = np.where(
        df["is_aait_trad"] == 1,
        trad_new_pct,
        round(df["alloc_adjust_legacy"], 2)
    )
    df["replace_aait_trad_round"]=df["replace_aait_trad_round"].apply(lambda x:round(x,10))
    df["t381_calc"] = apply_rounding(df, "replace_aait_trad_round")
    df["t381_calc"] = df["t381_calc"].apply(lambda x: round(x, 2))
    df["t815_calc"] = round(df["fund_weight"] / 100, 2)

    df = df_replace_nan_and_inf(df)
    gate_5_history = pd.concat([gate_5_history, df])

    return gate_5_history, df["t381_calc"].values, df["t815_calc"].values

def calc_gate_6(df, trad_active_pct, trad_target_pct):
    """Rebalance when TEMP Trad NOT overallocated and Legacy IS overallocated.

    This function represents Gate 6 - the sixth of the logic gates designed to
    handle different combinations of asset allocation and legacy balances. Gate
    6 is triggered when the pin-plan-subplan has both TEMP Traditional assets
    and legacy assets as overallocated.
    
    The purpose of each logic gate is to return a live calculation of the
    pin-plan-subplan's t381 and t815 weight distributions. These functions are
    direct implementations from the models built in excel by the LEC Actuary
    team.

    Args:
        df: A table representing the target pin-plan-subplan's asset allocations
          on the day of rebalancing.
        trad_active_pct: The current percentage of traditional assets
          representing the target pin-plan-subplan's full portfolio.
        trad_target_pct: The desired percentage of traditional assets
          representing the target pin-plan-subplan's full portfolio after the
          rebalancing has occured.
    
    Returns:
        A tuple of three distinct results to be unpacked. The first is a
        modification of the input DataFrame containing staging values that were
        used for calculating the new t381 and t815 values, and the new t381 and
        t815 values. The second output in the tuple is the t381_calc value, and
        the third is the t815_calc value.
    """
    gate_6_history =  pd.DataFrame()
    df = df_attach_metadata(df, run_id=RUN_ID, version=VERSION)

    # Round the trad % for active assets with a manual decimal rounding system.
    # NOTE: This can be replaced with the decimal_round function.
    if round(trad_active_pct, 4) > round_down(trad_active_pct, 2):
        trad_new_pct = round_up(trad_active_pct, 2)
    else:
        trad_new_pct = round_down(trad_active_pct, 4)

    total_overallocated_fund_weight = (
        df["over_allocation"] *
        (df["fund_weight"] / 100)
    ).sum()
    df["total_overallocated_fund_weight"] = total_overallocated_fund_weight

    df["revised_fund_weight_exclude_overalloc"] = np.where(
        df["over_allocation"] == 1,
        0,
        (df["fund_weight"] / 100) / (1 - total_overallocated_fund_weight)
    )
    df["revised_active_target_cash"] = (
        (
            df["prev_day_proj"].sum() -
            (df["is_aait_trad"] * df["prev_day_proj"]).sum()
        ) *
        df["revised_fund_weight_exclude_overalloc"]
    )
    df["legacy_bal_calc"] = df["raw_legacy_balance"]
    df["total_bal_calc"] = df["legacy_bal_calc"] + df["revised_active_target_cash"]
    df["check_101_pct"] = df["total_bal_calc"] / df["target_cash"]
    df["check_101_pct"] = df["check_101_pct"].fillna(0)
    df["check_101_temp"] = check_greater_than_101(df, "check_101_pct")
    df["check_101"] = np.where(df["over_allocation"] == 1, 0, df["check_101_temp"])
    df["new_alloc_round_up_pct"] = 0
    df["new_alloc_pct"] = np.where(
        df["is_aait_trad"] == 1,
        trad_new_pct,
        round(df["revised_fund_weight_exclude_overalloc"]*(1-trad_new_pct), 2)
        )
    df["iteration"] = 0
    df["target_sub_legacy_cash"] = 0
    df["refreshed_alloc"] = 0
    df["prev_over_alloc"] = df['over_allocation']
    df["prev_over_alloc_fund_weight"] = 0
    df["prorata_weights_less_101"] = 0
    df["inflating_factor"] = 0
    df["refresh_alloc_inflating_factor"] = 0
    df["refresh_alloc_inflating_factor_roundup"] = 0
    df["apply_rounding"] = 0
    df["new_alloc_pct"] = df["new_alloc_pct"].apply(lambda x: round(x,10))
    df["t381_calc"] = apply_rounding(df, "new_alloc_pct")
    df["t381_calc"] = df["t381_calc"].apply(lambda x: round(x, 2)).fillna(0)
    df["remove_overallocated_fund"] = np.where(
        df["over_allocation"] == 1,
        0,
        round((df["fund_weight"] / 100) / (1 - total_overallocated_fund_weight), 2)
    )
    df["t815_calc"] = apply_rounding(df, "remove_overallocated_fund", t815_making_whole_100_pct=True)
    df["t815_calc"] = df["t815_calc"].apply(lambda x: round(x, 2))

    model_weight_cnt = (np.where(df['fund_weight'] > 0, 1, 0)).sum()
    # if (np.where(df['fund_weight'] > 0, 1, 0) * df['is_aait_trad']).sum():
    #     if (df['over_allocation'] * df['is_aait_trad']).sum():
    #         prev_overalloc_cnt = (df['over_allocation']).sum()
    #     else:
    #         prev_overalloc_cnt = (df['over_allocation'] + df['is_aait_trad']).sum()
    # else:
    prev_overalloc_cnt = (df['over_allocation'] + df['check_101']).sum()

    # if prev_overalloc_cnt >= model_weight_cnt and df['check_101'].sum() != 0:
    #     df['t381_from_active_pct'] = np.where(
    #         df['is_aait_trad'] == 1, round_up(trad_active_pct, 2), round(df['active_pct'], 2)
    #     )
    #     df['t381_calc_updated'] = apply_rounding(df, "t381_from_active_pct")
    #     df['overalloc_excedes_active_ticker'] = 1
    #     enter_loop = False
    # else:
    #     df['t381_from_active_pct'] = 0
    #     df['t381_calc_updated'] = df["t381_calc"]
    #     df['overalloc_excedes_active_ticker'] = 0
    #     enter_loop = True

    df['t381_calc_updated'] = df["t381_calc"]
    df['overalloc_excedes_active_ticker'] = 0
    
    df["active_bal_recalc"] = 0
    df["active_bal_recalc"] = df["active_bal_recalc"].fillna(0)
    df["legacy_bal_recalc"] = 0
    df["total_bal_recalc"] = 0
    df["total_bal_recalc"] = df["total_bal_recalc"].fillna(0)
    df["next_iter_check_101_pct"] = 0
    df["next_iter_check_fund_101_pct"] = 0

    df = df_replace_nan_and_inf(df)
    gate_6_history = pd.concat([gate_6_history, df])

    if df["check_101"].sum() != 0:
        i = 1
        while df["check_101"].sum() != 0:
            
            df["new_alloc_round_up_pct"] = 0
            df["new_alloc_pct"] = 0
        
            df["iteration"] = i   
            df["target_sub_legacy_cash"] = (
                df["check_101"] *
                (df["target_cash"] - df["legacy_bal_calc"])
            )
            
            if i == 1:
                df["refreshed_alloc"]= np.where(
                    df['is_aait_trad']==1,
                    trad_new_pct,
                    np.where(
                        df["check_101"] == 1,
                        df["target_sub_legacy_cash"] / df["prev_day_proj"].sum(),
                        df["revised_fund_weight_exclude_overalloc"]
                        )
                    ) 
                df["prev_over_alloc"] = np.where(
                    (df["check_101"] == 1) | (df["over_allocation"] == 1),
                    1,
                    0
                )
            else:
                df["refreshed_alloc"] = np.where(
                    df["check_101"] == 1,
                    df["target_sub_legacy_cash"] / df["prev_day_proj"].sum(),
                    df["t381_calc"]
                )
                    
                df["prev_over_alloc"] = np.where(
                    (df["check_101"] == 1) | 
                    (
                        gate_6_history.loc[
                            gate_6_history["iteration"] == (i-1),
                            "prev_over_alloc"
                        ] == 1
                    ),
                    1,
                    0
                )
            
            #df["prev_over_alloc"] = np.where((df["check_101"]==1) | (df["over_allocation"]==1), 1, 0)
            prev_over_alloc_fund_weight = (
                df["prev_over_alloc"] * (df["fund_weight"] / 100)
            ).sum()
            df["prev_over_alloc_fund_weight"] = prev_over_alloc_fund_weight

            df["prorata_weights_less_101"] = np.where(
                df["prev_over_alloc"] == 1,
                0,
                (df["fund_weight"] / 100) / (1 - prev_over_alloc_fund_weight)
            )
            df["prorata_weights_less_101"] = df["prorata_weights_less_101"].fillna(0)
            df["inflating_factor"] = (
                df["prorata_weights_less_101"] *
                (1 - df["refreshed_alloc"].sum())
            )
            df["inflating_factor"] = df["inflating_factor"].fillna(0)
            df["refresh_alloc_inflating_factor"] = (
                df["inflating_factor"] +
                df["refreshed_alloc"]
            )
            df["refresh_alloc_inflating_factor"] = df["refresh_alloc_inflating_factor"].fillna(0)
            df["refresh_alloc_inflating_factor_roundup"] = [
                round_up(x, 2) for x in df["refresh_alloc_inflating_factor"]
            ]
            df["refresh_alloc_inflating_factor"] = df["refresh_alloc_inflating_factor"].fillna(0)

            neg_t381_ind = np.where(round(df['refresh_alloc_inflating_factor'], 8) < 0, 1, 0).sum()
            if neg_t381_ind > 0:
                df["apply_rounding"] = np.where(
                    df["is_aait_trad"] == 1,
                    df["refresh_alloc_inflating_factor_roundup"],
                    np.where(
                        df['refresh_alloc_inflating_factor'] < 0,
                        0,
                        round(df["refresh_alloc_inflating_factor"], 2)
                    )     
                )
            else:
                df["apply_rounding"] = np.where(
                    df["is_aait_trad"] == 1,
                    df["refresh_alloc_inflating_factor_roundup"],
                    round(df["refresh_alloc_inflating_factor"], 2)
                )
            
            df["apply_rounding"] = df["apply_rounding"].fillna(0)
            df["apply_rounding"] =df["apply_rounding"] .apply(lambda x: round(x,10))
            df["t381_calc"] = apply_rounding(df, "apply_rounding", neg_t381_ind=neg_t381_ind)
            df["t381_calc"] = df["t381_calc"].apply(lambda x: round(x, 2)).fillna(0)

            df["remove_overallocated_fund"] = np.where(
                df["over_allocation"] == 1,
                0,
                round((df["fund_weight"] / 100) / (1 - total_overallocated_fund_weight), 2)
            )
            df["t815_calc"] = apply_rounding(df, "remove_overallocated_fund", t815_making_whole_100_pct=True)
            df["t815_calc"] = df["t815_calc"].apply(lambda x: round(x, 2))

            model_weight_cnt = (np.where(df['fund_weight'] > 0, 1, 0)).sum()
            if (np.where(df['fund_weight'] > 0, 1, 0) * df['is_aait_trad']).sum():
                if (df['prev_over_alloc'] * df['is_aait_trad']).sum():
                    prev_overalloc_cnt = (df['prev_over_alloc']).sum()
                else:
                    prev_overalloc_cnt = (df['prev_over_alloc'] + df['is_aait_trad']).sum()
            else:
                prev_overalloc_cnt = (df['prev_over_alloc']).sum()
            
            if prev_overalloc_cnt >= model_weight_cnt and i > 1:
                df['t381_calc_updated'] = gate_6_history.loc[
                    gate_6_history["iteration"] == (i-1),
                    "t381_calc"
                ]
                df['overalloc_excedes_active_ticker'] = 1
            else:
                df['t381_calc_updated'] = df['t381_calc']
                df['overalloc_excedes_active_ticker'] = 0

            df["active_bal_recalc"] = df["prev_day_proj"].sum() * df["t381_calc_updated"]
            df["active_bal_recalc"] = df["active_bal_recalc"].fillna(0)
            df["legacy_bal_recalc"] = df["legacy_bal_calc"]
            df["legacy_bal_recalc"] = df["legacy_bal_recalc"].fillna(0)
            df["total_bal_recalc"] = df["legacy_bal_recalc"] + df["active_bal_recalc"]
            df["total_bal_recalc"] = df["total_bal_recalc"].fillna(0)
            df["next_iter_check_101_pct"] = df["total_bal_recalc"] / df["target_cash"]   
            df["next_iter_check_101_pct"] = [
                0 if math.isinf(x) else x
                for x in df["next_iter_check_101_pct"].fillna(0)
            ]
            df["next_iter_check_fund_101_pct"] = (
                df["next_iter_check_101_pct"] *
                np.where(
                    df["prev_over_alloc"] == 0,
                    1,
                    0
                )
            )
            
            df = df_replace_nan_and_inf(df)
            gate_6_history = pd.concat([gate_6_history, df])

            df["check_101"] = check_greater_than_101(df, "next_iter_check_fund_101_pct")
            i += 1

            if prev_overalloc_cnt >= model_weight_cnt or neg_t381_ind > 0:
                break
    
    return gate_6_history, df["t381_calc_updated"].values, df["t815_calc"].values

def calc_gate_7(df):
    """This gate returns the fund weights as the caled t381 and t815 values.

    Args:
        df: A table representing the target pin-plan-subplan's asset allocations
          on the day of rebalancing.
    
    Returns:
        A tuple of three distinct results to be unpacked. The first is a
        modification of the input DataFrame containing staging values that were
        used for calculating the new t381 and t815 values, and the new t381 and
        t815 values. The second output in the tuple is the t381_calc value, and
        the third is the t815_calc value.
    """
    gate_7_history = pd.DataFrame()
    df = df_attach_metadata(df, run_id=RUN_ID, version=VERSION)
    df['t381_calc'] = df['fund_weight'] / 100
    df['t815_calc'] = df['fund_weight'] / 100

    df = df_replace_nan_and_inf(df)
    gate_7_history = pd.concat([gate_7_history, df])

    return gate_7_history, df["t381_calc"].values, df["t815_calc"].values

def calc_gate_8(df, trad_active_pct, trad_target_pct):
    """Rebalance X-Case when TEMP Trad is overallocated and legacy Bal is Zero.

    This function represents Gate 8 - the final logic gate designed to
    handle different combinations of asset allocation and legacy balances. Gate
    8 is triggered when the pin-plan-subplan has TEMP Traditional assets
    overallocated, but Zero legacy assets.

    The purpose of each logic gate is to return a live calculation of the
    pin-plan-subplan's t381 and t815 weight distributions. These functions are
    direct implementations from the models built in excel by the LEC Actuary
    team.

    Args:
        df: A table representing the target pin-plan-subplan's asset allocations
          on the day of rebalancing.
        trad_active_pct: The current percentage of traditional assets
          representing the target pin-plan-subplan's full portfolio.
        trad_target_pct: The desired percentage of traditional assets
          representing the target pin-plan-subplan's full portfolio after the
          rebalancing has occured.
    
    Returns:
        A tuple of three distinct results to be unpacked. The first is a
        modification of the input DataFrame containing staging values that were
        used for calculating the new t381 and t815 values, and the new t381 and
        t815 values. The second output in the tuple is the t381_calc value, and
        the third is the t815_calc value.
    """
    gate_8_history = pd.DataFrame()
    
    # Round the trad % for active assets with a manual decimal rounding system.
    # NOTE: This can be replaced with the decimal_round function.
    if round(trad_active_pct, 4) > round_down(trad_active_pct, 2):
        trad_new_pct = round_up(trad_active_pct, 2)
    else:
        trad_new_pct = round_down(trad_active_pct, 4)

    # Sets all trad tickers to 0% and adjusts all other tickers on target %s.
    df["alloc_non_trad"] = np.where(
        df["is_aait_trad"] == 1,
        0,
        (df["fund_weight"] / 100) / (1 - trad_target_pct)
    )
    # Replace trad ticker allocations with the rounded %s.
    df["replace_aait_trad_new_pct"] = np.where(
        df["is_aait_trad"]==1,
        trad_new_pct,
        round(df["alloc_non_trad"] * (1 - trad_new_pct), 2)
    )
    # Get t381 values by applying rounding logic.
    df["replace_aait_trad_new_pct"]=df["replace_aait_trad_new_pct"].apply(lambda x: round(x,10))
    df["t381_calc"] = apply_rounding(df, "replace_aait_trad_new_pct")
    df["t381_calc"] = df["t381_calc"].apply(lambda x: round(x, 2))
    
    # Sum t381 %s for trad tickers
    trad_t381 = (df["t381_calc"] * df["is_aait_trad"]).sum()
    df["trad_t381"] = trad_t381

    df["remove_trad"] = round(df["alloc_non_trad"], 2)
    df["t815_calc"] = apply_rounding(df, "remove_trad")
    df["t815_calc"] = df["t815_calc"].apply(lambda x: round(x, 2))

    df = df_replace_nan_and_inf(df)
    df = df_attach_metadata(df, run_id=RUN_ID, version=VERSION)
    gate_8_history = pd.concat([gate_8_history, df])

    return gate_8_history, df["t381_calc"].values, df["t815_calc"].values

def query_t381(id, today):
    """Queries data from the stg_rebal_t381_set_all_final table.

    Args:
        id: partic_id containing a concatination of pin, plan, and sub_plan.
        today: The current date of the projection in YYYY-MM-dd format.

    Returns:
        A DataFrame containing the t381 set all data filtered on that specific
        pin-plan-subplan-date.
    """
    query = f'''
    select *, cast(concat(seq, post_num) as bigint) as iter_id
    from {SCHEMA}.stg_rebal_t381_set_all_final
    where lookup_key = '{id}'
    and trade_dt = '{today}'
    '''
    engine = sa.create_engine(CONN_URL)
    df = pd.read_sql(query, con=engine)
    df['ic220_priceid'] = df['ic220_priceid'].str.replace('#', '')
    df['key_fund_ticker'] = df['key_fund_ticker'].str.replace('#', '')
    
    return df

def query_t381_cash_trans_df(pin, plan, subplan, trade_dt):
    """Queries data from stg_t381_cash_trans_mapping.

    Args:
        pin: The participant id for the current projection.
        plan: The plan id for the current projection.
        subplan: The subplan id for the current projection.
        trade_dt: The date in chich a trade from this table occured.
    
    Returns:
        A DataFrame containing the t381 cash transaction mapping data
        filtered on that specific pin-plan-subplan-date.
    """
    query = f'''
    select *
    from {SCHEMA}.stg_t381_cash_trans_mapping
    where pin = '{pin}' and [plan] = '{plan}' and sub_plan = '{subplan}' and trade_dt = convert(date, '{trade_dt}')
    ;
    '''
    engine = sa.create_engine(CONN_URL)
    df = pd.read_sql(query, con=engine)

    return df

def safe_divide(a, b):
    """Forces safe division by preventing division by 0 or ensuring type safety.

    Args:
        a: The numerator for the division
        b: The denominator for the division
    
    Returns:
        a / b or 0 if the denominator is 0.
    """
    if b == 0:
        return 0
    
    a, b = np.nan_to_num(a), np.nan_to_num(b)
    return a / b

def calc_non_trad_bal_where_trad_0(df, baseline):
    """Calculates the non trad balance at the source level where trad bal is 0.

    The goal is to calculate the source level non-trad balance where trad
    balance for that source is 0. This is done to replicate a bug in OMNI
    for baselining. Therefore this operation is only performed when baseline is
    True.

    Args:
        df: A DataFrame containing the date_df data for sources, ic220_priceids,
          and available_balances. 
        baseline: Boolean value specifying if the run is baseline or corrected.

    Returns:
        A float indicating the total balance of all non-trad tickers with the
        same source as a trad ticker with 0 balance.
    """
    if baseline:
        df = df.groupby(["source", "ic220_priceid"]).agg('sum').reset_index()
        df['trad_balance'] = np.where(
            df["ic220_priceid"].isin(TRAD_TICKERS),
            df['available_balance'],
            None
        )
        df['non_trad_balance'] = np.where(
            df["ic220_priceid"].isin(TRAD_TICKERS),
            0,
            df['available_balance']
        )
        sources_where_trad_0 = df[df['trad_balance'] == 0]['source'].to_list()
        non_trad_bal_where_trad_0 = (np.where(
            df["source"].isin(sources_where_trad_0),
            df['non_trad_balance'],
            0
        )).sum()
    
    else:
        non_trad_bal_where_trad_0 = 0

    return non_trad_bal_where_trad_0


def calculate_two_step_rebalance(df=None, source_level=True):
	if source_level:
		cols = ["source", "ic220_priceid","pct"]		
	else:
		cols = ["ic220_priceid","pct"]
		
	df = df[cols].drop_duplicates()


def calc_key_fund_logic(t381, df_1, date_df, read_in, baseline):
    """Rebalances the participant balance based on the passed-in t381 values.

    Args:
        t381: Contains the t381 data from the stg_rebal_t381_set_all_final table
          created during the data prep process.
        df_1: Contains the separated projection data for non-trad assets. This
          is recombined into the date_df final output in the function but is
          also kept separate for debugging due to separate treatment of trad and
          non-trad assets.
        date_df: The BTR data filtered on the specific partic_id and current
          date of the projection.
        read_in: Boolean value specifying if we are reading in rebalancing data
          or calculating it live.
        baseline: Boolean value specifying if the run is baseline or corrected.
    
    Returns:
        A tuple of DataFrames to update within the projection_rebal function.
            df_1: Contains the separated projection data for non-trad assets
              with the updated balances after rebalancing.
            rebal_cash_hist: Debugging table that tracks the rebalancing cash
              history thoughout the key fund logic and cash rebalancing process.
            two_step_rebal_chk: Debugging table that tracks the data througout
              the two step rebalancing logic.
    """
    iter_id_unique = list(t381["iter_id"].unique())
    iter_id_unique.sort()
    j=0
    rebal_cash_df_hist = pd.DataFrame()
    two_step_rebal_chk = pd.DataFrame()
    illiquid_ind = date_df["bl_omni_illiquid_ind"].unique()[0]
        
    for i in iter_id_unique:
        j += 1
        t381_df = t381[t381["iter_id"]==i].drop_duplicates()
        if j == 1:
            t381_calc_df = df_1[["source","ic220_priceid","final_bal","final_bal_total"]]
            live_rebal_ind = t381_df["live_rebal_ind"].unique()[0]
            
            if read_in == False and baseline == True and live_rebal_ind == 1 and date_df['unsub_ind'].unique()[0] == 0:
                t381_calc_df = pd.merge(t381_calc_df,df_1[["ic220_priceid","t381_calc"]].drop_duplicates(), how="left", on ="ic220_priceid")
                t381_calc_df = t381_calc_df.drop_duplicates()
                t381_calc_df["t381_calc"] = t381_calc_df["t381_calc"].fillna(0)
                t381_calc_df["pct"] = t381_calc_df["t381_calc"]
                
                non_trad_bal_where_trad_0 = calc_non_trad_bal_where_trad_0(date_df[["source", "ic220_priceid", "available_balance"]], baseline)
                if non_trad_bal_where_trad_0 == 0:
                    two_step_rebal_chk = t381_calc_df[["ic220_priceid","pct"]].drop_duplicates()
                    two_step_rebal_chk.insert(0, 'source', 'AGG')
                    two_step_rebal_chk = pd.merge(two_step_rebal_chk, date_df[["ic220_priceid","available_balance"]].groupby("ic220_priceid").agg('sum').reset_index(), how='left', on ='ic220_priceid')
                    two_step_rebal_chk["non_trad_bal_where_trad_0"] = non_trad_bal_where_trad_0
                    two_step_rebal_chk["total_available_balance"] = two_step_rebal_chk["available_balance"].sum()
                    two_step_rebal_chk["ticker_dist"] = two_step_rebal_chk["available_balance"]/two_step_rebal_chk["total_available_balance"]

                    trad_dist = (np.where(two_step_rebal_chk["ic220_priceid"].isin(TRAD_TICKERS), 1, 0) * two_step_rebal_chk["ticker_dist"]).sum()
                    bad_trad_dist = trad_dist
                    
                else:
                    two_step_rebal_chk = t381_calc_df[["source","ic220_priceid","pct"]].drop_duplicates()
                    two_step_rebal_chk = pd.merge(two_step_rebal_chk, date_df[["source","ic220_priceid","available_balance"]].groupby(["source","ic220_priceid"]).agg('sum').reset_index(), how='left', on=['source','ic220_priceid'])
                    two_step_rebal_chk["non_trad_bal_where_trad_0"] = non_trad_bal_where_trad_0

                    temp = two_step_rebal_chk[["source", "available_balance"]].groupby("source").agg('sum').reset_index()
                    temp.columns = ['source', 'total_available_balance']
                    two_step_rebal_chk = pd.merge(two_step_rebal_chk, temp, how='left', on='source')
                    two_step_rebal_chk["ticker_dist"] = (two_step_rebal_chk["available_balance"]/two_step_rebal_chk["total_available_balance"]).fillna(0)

                    total_balance = temp['total_available_balance'].sum()
                    trad_balance = np.where(
                        two_step_rebal_chk["ic220_priceid"].isin(TRAD_TICKERS),
                        two_step_rebal_chk["available_balance"],
                        0
                    )
                    trad_dist = (trad_balance / total_balance).sum()
                    if total_balance == non_trad_bal_where_trad_0:
                        bad_trad_dist = 0
                    else:
                        bad_trad_dist = (trad_balance / (total_balance - non_trad_bal_where_trad_0)).sum()
                
                trad_pct = (np.where(two_step_rebal_chk["ic220_priceid"].isin(TRAD_TICKERS), 1, 0) * two_step_rebal_chk["pct"]).sum() / len(two_step_rebal_chk['source'].unique())
                two_step_rebal_chk["bad_trad_dist"] = bad_trad_dist
                
                if (bad_trad_dist > trad_pct) & illiquid_ind == 1:
                    all_src_two_step_rebal_chk_calc = pd.DataFrame()
                    for source in two_step_rebal_chk['source'].unique():
                        two_step_rebal_chk_calc = two_step_rebal_chk[two_step_rebal_chk['source'] == source]
                        two_step_rebal_chk_calc = two_step_rebal_chk_calc[~two_step_rebal_chk_calc["ic220_priceid"].isin(TRAD_TICKERS)].sort_values("ic220_priceid")
                        two_step_rebal_chk_calc["perc_non_trad"] = 0
                        two_step_rebal_chk_calc["perc_remaining_ticker"] = 0
                        two_step_rebal_chk_calc["calc_value"] = 0
                        two_step_rebal_chk_calc["used_recalc"] = 0
                        for i in range(two_step_rebal_chk_calc.shape[0]):
                            
                            if i == 0:
                                two_step_rebal_chk_calc.iloc[i,-4] = 1-trad_pct
                                two_step_rebal_chk_calc.iloc[i,-3] = 1
                                two_step_rebal_chk_calc.iloc[i,-2] = safe_divide(two_step_rebal_chk_calc.iloc[i,2], two_step_rebal_chk_calc.iloc[i,-4]) * two_step_rebal_chk_calc.iloc[i,-3]
                                two_step_rebal_chk_calc.iloc[i,-1] = round(two_step_rebal_chk_calc.iloc[i,-2],2)

                            else:
                                two_step_rebal_chk_calc.iloc[i,-4] = two_step_rebal_chk_calc.iloc[i-1,-4] - two_step_rebal_chk_calc.iloc[i-1,2]
                                two_step_rebal_chk_calc.iloc[i,-3] = two_step_rebal_chk_calc.iloc[i-1,-3] - two_step_rebal_chk_calc.iloc[i-1,-1]
                                two_step_rebal_chk_calc.iloc[i,-2] = safe_divide(two_step_rebal_chk_calc.iloc[i,2], two_step_rebal_chk_calc.iloc[i,-4]) * two_step_rebal_chk_calc.iloc[i,-3]
                                two_step_rebal_chk_calc.iloc[i,-1] = round(two_step_rebal_chk_calc.iloc[i,-2],2)

                        trad_dists = two_step_rebal_chk[two_step_rebal_chk["ic220_priceid"].isin(TRAD_TICKERS)][['source','ticker_dist']]
                        src_trad_dist = trad_dists[trad_dists["source"] == source]['ticker_dist'].values[0]
                        two_step_rebal_chk_calc["recalc_t381_pct"] = two_step_rebal_chk_calc["used_recalc"] * (1 - src_trad_dist)

                        all_src_two_step_rebal_chk_calc = pd.concat([all_src_two_step_rebal_chk_calc, two_step_rebal_chk_calc])
                    
                    two_step_rebal_chk = pd.merge(two_step_rebal_chk,all_src_two_step_rebal_chk_calc[["source","ic220_priceid","used_recalc","recalc_t381_pct"]].drop_duplicates(), how='left', on=['source','ic220_priceid'])
                    two_step_rebal_chk["used_recalc"] = two_step_rebal_chk["used_recalc"].fillna(0)                   
                    two_step_rebal_chk["recalc_t381_pct"] = np.where(two_step_rebal_chk["ic220_priceid"].isin(TRAD_TICKERS), two_step_rebal_chk["ticker_dist"], two_step_rebal_chk["recalc_t381_pct"])

                    if two_step_rebal_chk['source'].unique()[0] == 'AGG':
                        t381_calc_df = pd.merge(t381_calc_df,two_step_rebal_chk[["ic220_priceid","recalc_t381_pct"]], how='left', on='ic220_priceid').drop_duplicates()
                        t381_calc_df["t381_rebal_cash"] = t381_calc_df["final_bal_total"] * t381_calc_df["recalc_t381_pct"]
                    else:
                        t381_calc_df = pd.merge(t381_calc_df,two_step_rebal_chk[["source","ic220_priceid","recalc_t381_pct"]], how='left', on=['source','ic220_priceid']).drop_duplicates()
                        t381_calc_df["t381_rebal_cash"] = t381_calc_df["final_bal_total"] * t381_calc_df["recalc_t381_pct"]

                else:
                    t381_calc_df["t381_rebal_cash"] = t381_calc_df["final_bal_total"] * t381_calc_df["pct"]
                    two_step_rebal_chk["used_recalc"] = 0
                    two_step_rebal_chk["recalc_t381_pct"] = 0
                
                rebal_cash_df = t381_calc_df[["source","ic220_priceid","final_bal"]]
                rebal_cash_df["ic220_priceid_t381"] = np.nan
                rebal_cash_df["t381_rebal_cash_key_fund"] = 0
                rebal_cash_df["key_fund_ticker"] = None
                rebal_cash_df["t381_rebal_cash"] = t381_calc_df["t381_rebal_cash"].fillna(0)

                rebal_cash_df["iteration"] = j
                rebal_cash_df_hist = pd.concat([rebal_cash_df_hist, rebal_cash_df])
                
                break

            elif t381_df["key_fund_ticker"].unique()[0]==None:
                t381_calc_df = pd.merge(t381_calc_df,t381_df[["ic220_priceid","pct"]].drop_duplicates(), how="left", on ="ic220_priceid")
                t381_calc_df = t381_calc_df.drop_duplicates()
                t381_calc_df["pct"] = t381_calc_df["pct"].fillna(0)
                
                non_trad_bal_where_trad_0 = calc_non_trad_bal_where_trad_0(date_df[["source", "ic220_priceid", "available_balance"]], baseline)
                if non_trad_bal_where_trad_0 == 0:
                    two_step_rebal_chk = t381_calc_df[["ic220_priceid","pct"]].drop_duplicates()
                    two_step_rebal_chk.insert(0, 'source', 'AGG')
                    two_step_rebal_chk = pd.merge(two_step_rebal_chk, date_df[["ic220_priceid","available_balance"]].groupby("ic220_priceid").agg('sum').reset_index(), how='left', on ='ic220_priceid')
                    two_step_rebal_chk["non_trad_bal_where_trad_0"] = non_trad_bal_where_trad_0
                    two_step_rebal_chk["total_available_balance"] = two_step_rebal_chk["available_balance"].sum()
                    two_step_rebal_chk["ticker_dist"] = two_step_rebal_chk["available_balance"]/two_step_rebal_chk["total_available_balance"]

                    trad_dist = (np.where(two_step_rebal_chk["ic220_priceid"].isin(TRAD_TICKERS), 1, 0) * two_step_rebal_chk["ticker_dist"]).sum()
                    bad_trad_dist = trad_dist
                    
                else:
                    two_step_rebal_chk = t381_calc_df[["source","ic220_priceid","pct"]].drop_duplicates()
                    two_step_rebal_chk = pd.merge(two_step_rebal_chk, date_df[["source","ic220_priceid","available_balance"]].groupby(["source","ic220_priceid"]).agg('sum').reset_index(), how='left', on=['source','ic220_priceid'])
                    two_step_rebal_chk["non_trad_bal_where_trad_0"] = non_trad_bal_where_trad_0

                    temp = two_step_rebal_chk[["source", "available_balance"]].groupby("source").agg('sum').reset_index()
                    temp.columns = ['source', 'total_available_balance']
                    two_step_rebal_chk = pd.merge(two_step_rebal_chk, temp, how='left', on='source')
                    two_step_rebal_chk["ticker_dist"] = (two_step_rebal_chk["available_balance"]/two_step_rebal_chk["total_available_balance"]).fillna(0)

                    total_balance = temp['total_available_balance'].sum()
                    trad_balance = np.where(
                        two_step_rebal_chk["ic220_priceid"].isin(TRAD_TICKERS),
                        two_step_rebal_chk["available_balance"],
                        0
                    )
                    trad_dist = (trad_balance / total_balance).sum()
                    if total_balance == non_trad_bal_where_trad_0:
                        bad_trad_dist = 0
                    else:
                        bad_trad_dist = (trad_balance / (total_balance - non_trad_bal_where_trad_0)).sum()
                
                trad_pct = (np.where(two_step_rebal_chk["ic220_priceid"].isin(TRAD_TICKERS), 1, 0) * two_step_rebal_chk["pct"]).sum() / len(two_step_rebal_chk['source'].unique())
                two_step_rebal_chk["bad_trad_dist"] = bad_trad_dist
                
                if (bad_trad_dist > trad_pct) & illiquid_ind == 1:
                    all_src_two_step_rebal_chk_calc = pd.DataFrame()
                    for source in two_step_rebal_chk['source'].unique():
                        two_step_rebal_chk_calc = two_step_rebal_chk[two_step_rebal_chk['source'] == source]
                        two_step_rebal_chk_calc = two_step_rebal_chk_calc[~two_step_rebal_chk_calc["ic220_priceid"].isin(TRAD_TICKERS)].sort_values("ic220_priceid")
                        two_step_rebal_chk_calc["perc_non_trad"] = 0
                        two_step_rebal_chk_calc["perc_remaining_ticker"] = 0
                        two_step_rebal_chk_calc["calc_value"] = 0
                        two_step_rebal_chk_calc["used_recalc"] = 0
                        for i in range(two_step_rebal_chk_calc.shape[0]):
                            
                            if i == 0:
                                two_step_rebal_chk_calc.iloc[i,-4] = 1-trad_pct
                                two_step_rebal_chk_calc.iloc[i,-3] = 1
                                two_step_rebal_chk_calc.iloc[i,-2] = safe_divide(two_step_rebal_chk_calc.iloc[i,2], two_step_rebal_chk_calc.iloc[i,-4]) * two_step_rebal_chk_calc.iloc[i,-3]
                                two_step_rebal_chk_calc.iloc[i,-1] = round(two_step_rebal_chk_calc.iloc[i,-2],2)

                            else:
                                two_step_rebal_chk_calc.iloc[i,-4] = two_step_rebal_chk_calc.iloc[i-1,-4] - two_step_rebal_chk_calc.iloc[i-1,2]
                                two_step_rebal_chk_calc.iloc[i,-3] = two_step_rebal_chk_calc.iloc[i-1,-3] - two_step_rebal_chk_calc.iloc[i-1,-1]
                                two_step_rebal_chk_calc.iloc[i,-2] = safe_divide(two_step_rebal_chk_calc.iloc[i,2], two_step_rebal_chk_calc.iloc[i,-4]) * two_step_rebal_chk_calc.iloc[i,-3]
                                two_step_rebal_chk_calc.iloc[i,-1] = round(two_step_rebal_chk_calc.iloc[i,-2],2)

                        trad_dists = two_step_rebal_chk[two_step_rebal_chk["ic220_priceid"].isin(TRAD_TICKERS)][['source','ticker_dist']]
                        src_trad_dist = trad_dists[trad_dists["source"] == source]['ticker_dist'].values[0]
                        two_step_rebal_chk_calc["recalc_t381_pct"] = two_step_rebal_chk_calc["used_recalc"] * (1 - src_trad_dist)

                        all_src_two_step_rebal_chk_calc = pd.concat([all_src_two_step_rebal_chk_calc, two_step_rebal_chk_calc])
                    
                    two_step_rebal_chk = pd.merge(two_step_rebal_chk,all_src_two_step_rebal_chk_calc[["source","ic220_priceid","used_recalc","recalc_t381_pct"]].drop_duplicates(), how='left', on=['source','ic220_priceid'])
                    two_step_rebal_chk["used_recalc"] = two_step_rebal_chk["used_recalc"].fillna(0)                   
                    two_step_rebal_chk["recalc_t381_pct"] = np.where(two_step_rebal_chk["ic220_priceid"].isin(TRAD_TICKERS), two_step_rebal_chk["ticker_dist"], two_step_rebal_chk["recalc_t381_pct"])

                    if two_step_rebal_chk['source'].unique()[0] == 'AGG':
                        t381_calc_df = pd.merge(t381_calc_df,two_step_rebal_chk[["ic220_priceid","recalc_t381_pct"]], how='left', on='ic220_priceid').drop_duplicates()
                        t381_calc_df["t381_rebal_cash"] = t381_calc_df["final_bal_total"] * t381_calc_df["recalc_t381_pct"]
                    else:
                        t381_calc_df = pd.merge(t381_calc_df,two_step_rebal_chk[["source","ic220_priceid","recalc_t381_pct"]], how='left', on=['source','ic220_priceid']).drop_duplicates()
                        t381_calc_df["t381_rebal_cash"] = t381_calc_df["final_bal_total"] * t381_calc_df["recalc_t381_pct"]

                else:
                    t381_calc_df["t381_rebal_cash"] = t381_calc_df["final_bal_total"] * t381_calc_df["pct"]
                    two_step_rebal_chk["used_recalc"] = 0
                    two_step_rebal_chk["recalc_t381_pct"] = 0

                rebal_cash_df = t381_calc_df[["source","ic220_priceid","final_bal"]]
                rebal_cash_df["ic220_priceid_t381"] = np.nan
                rebal_cash_df["t381_rebal_cash_key_fund"] = 0
                rebal_cash_df["key_fund_ticker"] = None
                rebal_cash_df["t381_rebal_cash"] = t381_calc_df["t381_rebal_cash"].fillna(0)

                rebal_cash_df["iteration"] = j
                rebal_cash_df_hist = pd.concat([rebal_cash_df_hist, rebal_cash_df])
            
            else:
                key_fund_calc  = pd.merge(
                    t381_df[["key_fund_ticker","ic220_priceid","pct"]].drop_duplicates(), df_1[["ic220_priceid","source","final_bal"]]
                    ,how='left'
                    ,left_on="key_fund_ticker"
                    ,right_on = "ic220_priceid"
                    ,suffixes = ('_t381','_df1')
                )
                key_fund_calc = key_fund_calc.drop_duplicates()
                key_fund_calc["pct"] = key_fund_calc["pct"].fillna(0)
                key_fund_calc["t381_rebal_cash_key_fund"] = key_fund_calc["final_bal"] * key_fund_calc["pct"]

                rebal_cash_df = t381_calc_df[["source","ic220_priceid","final_bal"]]
                key_fund_calc_temp = key_fund_calc[["ic220_priceid_t381","source","t381_rebal_cash_key_fund"]].groupby(["ic220_priceid_t381","source"]).agg('sum').reset_index()
                key_fund_calc_temp.columns = ["ic220_priceid","source","t381_rebal_cash_key_fund"]
                rebal_cash_df = pd.merge(
                    rebal_cash_df, key_fund_calc_temp
                    , how='left'
                    , on = ["ic220_priceid","source"]
                )
                rebal_cash_df = pd.merge(
                    rebal_cash_df, key_fund_calc[["key_fund_ticker"]]
                    ,how='left'
                    ,left_on = ["ic220_priceid"]
                    ,right_on = ["key_fund_ticker"]
                )
                rebal_cash_df["t381_rebal_cash_key_fund"] = rebal_cash_df["t381_rebal_cash_key_fund"].fillna(0)
                rebal_cash_df = rebal_cash_df.drop_duplicates()
                rebal_cash_df["t381_rebal_cash"] = np.where(
                    rebal_cash_df['ic220_priceid'] == rebal_cash_df["key_fund_ticker"],
                    rebal_cash_df["t381_rebal_cash_key_fund"],
                    rebal_cash_df["t381_rebal_cash_key_fund"] + rebal_cash_df["final_bal"]
                )
                rebal_cash_df["t381_rebal_cash"] = rebal_cash_df["t381_rebal_cash"].fillna(0)

                rebal_cash_df["iteration"] = j
                rebal_cash_df_hist = pd.concat([rebal_cash_df_hist, rebal_cash_df])
        
        else:
            temp_df = rebal_cash_df_hist[rebal_cash_df_hist["iteration"] == j-1]
            t381_calc_df = temp_df[["source","ic220_priceid","t381_rebal_cash"]]
            t381_calc_df = pd.merge(t381_calc_df,t381_calc_df[["source","t381_rebal_cash"]].groupby("source").agg('sum').reset_index(), how='left', on='source', suffixes=('','_src_total'))
            t381_calc_df.columns = ["source", "ic220_priceid", "available_balance", "t381_rebal_cash_src_total"]
            if read_in == False and baseline == True and t381_df["key_fund_ticker"].unique()[0]==None and live_rebal_ind == 1 and date_df['unsub_ind'].unique()[0] == 0:
                t381_calc_df = pd.merge(t381_calc_df,df_1[["ic220_priceid","t381_calc"]].drop_duplicates(), how="left", on ="ic220_priceid")
                t381_calc_df = t381_calc_df.drop_duplicates()
                t381_calc_df["t381_calc"] = t381_calc_df["t381_calc"].fillna(0)
                t381_calc_df["pct"] = t381_calc_df["t381_calc"]
                
                non_trad_bal_where_trad_0 = calc_non_trad_bal_where_trad_0(t381_calc_df[["source", "ic220_priceid", "available_balance"]], baseline)
                if non_trad_bal_where_trad_0 == 0:
                    two_step_rebal_chk = t381_calc_df[["ic220_priceid","pct"]].drop_duplicates()
                    two_step_rebal_chk.insert(0, 'source', 'AGG')
                    two_step_rebal_chk = pd.merge(two_step_rebal_chk, t381_calc_df[["ic220_priceid","available_balance"]].groupby("ic220_priceid").agg('sum').reset_index(), how='left', on ='ic220_priceid')
                    two_step_rebal_chk["non_trad_bal_where_trad_0"] = non_trad_bal_where_trad_0
                    two_step_rebal_chk["total_available_balance"] = two_step_rebal_chk["available_balance"].sum()
                    two_step_rebal_chk["ticker_dist"] = two_step_rebal_chk["available_balance"]/two_step_rebal_chk["total_available_balance"]
   
                    trad_dist = (np.where(two_step_rebal_chk["ic220_priceid"].isin(TRAD_TICKERS), 1, 0) * two_step_rebal_chk["ticker_dist"]).sum()
                    bad_trad_dist = trad_dist
                    
                else:
                    two_step_rebal_chk = t381_calc_df[["source","ic220_priceid","pct"]].drop_duplicates()
                    two_step_rebal_chk = pd.merge(two_step_rebal_chk, t381_calc_df[["source","ic220_priceid","available_balance"]].groupby(["source","ic220_priceid"]).agg('sum').reset_index(), how='left', on=['source','ic220_priceid'])
                    two_step_rebal_chk["non_trad_bal_where_trad_0"] = non_trad_bal_where_trad_0

                    temp = two_step_rebal_chk[["source", "available_balance"]].groupby("source").agg('sum').reset_index()
                    temp.columns = ['source', 'total_available_balance']
                    two_step_rebal_chk = pd.merge(two_step_rebal_chk, temp, how='left', on='source')
                    two_step_rebal_chk["ticker_dist"] = (two_step_rebal_chk["available_balance"]/two_step_rebal_chk["total_available_balance"]).fillna(0)

                    total_balance = temp['total_available_balance'].sum()
                    trad_balance = np.where(
                        two_step_rebal_chk["ic220_priceid"].isin(TRAD_TICKERS),
                        two_step_rebal_chk["available_balance"],
                        0
                    )
                    trad_dist = (trad_balance / total_balance).sum()
                    trad_dist = (trad_balance / total_balance).sum()
                    if total_balance == non_trad_bal_where_trad_0:
                        bad_trad_dist = 0
                    else:
                        bad_trad_dist = (trad_balance / (total_balance - non_trad_bal_where_trad_0)).sum()
                
                trad_pct = (np.where(two_step_rebal_chk["ic220_priceid"].isin(TRAD_TICKERS), 1, 0) * two_step_rebal_chk["pct"]).sum() / len(two_step_rebal_chk['source'].unique())
                two_step_rebal_chk["bad_trad_dist"] = bad_trad_dist
                
                if (bad_trad_dist > trad_pct) & illiquid_ind == 1:
                    all_src_two_step_rebal_chk_calc = pd.DataFrame()
                    for source in two_step_rebal_chk['source'].unique():
                        two_step_rebal_chk_calc = two_step_rebal_chk[two_step_rebal_chk['source'] == source]
                        two_step_rebal_chk_calc = two_step_rebal_chk_calc[~two_step_rebal_chk_calc["ic220_priceid"].isin(TRAD_TICKERS)].sort_values("ic220_priceid")
                        two_step_rebal_chk_calc["perc_non_trad"] = 0
                        two_step_rebal_chk_calc["perc_remaining_ticker"] = 0
                        two_step_rebal_chk_calc["calc_value"] = 0
                        two_step_rebal_chk_calc["used_recalc"] = 0
                        for i in range(two_step_rebal_chk_calc.shape[0]):
                            
                            if i == 1:
                                two_step_rebal_chk_calc.iloc[i,-4] = 1-trad_pct
                                two_step_rebal_chk_calc.iloc[i,-3] = 1
                                two_step_rebal_chk_calc.iloc[i,-2] = safe_divide(two_step_rebal_chk_calc.iloc[i,2], two_step_rebal_chk_calc.iloc[i,-4]) * two_step_rebal_chk_calc.iloc[i,-3]
                                two_step_rebal_chk_calc.iloc[i,-1] = round(two_step_rebal_chk_calc.iloc[i,-2],2)

                            else:
                                two_step_rebal_chk_calc.iloc[i,-4] = two_step_rebal_chk_calc.iloc[i-1,-4] - two_step_rebal_chk_calc.iloc[i-1,2]
                                two_step_rebal_chk_calc.iloc[i,-3] = two_step_rebal_chk_calc.iloc[i-1,-3] - two_step_rebal_chk_calc.iloc[i-1,-1]
                                two_step_rebal_chk_calc.iloc[i,-2] = safe_divide(two_step_rebal_chk_calc.iloc[i,2], two_step_rebal_chk_calc.iloc[i,-4]) * two_step_rebal_chk_calc.iloc[i,-3]
                                two_step_rebal_chk_calc.iloc[i,-1] = round(two_step_rebal_chk_calc.iloc[i,-2],2)

                        trad_dists = two_step_rebal_chk[two_step_rebal_chk["ic220_priceid"].isin(TRAD_TICKERS)][['source','ticker_dist']]
                        src_trad_dist = trad_dists[trad_dists["source"] == source]['ticker_dist'].values[0]
                        two_step_rebal_chk_calc["recalc_t381_pct"] = two_step_rebal_chk_calc["used_recalc"] * (1 - src_trad_dist)

                        all_src_two_step_rebal_chk_calc = pd.concat([all_src_two_step_rebal_chk_calc, two_step_rebal_chk_calc])
                    
                    two_step_rebal_chk = pd.merge(two_step_rebal_chk,all_src_two_step_rebal_chk_calc[["source","ic220_priceid","used_recalc","recalc_t381_pct"]].drop_duplicates(), how='left', on=['source','ic220_priceid'])
                    two_step_rebal_chk["used_recalc"] = two_step_rebal_chk["used_recalc"].fillna(0)                   
                    two_step_rebal_chk["recalc_t381_pct"] = np.where(two_step_rebal_chk["ic220_priceid"].isin(TRAD_TICKERS), two_step_rebal_chk["ticker_dist"], two_step_rebal_chk["recalc_t381_pct"])

                    if two_step_rebal_chk['source'].unique()[0] == 'AGG':
                        t381_calc_df = pd.merge(t381_calc_df,two_step_rebal_chk[["ic220_priceid","recalc_t381_pct"]], how='left', on='ic220_priceid').drop_duplicates()
                        t381_calc_df["t381_rebal_cash"] = t381_calc_df["t381_rebal_cash_src_total"] * t381_calc_df["recalc_t381_pct"]
                    else:
                        t381_calc_df = pd.merge(t381_calc_df,two_step_rebal_chk[["source","ic220_priceid","recalc_t381_pct"]], how='left', on=['source','ic220_priceid']).drop_duplicates()
                        t381_calc_df["t381_rebal_cash"] = t381_calc_df["t381_rebal_cash_src_total"] * t381_calc_df["recalc_t381_pct"]

                else:
                    t381_calc_df["t381_rebal_cash"] = t381_calc_df["t381_rebal_cash_src_total"] * t381_calc_df["pct"]
                    two_step_rebal_chk["used_recalc"] = 0
                    two_step_rebal_chk["recalc_t381_pct"] = 0

                rebal_cash_df = t381_calc_df[["source","ic220_priceid","t381_rebal_cash"]]
                rebal_cash_df.columns=["source","ic220_priceid","final_bal"]
                rebal_cash_df["t381_rebal_cash_key_fund"] = 0
                rebal_cash_df["key_fund_ticker"] = None
                rebal_cash_df["t381_rebal_cash"] = t381_calc_df["t381_rebal_cash"].fillna(0)

                rebal_cash_df["iteration"] = j
                rebal_cash_df_hist = pd.concat([rebal_cash_df_hist, rebal_cash_df])
            
            elif t381_df["key_fund_ticker"].unique()[0]==None:
                t381_calc_df = pd.merge(t381_calc_df,t381_df[["ic220_priceid","pct"]].drop_duplicates(), how="left", on ="ic220_priceid")
                t381_calc_df = t381_calc_df.drop_duplicates()
                t381_calc_df["pct"] = t381_calc_df["pct"].fillna(0)
                                
                non_trad_bal_where_trad_0 = calc_non_trad_bal_where_trad_0(t381_calc_df[["source", "ic220_priceid", "available_balance"]], baseline)
                if non_trad_bal_where_trad_0 == 0:
                    two_step_rebal_chk = t381_calc_df[["ic220_priceid","pct"]].drop_duplicates()
                    two_step_rebal_chk.insert(0, 'source', 'AGG')
                    two_step_rebal_chk = pd.merge(two_step_rebal_chk, t381_calc_df[["ic220_priceid","available_balance"]].groupby("ic220_priceid").agg('sum').reset_index(), how='left', on ='ic220_priceid')
                    two_step_rebal_chk["non_trad_bal_where_trad_0"] = non_trad_bal_where_trad_0
                    two_step_rebal_chk["total_available_balance"] = two_step_rebal_chk["available_balance"].sum()
                    two_step_rebal_chk["ticker_dist"] = two_step_rebal_chk["available_balance"]/two_step_rebal_chk["total_available_balance"]
   
                    trad_dist = (np.where(two_step_rebal_chk["ic220_priceid"].isin(TRAD_TICKERS), 1, 0) * two_step_rebal_chk["ticker_dist"]).sum()
                    bad_trad_dist = trad_dist
                    
                else:
                    two_step_rebal_chk = t381_calc_df[["source","ic220_priceid","pct"]].drop_duplicates()
                    two_step_rebal_chk = pd.merge(two_step_rebal_chk, t381_calc_df[["source","ic220_priceid","available_balance"]].groupby(["source","ic220_priceid"]).agg('sum').reset_index(), how='left', on=['source','ic220_priceid'])
                    two_step_rebal_chk["non_trad_bal_where_trad_0"] = non_trad_bal_where_trad_0

                    temp = two_step_rebal_chk[["source", "available_balance"]].groupby("source").agg('sum').reset_index()
                    temp.columns = ['source', 'total_available_balance']
                    two_step_rebal_chk = pd.merge(two_step_rebal_chk, temp, how='left', on='source')
                    two_step_rebal_chk["ticker_dist"] = (two_step_rebal_chk["available_balance"]/two_step_rebal_chk["total_available_balance"]).fillna(0)

                    total_balance = temp['total_available_balance'].sum()
                    trad_balance = np.where(
                        two_step_rebal_chk["ic220_priceid"].isin(TRAD_TICKERS),
                        two_step_rebal_chk["available_balance"],
                        0
                    )
                    trad_dist = (trad_balance / total_balance).sum()
                    if total_balance == non_trad_bal_where_trad_0:
                        bad_trad_dist = 0
                    else:
                        bad_trad_dist = (trad_balance / (total_balance - non_trad_bal_where_trad_0)).sum()
                
                trad_pct = (np.where(two_step_rebal_chk["ic220_priceid"].isin(TRAD_TICKERS), 1, 0) * two_step_rebal_chk["pct"]).sum() / len(two_step_rebal_chk['source'].unique())
                two_step_rebal_chk["bad_trad_dist"] = bad_trad_dist
                
                if (bad_trad_dist > trad_pct) & illiquid_ind == 1:
                    all_src_two_step_rebal_chk_calc = pd.DataFrame()
                    for source in two_step_rebal_chk['source'].unique():
                        two_step_rebal_chk_calc = two_step_rebal_chk[two_step_rebal_chk['source'] == source]
                        two_step_rebal_chk_calc = two_step_rebal_chk_calc[~two_step_rebal_chk_calc["ic220_priceid"].isin(TRAD_TICKERS)].sort_values("ic220_priceid")
                        two_step_rebal_chk_calc["perc_non_trad"] = 0
                        two_step_rebal_chk_calc["perc_remaining_ticker"] = 0
                        two_step_rebal_chk_calc["calc_value"] = 0
                        two_step_rebal_chk_calc["used_recalc"] = 0
                        for i in range(two_step_rebal_chk_calc.shape[0]):
                            
                            if i == 0:
                                two_step_rebal_chk_calc.iloc[i,-4] = 1-trad_pct
                                two_step_rebal_chk_calc.iloc[i,-3] = 1
                                two_step_rebal_chk_calc.iloc[i,-2] = safe_divide(two_step_rebal_chk_calc.iloc[i,2], two_step_rebal_chk_calc.iloc[i,-4]) * two_step_rebal_chk_calc.iloc[i,-3]
                                two_step_rebal_chk_calc.iloc[i,-1] = round(two_step_rebal_chk_calc.iloc[i,-2],2)

                            else:
                                two_step_rebal_chk_calc.iloc[i,-4] = two_step_rebal_chk_calc.iloc[i-1,-4] - two_step_rebal_chk_calc.iloc[i-1,2]
                                two_step_rebal_chk_calc.iloc[i,-3] = two_step_rebal_chk_calc.iloc[i-1,-3] - two_step_rebal_chk_calc.iloc[i-1,-1]
                                two_step_rebal_chk_calc.iloc[i,-2] = safe_divide(two_step_rebal_chk_calc.iloc[i,2], two_step_rebal_chk_calc.iloc[i,-4]) * two_step_rebal_chk_calc.iloc[i,-3]
                                two_step_rebal_chk_calc.iloc[i,-1] = round(two_step_rebal_chk_calc.iloc[i,-2],2)

                        trad_dists = two_step_rebal_chk[two_step_rebal_chk["ic220_priceid"].isin(TRAD_TICKERS)][['source','ticker_dist']]
                        src_trad_dist = trad_dists[trad_dists["source"] == source]['ticker_dist'].values[0]
                        two_step_rebal_chk_calc["recalc_t381_pct"] = two_step_rebal_chk_calc["used_recalc"] * (1 - src_trad_dist)

                        all_src_two_step_rebal_chk_calc = pd.concat([all_src_two_step_rebal_chk_calc, two_step_rebal_chk_calc])
                    
                    two_step_rebal_chk = pd.merge(two_step_rebal_chk,all_src_two_step_rebal_chk_calc[["source","ic220_priceid","used_recalc","recalc_t381_pct"]].drop_duplicates(), how='left', on=['source','ic220_priceid'])
                    two_step_rebal_chk["used_recalc"] = two_step_rebal_chk["used_recalc"].fillna(0)                   
                    two_step_rebal_chk["recalc_t381_pct"] = np.where(two_step_rebal_chk["ic220_priceid"].isin(TRAD_TICKERS), two_step_rebal_chk["ticker_dist"], two_step_rebal_chk["recalc_t381_pct"])

                    if two_step_rebal_chk['source'].unique()[0] == 'AGG':
                        t381_calc_df = pd.merge(t381_calc_df,two_step_rebal_chk[["ic220_priceid","recalc_t381_pct"]], how='left', on='ic220_priceid').drop_duplicates()
                        t381_calc_df["t381_rebal_cash"] = t381_calc_df["t381_rebal_cash_src_total"] * t381_calc_df["recalc_t381_pct"]
                    else:
                        t381_calc_df = pd.merge(t381_calc_df,two_step_rebal_chk[["source","ic220_priceid","recalc_t381_pct"]], how='left', on=['source','ic220_priceid']).drop_duplicates()
                        t381_calc_df["t381_rebal_cash"] = t381_calc_df["t381_rebal_cash_src_total"] * t381_calc_df["recalc_t381_pct"]

                else:
                    t381_calc_df["t381_rebal_cash"] = t381_calc_df["t381_rebal_cash_src_total"] * t381_calc_df["pct"]
                    two_step_rebal_chk["used_recalc"] = 0
                    two_step_rebal_chk["recalc_t381_pct"] = 0
                
                rebal_cash_df = t381_calc_df[["source","ic220_priceid","t381_rebal_cash"]]
                rebal_cash_df.columns=["source","ic220_priceid","final_bal"]
                rebal_cash_df["t381_rebal_cash_key_fund"] = 0
                rebal_cash_df["key_fund_ticker"] = None
                rebal_cash_df["t381_rebal_cash"] = t381_calc_df["t381_rebal_cash"].fillna(0)

                rebal_cash_df["iteration"] = j
                rebal_cash_df_hist = pd.concat([rebal_cash_df_hist, rebal_cash_df])
            
            else:
                key_fund_calc  = pd.merge(
                    t381_df[["key_fund_ticker","ic220_priceid","pct"]].drop_duplicates(), t381_calc_df
                    ,how='left'
                    ,left_on="key_fund_ticker"
                    ,right_on = "ic220_priceid"
                    ,suffixes = ('_t381','_df1')
                )
                key_fund_calc = key_fund_calc.drop_duplicates()
                key_fund_calc["pct"] = key_fund_calc["pct"].fillna(0)
                key_fund_calc["t381_rebal_cash_key_fund"] = key_fund_calc["available_balance"]*key_fund_calc["pct"]

                rebal_cash_df = t381_calc_df[["source","ic220_priceid","available_balance"]]
                rebal_cash_df.columns=["source","ic220_priceid","final_bal"]
                key_fund_calc_temp = key_fund_calc[["ic220_priceid_t381","source","t381_rebal_cash_key_fund"]].groupby(["ic220_priceid_t381","source"]).agg('sum').reset_index()
                key_fund_calc_temp.columns = ["ic220_priceid","source","t381_rebal_cash_key_fund"]
                rebal_cash_df = pd.merge(
                    rebal_cash_df, key_fund_calc_temp
                    , how='left'
                    , on = ["ic220_priceid","source"]
                )
                rebal_cash_df = pd.merge(
                    rebal_cash_df, key_fund_calc[["key_fund_ticker"]]
                    ,how='left'
                    ,left_on = ["ic220_priceid"]
                    ,right_on = ["key_fund_ticker"]
                )
                rebal_cash_df["t381_rebal_cash_key_fund"] = rebal_cash_df["t381_rebal_cash_key_fund"].fillna(0)
                rebal_cash_df = rebal_cash_df.drop_duplicates()
                rebal_cash_df["t381_rebal_cash"] = np.where(
                    rebal_cash_df['ic220_priceid'] == rebal_cash_df["key_fund_ticker"],
                    rebal_cash_df["t381_rebal_cash_key_fund"],
                    rebal_cash_df["t381_rebal_cash_key_fund"] + rebal_cash_df["final_bal"]
                )
                rebal_cash_df["t381_rebal_cash"] = rebal_cash_df["t381_rebal_cash"].fillna(0)
                
                rebal_cash_df["iteration"] = j
                rebal_cash_df_hist = pd.concat([rebal_cash_df_hist, rebal_cash_df])
    
    df_1 = pd.merge(df_1,rebal_cash_df[["source","ic220_priceid","t381_rebal_cash"]], how="left", on = ["source","ic220_priceid"])
    
    return df_1, rebal_cash_df_hist, two_step_rebal_chk

def apply_fees(df_1, date_df, partic_proj):
    """Applies fees when the projection date is at the end of the quarter.
    
    Args:
        df_1: Main DataFrame containing the projections results for each day.
        date_df: The BTR data filtered on the specific partic_id and current
          date of the projection.
        partic_proj: Historical projection results from previous days of the
          projection
    
    Returns:
        df_1 with the proj_fee_amt joined to df_1 at the source-ticker level.
        This means trad is aggregated from the vintage level to the source
        level.
    """
    fee_df = partic_proj.loc[
        (
            partic_proj['calendar_dates'] >=
            date_df['quarter_start_dt'].drop_duplicates().reset_index(drop=True)[0]
        ) &
        (
            partic_proj['calendar_dates'] <=
            date_df['quarter_end_dt'].drop_duplicates().reset_index(drop=True)[0]
        ),
        [
            'ic220_priceid',
            'source',
            'projection',
        ]
    ]
    fee_df = fee_df.rename({'projection': 'proj_fee_amt'}, axis=1)
    df_1 = pd.merge(
        df_1,
        fee_df.groupby(['source', 'ic220_priceid']).agg('sum').reset_index(),
        on=['source', 'ic220_priceid'],
        how='left',
    )
    df_1['proj_fee_cash'] = df_1['proj_fee_amt'] * df_1['fee_rate']

    return df_1

def projection_rebal(date_df, partic_proj, rebal_output, read_in, baseline, drift_rebal_ind, id, today, t444_4_date):
    """Calculates the daily projection when there is a rebalancing event.

    This function is called when it is not the first day of the projection, and
    when there is a rebalancing event occuring on the current day. Since there
    is a separate treatment between trad and non-trad assets, these tickers are
    separated and their debugging data is tracked separately in the trad and
    non-trad debug tables respectively. This also includes a section of logic
    that handles rebalancing since this is the projection function for
    rebalancing events. Throughout the function, there are multiple steps of
    logic that are applied in order. This is labeled clearly with inline
    comments, and many sections will also output their own debug tables that
    are eventually exported with the primary projection results.

    Args:
        date_df: The BTR data filtered on the specific partic_id and current
          date of the projection.
        partic_proj: The partic_id level projection results from the results
          dictionary. This is passed as a separate argument so the original
          date_df DataFrame can remain unaltered outside of the scope of the
          projection functions.
        rebal_output: A DataFrame containing the rebal data from the calculated
          t381s from the rebalancing logic.
        read_in: Boolean value specifying if we are reading in rebalancing data
          or calculating it live.
        baseline: Boolean value specifying if the run is baseline or corrected.
        drift_rebal_ind: An indicator as to whether this rebalacing was 
          triggered by drift or not.
        id: partic_id containing a concatination of pin, plan, and sub_plan.
        today: The current date of the projection in YYYY-MM-DD fomat.
        t444_4_date: An indicator that is stored and updates for when a t444_4
          debit transaction is encountered. This is to keep track of the most
          recent transaction date throughout the projection which is needed for
          properly applying the accrued dividends logic.
    
    Returns:
        A tuple of DataFrames corresponding to main projection output, and extra
        debugging table output.
            date_df: Contains the prepped BTR data, but also now will constantly
              update with the new fields pertaining to the projection.
            df_2: Contains the separated projection data for trad assets. This
              is recombined into the date_df final output in the function but is
              also kept separate for debugging due to different treatment of
              trad and non-trad assets.
            df_1: Contains the separated projection data for non-trad assets.
              This is recombined into the date_df final output in the function
              but is also kept separate for debugging due to separate treatment
              of trad and non-trad assets.
            residual_chk: A DataFrame that checks for residual cash between the
              calculated final balance and the sum of total cash transactions
              and available balance.
            accrued_div_calc: Debugging table that tracks the data throughout
              the accrued dividends logic.
            rebal_cash_hist: Debugging table that tracks the rebalancing cash
              history thoughout the key fund logic and cash rebalancing process.
            two_step_rebal_chk: Debugging table that tracks the data througout
              the two step rebalancing logic.
            t444_4_date: An indicator that is stored and updates for when a
              t444_4 debit transaction is encountered. This is to keep track of
              the most recent transaction date throughout the projection which
              is needed for properly applying the accrued dividends logic.
    """
    # Separate trad and non-trad tickers from the participant DataFrame.
    is_aait_trad = date_df["ic220_priceid"].isin(TRAD_TICKERS)
    non_trad_df_1 = date_df.loc[~is_aait_trad, ["ic220_priceid","source","prev_day_proj","balance", "t114_1", "t381_4", "t394_12", "cash_sum", "debit_t381_cash_trans", "credit_t381_cash_trans"]].groupby(["ic220_priceid","source"]).agg('sum').reset_index()
    non_trad_df_2 = date_df.loc[~is_aait_trad, ["ic220_priceid","t815_pct"]].drop_duplicates()
    non_trad_df_2 = pd.merge(non_trad_df_1,non_trad_df_2, how='left', on="ic220_priceid")
    non_trad_df_2 = pd.merge(non_trad_df_2,date_df.loc[~is_aait_trad, ["source","ic220_priceid","fee_rate"]].drop_duplicates(), how='left', on=["source", "ic220_priceid"])
        
    trad_df_1 = date_df.loc[is_aait_trad, ["ic220_priceid","source","prev_day_proj","balance"]].groupby(["ic220_priceid","source"]).agg('sum').reset_index()
    trad_df_2 = date_df.loc[is_aait_trad, ["ic220_priceid","source","t114_1", "t381_4", "t394_12", "cash_sum","t815_pct","fee_rate", "debit_t381_cash_trans", "credit_t381_cash_trans"]].drop_duplicates()
    trad_df_2 = pd.merge(trad_df_1,trad_df_2, how='left', on=["ic220_priceid","source"])
    # Join the separated versions and set the t815 and t381 calcs to 0.
    df_1 = pd.concat((non_trad_df_2,trad_df_2), axis=0)

    t815_read_in_ind = date_df["t815_read_in_ind"].unique()[0]
    
    if (read_in == False and baseline == True and t815_read_in_ind == 1) or (read_in == False and baseline == False and today >= date_df['proj_end_dt'].unique()[0]) or date_df['unsub_ind'].unique()[0] == 1:
        df_1 = pd.merge(df_1,rebal_output[["ic220_priceid","t381_calc"]], how='left', on='ic220_priceid')
        df_1["t815_calc"] = df_1["t815_pct"]
        df_1["t815_read_in_ind"] = t815_read_in_ind
    else:
        df_1 = pd.merge(df_1,rebal_output[["ic220_priceid","t381_calc","t815_calc"]], how='left', on='ic220_priceid')
        df_1["t815_read_in_ind"] = t815_read_in_ind

    df_1 = pd.merge(df_1, df_1[["source","t114_1", "t381_4", "t394_12"]].groupby("source").agg('sum').reset_index(), suffixes=("","_total"), how='left', on='source')
    
    if read_in:
        df_1["proj_t114_1"] = df_1["t114_1_total"] * df_1["t815_pct"]
        df_1["proj_t381_4"] = df_1["t381_4_total"] * df_1["t815_pct"]
        df_1["proj_t394_12"] = df_1["t394_12_total"] * df_1["t815_pct"]
    elif read_in == False and baseline == False and today > date_df['proj_end_dt'].unique()[0]:
        df_1["proj_t114_1"] = 0
        df_1["proj_t381_4"] = 0
        df_1["proj_t394_12"] = 0
    else:
        df_1["proj_t114_1"] = df_1["t114_1_total"]*df_1["t815_calc"]
        df_1["proj_t381_4"] = df_1["t381_4_total"] * df_1["t815_calc"]
        df_1["proj_t394_12"] = df_1["t394_12_total"] * df_1["t815_calc"]

    ## START ACCRUED DIVIDENDS LOGIC ##
    if date_df['t444_debit_ind'].drop_duplicates().reset_index(drop=True)[0] == 1:
        t444_4_date = today
        accrued_div_calc = partic_proj.loc[
            (
                partic_proj['calendar_dates'].apply(lambda x: x.month) ==
                date_df['calendar_dates'].drop_duplicates().reset_index(drop=True)[0].month
            ) &
            (
                partic_proj['calendar_dates'].apply(lambda x: x.year) ==
                date_df['calendar_dates'].drop_duplicates().reset_index(drop=True)[0].year
            ) &
            (
                partic_proj['calendar_dates'].apply(lambda x: x.day) <=
                date_df['calendar_dates'].drop_duplicates().reset_index(drop=True)[0].day
            ),
            [
                'pin',
                'plan',
                'sub_plan',
                'calendar_dates',
                'source',
                'ic220_priceid',
                'contract_vintage',
                'lead_transformed_rate_price',
                'projection',
            ]
        ]
        accrued_div_calc['div_calc'] = accrued_div_calc['lead_transformed_rate_price'] * accrued_div_calc['projection']
        df_1 = pd.merge(
            df_1,
            accrued_div_calc[['source', 'ic220_priceid', 'div_calc']].groupby(['source', 'ic220_priceid']).agg('sum').reset_index(),
            how='left',
            on=['source', 'ic220_priceid'],
        )
        df_1['div_calc'] = df_1['div_calc'].fillna(0)
    
    elif date_df['month_last_working_dt_ind'].drop_duplicates().reset_index(drop=True)[0] == 1:
        if t444_4_date != 0:
            if t444_4_date.year == today.year and t444_4_date.month == today.month:
                accrued_div_calc = partic_proj.loc[
                    (
                        partic_proj['calendar_dates'].apply(lambda x: x.month) ==
                        date_df['calendar_dates'].drop_duplicates().reset_index(drop=True)[0].month
                    ) &
                    (
                        partic_proj['calendar_dates'].apply(lambda x: x.year) ==
                        date_df['calendar_dates'].drop_duplicates().reset_index(drop=True)[0].year
                    ) &
                    (
                        partic_proj['calendar_dates'].apply(lambda x: x.year) > t444_4_date.day
                    ),
                    [
                        'pin',
                        'plan',
                        'sub_plan',
                        'calendar_dates',
                        'source',
                        'ic220_priceid',
                        'contract_vintage',
                        'lead_transformed_rate_price',
                        'projection',
                    ]
                ]
            else:
                accrued_div_calc = partic_proj.loc[
                    (
                        partic_proj['calendar_dates'].apply(lambda x: x.month) ==
                        date_df['calendar_dates'].drop_duplicates().reset_index(drop=True)[0].month
                    ) &
                    (
                        partic_proj['calendar_dates'].apply(lambda x: x.year) ==
                        date_df['calendar_dates'].drop_duplicates().reset_index(drop=True)[0].year
                    ),
                    [
                        'pin',
                        'plan',
                        'sub_plan',
                        'calendar_dates',
                        'source',
                        'ic220_priceid',
                        'contract_vintage',
                        'lead_transformed_rate_price',
                        'projection',
                    ]
                ]
        else:
            accrued_div_calc = partic_proj.loc[
                (
                    partic_proj['calendar_dates'].apply(lambda x: x.month) ==
                    date_df['calendar_dates'].drop_duplicates().reset_index(drop=True)[0].month
                ) &
                (
                    partic_proj['calendar_dates'].apply(lambda x: x.year) ==
                    date_df['calendar_dates'].drop_duplicates().reset_index(drop=True)[0].year
                ),
                [
                    'pin',
                    'plan',
                    'sub_plan',
                    'calendar_dates',
                    'source',
                    'ic220_priceid',
                    'contract_vintage',
                    'lead_transformed_rate_price',
                    'projection',
                ]
            ]
        accrued_div_calc['div_calc'] = accrued_div_calc['lead_transformed_rate_price'] * accrued_div_calc['projection']
        df_1 = pd.merge(
            df_1,
            accrued_div_calc[['source', 'ic220_priceid', 'div_calc']].groupby(['source', 'ic220_priceid']).agg('sum').reset_index(),
            how='left',
            on=['source', 'ic220_priceid'],
        )
        df_1['div_calc'] = df_1['div_calc'].fillna(0)

    else:
        accrued_div_calc = pd.DataFrame()
        df_1['div_calc'] = 0
    ## END ACCRUED DIVIDENDS LOGIC ##

    ## START FEE LOGIC ##
    if date_df["fee_dt_ind"].max() == 1:
        df_1 = apply_fees(df_1, date_df, partic_proj)
        if read_in == False and baseline == False and today > date_df['proj_end_dt'].unique()[0] :
            df_1['proj_fee_cash'] = np.where(
                df_1['proj_fee_cash'] > 0,
                0,
                df_1['proj_fee_cash']
            )
    else:
        df_1['proj_fee_amt'] = 0
        df_1['proj_fee_cash'] = 0
    ## END FEE LOGIC ##
    
    # NEGATIVE PROJECTION LOGIC START #
    date_df["available_balance"] = date_df["prev_day_proj"]*date_df["return_perc"]

    df_1 = pd.merge(df_1,date_df[["source","ic220_priceid","available_balance"]].groupby(["source","ic220_priceid"]).agg('sum').reset_index(), how='left', on=["source","ic220_priceid"])

    if date_df["t381_cash_trans_ind"].unique()[0] == 1:
        t381_cash_trans_df = query_t381_cash_trans_df(date_df["pin"].unique()[0], date_df["plan"].unique()[0], date_df["sub_plan"].unique()[0], today)
        t381_cash_trans_df = pd.merge(t381_cash_trans_df, df_1[["ic220_priceid", "source", "available_balance","debit_t381_cash_trans"]], left_on = ["debit_ticker", "source"], right_on = ["ic220_priceid", "source"], how='left')
        t381_cash_trans_df = pd.merge(t381_cash_trans_df, df_1[["ic220_priceid", "source", "credit_t381_cash_trans"]], left_on = ["credit_ticker", "source"], right_on = ["ic220_priceid", "source"], how='left')
        t381_cash_trans_df["t381_debit_capping_chk"] = np.where(
            t381_cash_trans_df["debit_t381_cash_trans"] != 0,
            np.where(t381_cash_trans_df["debit_t381_cash_trans"] > t381_cash_trans_df["available_balance"], 1, 0 ),
            0
        )
        t381_cash_trans_df["new_debit_t381_cash_trans"] = np.where(t381_cash_trans_df["t381_debit_capping_chk"] == 1, t381_cash_trans_df["available_balance"], t381_cash_trans_df["debit_t381_cash_trans"])
        t381_cash_trans_df["credit_t381_cash_trans_adjustment"] = np.where(t381_cash_trans_df["t381_debit_capping_chk"]==1, t381_cash_trans_df["available_balance"] / t381_cash_trans_df["debit_t381_cash_trans"], 1)
        t381_cash_trans_df["new_credit_t381_cash_trans"] = t381_cash_trans_df["credit_t381_cash_trans"]*t381_cash_trans_df["credit_t381_cash_trans_adjustment"]

        df_1 = pd.merge(df_1, t381_cash_trans_df[["credit_ticker", "source","new_credit_t381_cash_trans"]].drop_duplicates(), left_on = ["ic220_priceid", "source"], right_on = ["credit_ticker", "source"], how='left')
        df_1 = pd.merge(df_1, t381_cash_trans_df[["debit_ticker","source","new_debit_t381_cash_trans"]].drop_duplicates(), left_on = ["ic220_priceid", "source"], right_on = ["debit_ticker", "source"], how='left')
        df_1["new_credit_t381_cash_trans"] = df_1["new_credit_t381_cash_trans"].fillna(0)
        df_1["new_debit_t381_cash_trans"] = df_1["new_debit_t381_cash_trans"].fillna(0)

    else:
        df_1["new_credit_t381_cash_trans"] = 0
        df_1["new_debit_t381_cash_trans"] = 0

    if read_in == False and baseline == False and today > date_df['proj_end_dt'].unique()[0]:
        df_1["total_trans_cash"] = df_1['div_calc'] + df_1["proj_fee_cash"]
    else:
        if (baseline and date_df['order_of_ops_bl_ind'].unique()[0]) or (not baseline and date_df['order_of_ops_cr_ind'].unique()[0]):
            df_1["total_trans_cash"] = df_1['div_calc'] + df_1["proj_fee_cash"] + df_1["cash_sum"]
        else:
            df_1["total_trans_cash"] = df_1["proj_t114_1"] + df_1["proj_t381_4"] + df_1["proj_t394_12"] + df_1['div_calc'] + df_1["proj_fee_cash"] + df_1["cash_sum"] + df_1["new_debit_t381_cash_trans"] + df_1["new_credit_t381_cash_trans"]
    df_1["residual"] = df_1["available_balance"] + df_1["total_trans_cash"]
    df_1["neg_proj_ind"] = np.where(df_1["residual"]<0, 1, 0)
    df_1["pos_proj_ind"] = 1-df_1["neg_proj_ind"]
    df_1['pos_residual'] = df_1["pos_proj_ind"]*df_1["residual"]
    df_1['neg_residual'] = df_1["neg_proj_ind"]*df_1["residual"]
    df_1 = pd.merge(df_1, df_1[["source","pos_residual","neg_residual"]].groupby("source").agg('sum').reset_index(), suffixes=("","_src_total"), how='left', on='source')
    df_1["positive_pct"] = df_1['pos_residual']/df_1['pos_residual_src_total']
    df_1["redist_neg"] = df_1['neg_residual_src_total']*df_1["positive_pct"]
    df_1["final_bal"] = np.where((df_1["residual"]+df_1["redist_neg"])<0, 0, df_1["residual"]+df_1["redist_neg"])
    df_1["final_bal"] = df_1["final_bal"].fillna(0)
    df_1["final_bal_neg_proj_chk"] = df_1["final_bal"].copy()

    df_1["total_trans_calc"] = df_1["final_bal"] - df_1["available_balance"]

    residual_chk = df_1[["source","available_balance","final_bal","total_trans_cash"]].groupby("source").agg('sum').reset_index()
    residual_chk["trans_residual"] = residual_chk["available_balance"] + residual_chk["total_trans_cash"] - residual_chk["final_bal"]
    ## NEGATIVE PROJECTION LOGIC END ##
    
    # KEY FUND LOGIC START 
    # Calculate cash projections
    df_1 = pd.merge(df_1, df_1[["source","final_bal"]].groupby("source").agg('sum').reset_index(), suffixes=("","_total"), how='left', on='source')
    t381 = query_t381(id, today)

    if (
        (baseline or date_df['unsub_ind'].unique()[0] == 1 or (not baseline and today >= date_df['proj_end_dt'].unique()[0]))
        and not t381.empty
        and drift_rebal_ind == 0
    ):
        df_1["recalc_t381_pct"] = 0
        df_1, rebal_cash_hist, two_step_rebal_chk = calc_key_fund_logic(t381, df_1, date_df, read_in, baseline)
    else:
        two_step_rebal_chk = df_1[["ic220_priceid","t381_calc"]].drop_duplicates()
        two_step_rebal_chk.columns = ['ic220_priceid', 'pct']
        two_step_rebal_chk.insert(0, 'source', 'AGG')
        two_step_rebal_chk = pd.merge(two_step_rebal_chk, date_df[["ic220_priceid","available_balance"]].groupby("ic220_priceid").agg('sum').reset_index(), how='left', on ='ic220_priceid')
        two_step_rebal_chk['non_trad_bal_where_trad_0'] = 0
        two_step_rebal_chk["total_available_balance"] = two_step_rebal_chk["available_balance"].sum()
        two_step_rebal_chk["ticker_dist"] = two_step_rebal_chk["available_balance"]/two_step_rebal_chk["total_available_balance"]
        two_step_rebal_chk["bad_trad_dist"] = 0

        trad_pct = (np.where(two_step_rebal_chk["ic220_priceid"].isin(TRAD_TICKERS), 1, 0) * two_step_rebal_chk["pct"]).sum()
        trad_dist = (np.where(two_step_rebal_chk["ic220_priceid"].isin(TRAD_TICKERS), 1, 0) * two_step_rebal_chk["ticker_dist"]).sum()
        
        if baseline:
            illiquid_ind = date_df["bl_omni_illiquid_ind"].unique()[0]
        else:
            illiquid_ind = date_df["cr_omni_illiquid_ind"].unique()[0]

        if (trad_dist>trad_pct) & illiquid_ind == 1:
            two_step_rebal_chk_calc = two_step_rebal_chk[~two_step_rebal_chk["ic220_priceid"].isin(TRAD_TICKERS)].sort_values("ic220_priceid")
            two_step_rebal_chk_calc["perc_non_trad"] = 0
            two_step_rebal_chk_calc["perc_remaining_ticker"] = 0
            two_step_rebal_chk_calc["calc_value"] = 0
            two_step_rebal_chk_calc["used_recalc"] = 0
            for i in range(two_step_rebal_chk_calc.shape[0]):
                
                if i == 0:
                    two_step_rebal_chk_calc.iloc[i,-4] = 1-trad_pct
                    two_step_rebal_chk_calc.iloc[i,-3] = 1
                    two_step_rebal_chk_calc.iloc[i,-2] = safe_divide(two_step_rebal_chk_calc.iloc[i,2], two_step_rebal_chk_calc.iloc[i,-4])*two_step_rebal_chk_calc.iloc[i,-3]
                    two_step_rebal_chk_calc.iloc[i,-1] = round(two_step_rebal_chk_calc.iloc[i,-2],2)

                else:
                    two_step_rebal_chk_calc.iloc[i,-4] = two_step_rebal_chk_calc.iloc[i-1,-4] - two_step_rebal_chk_calc.iloc[i-1,2]
                    two_step_rebal_chk_calc.iloc[i,-3] = two_step_rebal_chk_calc.iloc[i-1,-3] - two_step_rebal_chk_calc.iloc[i-1,-1]
                    two_step_rebal_chk_calc.iloc[i,-2] = safe_divide(two_step_rebal_chk_calc.iloc[i,2], two_step_rebal_chk_calc.iloc[i,-4])*two_step_rebal_chk_calc.iloc[i,-3]
                    two_step_rebal_chk_calc.iloc[i,-1] = round(two_step_rebal_chk_calc.iloc[i,-2],2)

            two_step_rebal_chk = pd.merge(two_step_rebal_chk,two_step_rebal_chk_calc[["ic220_priceid","used_recalc"]], how='left', on ='ic220_priceid')
            two_step_rebal_chk["used_recalc"] = two_step_rebal_chk["used_recalc"].fillna(0)
            
            two_step_rebal_chk["recalc_t381_pct"] = np.where(two_step_rebal_chk["ic220_priceid"].isin(TRAD_TICKERS), two_step_rebal_chk["ticker_dist"], two_step_rebal_chk["used_recalc"]*(1-trad_dist))
            
            df_1 = pd.merge(df_1, two_step_rebal_chk[["ic220_priceid","recalc_t381_pct"]], how='left', on='ic220_priceid').drop_duplicates()
            df_1["t381_rebal_cash"] = df_1["final_bal_total"] * df_1["recalc_t381_pct"]

        else:
            df_1["recalc_t381_pct"] = 0
            df_1["t381_rebal_cash"] = df_1["final_bal_total"]*df_1["t381_calc"]
            two_step_rebal_chk = pd.DataFrame()

        rebal_cash_hist = pd.DataFrame(columns=[
            'source',
            'ic220_priceid',
            'final_bal',
            'ic220_priceid_t381',
            't381_rebal_cash_key_fund',
            'key_fund_ticker',
            't381_rebal_cash',
            'iteration',
        ])
    
    # KEY FUND LOGIC END #
    df_1["proj_t381_rebal_cash"] = (df_1["t381_rebal_cash"] - df_1["final_bal"]).fillna(0)

    # Calculate all transaction cash for trad only.
    if (baseline and date_df['order_of_ops_bl_ind'].unique()[0]) or (not baseline and date_df['order_of_ops_cr_ind'].unique()[0] and today <= date_df['proj_end_dt'].unique()[0]):
        df_1["aait_trad_trans_cash"] = np.where(df_1["ic220_priceid"].isin(TRAD_TICKERS), 1, 0) * (df_1["total_trans_calc"] + df_1["proj_t114_1"] + df_1["proj_t381_4"] + df_1["proj_t394_12"] + df_1["new_debit_t381_cash_trans"] + df_1["new_credit_t381_cash_trans"])
    else:
        df_1["aait_trad_trans_cash"] = np.where(df_1["ic220_priceid"].isin(TRAD_TICKERS), 1, 0) * df_1["total_trans_calc"]

    # Separate the active vintage from all other vintages and agg sum on source.
    df_2 = date_df.loc[date_df["ic220_priceid"].isin(TRAD_TICKERS),["calendar_dates","ic220_priceid","source","balance","return_perc","contract_vintage","contract_open","contract_close","prev_day_proj", "available_balance"]]
    df_2["active_vintage"] = np.where((df_2["calendar_dates"]>=df_2["contract_open"]) & (df_2["calendar_dates"]<=df_2["contract_close"]),1,0)
    df_2 = pd.merge(df_2, df_1[["source","aait_trad_trans_cash"]].groupby("source").agg('sum').reset_index(), how='left', on='source')
    df_2 = pd.merge(df_2, df_2[["source","prev_day_proj"]].groupby("source").agg('sum').reset_index(), suffixes=("","_src_total"), how='left', on='source')
    # Set the sell and buy values for trad
    df_3 = df_1.loc[df_1["ic220_priceid"].isin(TRAD_TICKERS),["source","proj_t381_rebal_cash"]].drop_duplicates()
    df_3["sell"] = np.where(df_3["proj_t381_rebal_cash"]<0, df_3["proj_t381_rebal_cash"], 0)
    df_3["buy"] = np.where(df_3["proj_t381_rebal_cash"]>0, df_3["proj_t381_rebal_cash"], 0)

    # Apply excel logic to get prorate and new vintage values.
    if df_3["buy"].sum()<=df_3["sell"].sum()*-1:
        df_3["prorate"] = df_3["buy"]+df_3["sell"]
        df_3["new_vintage"] = 0
    else:
        df_3["prorate"] = df_3["sell"] + ((df_3["buy"]/df_3["buy"].sum())*df_3["sell"].sum()*-1)
        df_3["new_vintage"] = (df_3["buy"]/df_3["buy"].sum())*(df_3["sell"].sum()+df_3["buy"].sum())

    # Join calculated values and aggregate sum by source.
    df_2 = pd.merge(df_2, df_3[["source","buy","sell","prorate","new_vintage"]], how='left', on='source')
    df_2 = pd.merge(df_2, df_2[["source", "available_balance"]].groupby('source').agg('sum').reset_index(), how='left', on='source', suffixes=("","_src_total"))
    df_2["pre_rebal_bal"] = np.where(
        df_2["aait_trad_trans_cash"] >= 0,
        np.where(
            df_2["active_vintage"]==1,
            (df_2["prev_day_proj"]*df_2["return_perc"]) + df_2["aait_trad_trans_cash"],
            df_2["prev_day_proj"]*df_2["return_perc"]
        ),
        (df_2["prev_day_proj"]*df_2["return_perc"]) + df_2["aait_trad_trans_cash"]*(df_2["available_balance"]/df_2["available_balance_src_total"])
    )
    df_2 = pd.merge(df_2, df_2[["source","pre_rebal_bal"]].groupby("source").agg('sum').reset_index(), suffixes=("","_src_sum"), how='left', on="source")
    # Calculate prorate out and aggregate sum by vintage.
    df_2["prorate_out"] = np.where(df_2["prorate"]<0, ((df_2["pre_rebal_bal_src_sum"]+df_2["prorate"])/df_2["pre_rebal_bal_src_sum"])*df_2["pre_rebal_bal"], df_2["pre_rebal_bal"] )
    df_2 = pd.merge(df_2, df_2[["contract_vintage","pre_rebal_bal","prorate_out"]].groupby(["contract_vintage"]).agg('sum').reset_index(), suffixes=("","_vint_sum"), how='left', on="contract_vintage")
    # Calculate prorate in and calculate the trad transaction amount.
    df_2["prorate_in"] = np.where(df_2["prorate"]>0, df_2["prorate_out"] + (df_2["prorate"]/df_3["sell"].sum()*-1) * (df_2["pre_rebal_bal_vint_sum"]-df_2["prorate_out_vint_sum"]), df_2["prorate_out"] )
    df_2["new_vintage_post_rebal"] = df_2["prorate_in"]+ (df_2["new_vintage"]*df_2["active_vintage"])
    df_2["trad_trans"] = df_2["new_vintage_post_rebal"]-df_2["pre_rebal_bal"]
    df_2["trad_trans"] = df_2["trad_trans"].fillna(0)
    # Calculate the projections for vintages (active and inactive logic differ).
    df_2["projection"] = df_2["pre_rebal_bal"] + df_2["trad_trans"]
    df_2["projection"] = df_2["projection"].fillna(0)

    # Calculate projections for df_1.
    if (baseline and date_df['order_of_ops_bl_ind'].unique()[0]) or (not baseline and date_df['order_of_ops_cr_ind'].unique()[0] and today <= date_df['proj_end_dt'].unique()[0]):
        df_1["projection"] = df_1["available_balance"] + df_1["total_trans_calc"] + df_1["proj_t381_rebal_cash"] + df_1["proj_t114_1"] + df_1["proj_t381_4"] + df_1["proj_t394_12"] + df_1["new_debit_t381_cash_trans"] + df_1["new_credit_t381_cash_trans"]
    else:
        df_1["projection"] = df_1["available_balance"] + df_1["total_trans_calc"] + df_1["proj_t381_rebal_cash"]
    df_1["projection"] = df_1["projection"].fillna(0)
    
    # Separate non-trad tickers.
    df_4 = df_1[~df_1["ic220_priceid"].isin(TRAD_TICKERS)]
    # Set the pin-plan-subplans based on the participant level input DataFrame.
    df_1["pin"] = date_df["pin"].unique()[0]
    df_1["plan"] = date_df["plan"].unique()[0]
    df_1["sub_plan"] = date_df["sub_plan"].unique()[0]
    df_1["scenario_id"] = date_df["scenario_id"].unique()[0]
    df_1["calendar_dates"] = date_df["calendar_dates"].unique()[0]
    df_2["pin"] = date_df["pin"].unique()[0]
    df_2["plan"] = date_df["plan"].unique()[0]
    df_2["sub_plan"] = date_df["sub_plan"].unique()[0]
    df_2["scenario_id"] = date_df["scenario_id"].unique()[0]
    
    # Recombine trad and non-trad DataFrames.
    df_5 = pd.concat((df_2[["ic220_priceid","source","contract_vintage","projection"]], df_4[["ic220_priceid","source","projection"]]), axis=0)
    date_df = pd.merge(date_df, df_5, how='left', on=["source", "ic220_priceid", "contract_vintage"])

    return date_df, df_2, df_1, residual_chk, accrued_div_calc, rebal_cash_hist, two_step_rebal_chk, t444_4_date

def projection_else(date_df, partic_proj, partic_proj_non_trad, today, yest, read_in, baseline, t444_4_date):
    """Calculates the daily projection when there is not a rebalancing event.

    This function is called when it is not the first day of the projection, and
    when there is not a rebalancing event occuring. Since there is a separate
    treatment between trad and non-trad assets, these tickers are separated and
    their debugging data is tracked separately in the trad and non-trad debug
    tables respectively. Throughout the function, there are multiple steps of
    logic that are applied in order. This is labeled clearly with inline
    comments, and many sections will also output their own debug tables that
    are eventually exported with the primary projection results.

    Args:
        date_df: The BTR data filtered on the specific partic_id and current
          date of the projection.
        partic_proj: The partic_id level projection results from the results
          dictionary. This is passed as a separate argument so the original
          date_df DataFrame can remain unaltered outside of the scope of the
          projection functions.
        partic_proj_non_trad: The partic_id level projection results for 
          non-trad assets from the results dictionary. This is passed as a
          separate argument so the original date_df DataFrame can remain
          unaltered outside of the scope of the projection functions.
        today: The current date of the projection in YYYY-MM-DD fomat.
        yest: The current date of the projection -1 in YYYY-MM-DD fomat.
        read_in: Boolean value specifying if we are reading in rebalancing data
          or calculating it live.
        baseline: Boolean value specifying if the run is baseline or corrected.
        t444_4_date: An indicator that is stored and updates for when a t444_4
          debit transaction is encountered. This is to keep track of the most
          recent transaction date throughout the projection which is needed for
          properly applying the accrued dividends logic.
    
    Returns:
        A tuple of DataFrames corresponding to main projection output, and extra
        debugging table output.
            date_df: Contains the prepped BTR data, but also now will constantly
              update with the new fields pertaining to the projection.
            df_2: Contains the separated projection data for trad assets. This
              is recombined into the date_df final output in the function but is
              also kept separate for debugging due to different treatment of
              trad and non-trad assets.
            df_1: Contains the separated projection data for non-trad assets.
              This is recombined into the date_df final output in the function
              but is also kept separate for debugging due to separate treatment
              of trad and non-trad assets.
            residual_chk: A DataFrame that checks for residual cash between the
              calculated final balance and the sum of total cash transactions
              and available balance.
            accrued_div_calc: Debugging table that tracks the data throughout
              the accrued dividends logic.
            rebal_cash_hist: Debugging table that tracks the rebalancing cash
              history thoughout the key fund logic and cash rebalancing process.
            two_step_rebal_chk: Debugging table that tracks the data througout
              the two step rebalancing logic.
            t444_4_date: An indicator that is stored and updates for when a
              t444_4 debit transaction is encountered. This is to keep track of
              the most recent transaction date throughout the projection which
              is needed for properly applying the accrued dividends logic.
    """
    # Separate trad and non-trad tickers from the participant DataFrame.
    is_aait_trad = date_df["ic220_priceid"].isin(TRAD_TICKERS)
    non_trad_df_1 = date_df.loc[~is_aait_trad, ["ic220_priceid","source","prev_day_proj","balance", "t114_1", "t381_4", "t394_12", "cash_sum", "debit_t381_cash_trans", "credit_t381_cash_trans"]].groupby(["ic220_priceid","source"]).agg('sum').reset_index()
    non_trad_df_2 = date_df.loc[~is_aait_trad, ["ic220_priceid","t815_pct"]].drop_duplicates()
    non_trad_df_2 = pd.merge(non_trad_df_1,non_trad_df_2, how='left', on="ic220_priceid")
    non_trad_df_2 = pd.merge(non_trad_df_2,date_df.loc[~is_aait_trad, ["source","ic220_priceid","fee_rate"]].drop_duplicates(), how='left', on=["source", "ic220_priceid"])
    
    trad_df_1 = date_df.loc[is_aait_trad, ["ic220_priceid","source","prev_day_proj","balance"]].groupby(["ic220_priceid","source"]).agg('sum').reset_index()
    trad_df_2 = date_df.loc[is_aait_trad, ["ic220_priceid","source","t114_1", "t381_4", "t394_12", "cash_sum","t815_pct","fee_rate", "debit_t381_cash_trans", "credit_t381_cash_trans"]].drop_duplicates()
    trad_df_2 = pd.merge(trad_df_1,trad_df_2, how='left', on=["ic220_priceid","source"])
    # Join the separated versions and set the t815 and t381 calcs to 0.
    df_1 = pd.concat((non_trad_df_2,trad_df_2), axis=0)

    t815_read_in_ind = date_df["t815_read_in_ind"].unique()[0]
    
    if (read_in == False and baseline == True and t815_read_in_ind == 1) or (read_in == False and baseline == False and today >= date_df['proj_end_dt'].unique()[0]) or date_df['unsub_ind'].unique()[0] == 1:
        df_1["t381_calc"] = 0
        df_1["t815_calc"] = df_1["t815_pct"]
        df_1["t815_read_in_ind"] = t815_read_in_ind
    else:
        df_1["t815_calc"] = partic_proj_non_trad.loc[partic_proj_non_trad["calendar_dates"]==yest,["t815_calc"]].values
        df_1["t815_read_in_ind"] = partic_proj_non_trad.loc[partic_proj_non_trad["calendar_dates"]==yest,["t815_read_in_ind"]].values

    df_1 = pd.merge(df_1, df_1[["source","t114_1", "t381_4", "t394_12"]].groupby("source").agg('sum').reset_index(), suffixes=("","_total"), how='left', on='source')
    
    if read_in:
        df_1["proj_t114_1"] = df_1["t114_1_total"] * df_1["t815_pct"]
        df_1["proj_t381_4"] = df_1["t381_4_total"] * df_1["t815_pct"]
        df_1["proj_t394_12"] = df_1["t394_12_total"] * df_1["t815_pct"]
    elif read_in == False and baseline == False and today > date_df['proj_end_dt'].unique()[0]:
        df_1["proj_t114_1"] = 0
        df_1["proj_t381_4"] = 0
        df_1["proj_t394_12"] = 0
    else:
        df_1["proj_t114_1"] = df_1["t114_1_total"]*df_1["t815_calc"]
        df_1["proj_t381_4"] = df_1["t381_4_total"] * df_1["t815_calc"]
        df_1["proj_t394_12"] = df_1["t394_12_total"] * df_1["t815_calc"]

    ## START ACCRUED DIVIDENDS LOGIC ##
    if date_df['t444_debit_ind'].drop_duplicates().reset_index(drop=True)[0] == 1:
        t444_4_date = today
        accrued_div_calc = partic_proj.loc[
            (
                partic_proj['calendar_dates'].apply(lambda x: x.month) ==
                date_df['calendar_dates'].drop_duplicates().reset_index(drop=True)[0].month
            ) &
            (
                partic_proj['calendar_dates'].apply(lambda x: x.year) ==
                date_df['calendar_dates'].drop_duplicates().reset_index(drop=True)[0].year
            ) &
            (
                partic_proj['calendar_dates'].apply(lambda x: x.day) <=
                date_df['calendar_dates'].drop_duplicates().reset_index(drop=True)[0].day
            ),
            [
                'pin',
                'plan',
                'sub_plan',
                'calendar_dates',
                'source',
                'ic220_priceid',
                'contract_vintage',
                'lead_transformed_rate_price',
                'projection',
            ]
        ]
        accrued_div_calc['div_calc'] = accrued_div_calc['lead_transformed_rate_price'] * accrued_div_calc['projection']
        df_1 = pd.merge(
            df_1,
            accrued_div_calc[['source', 'ic220_priceid', 'div_calc']].groupby(['source', 'ic220_priceid']).agg('sum').reset_index(),
            how='left',
            on=['source', 'ic220_priceid'],
        )
        df_1['div_calc'] = df_1['div_calc'].fillna(0)
    
    elif date_df['month_last_working_dt_ind'].drop_duplicates().reset_index(drop=True)[0] == 1:
        if t444_4_date != 0:
            if t444_4_date.year == today.year and t444_4_date.month == today.month:
                accrued_div_calc = partic_proj.loc[
                    (
                        partic_proj['calendar_dates'].apply(lambda x: x.month) ==
                        date_df['calendar_dates'].drop_duplicates().reset_index(drop=True)[0].month
                    ) &
                    (
                        partic_proj['calendar_dates'].apply(lambda x: x.year) ==
                        date_df['calendar_dates'].drop_duplicates().reset_index(drop=True)[0].year
                    ) &
                    (
                        partic_proj['calendar_dates'].apply(lambda x: x.year) > t444_4_date.day
                    ),
                    [
                        'pin',
                        'plan',
                        'sub_plan',
                        'calendar_dates',
                        'source',
                        'ic220_priceid',
                        'contract_vintage',
                        'lead_transformed_rate_price',
                        'projection',
                    ]
                ]
            else:
                accrued_div_calc = partic_proj.loc[
                    (
                        partic_proj['calendar_dates'].apply(lambda x: x.month) ==
                        date_df['calendar_dates'].drop_duplicates().reset_index(drop=True)[0].month
                    ) &
                    (
                        partic_proj['calendar_dates'].apply(lambda x: x.year) ==
                        date_df['calendar_dates'].drop_duplicates().reset_index(drop=True)[0].year
                    ),
                    [
                        'pin',
                        'plan',
                        'sub_plan',
                        'calendar_dates',
                        'source',
                        'ic220_priceid',
                        'contract_vintage',
                        'lead_transformed_rate_price',
                        'projection',
                    ]
                ]
        else:
            accrued_div_calc = partic_proj.loc[
                (
                    partic_proj['calendar_dates'].apply(lambda x: x.month) ==
                    date_df['calendar_dates'].drop_duplicates().reset_index(drop=True)[0].month
                ) &
                (
                    partic_proj['calendar_dates'].apply(lambda x: x.year) ==
                    date_df['calendar_dates'].drop_duplicates().reset_index(drop=True)[0].year
                ),
                [
                    'pin',
                    'plan',
                    'sub_plan',
                    'calendar_dates',
                    'source',
                    'ic220_priceid',
                    'contract_vintage',
                    'lead_transformed_rate_price',
                    'projection',
                ]
            ]

        accrued_div_calc['div_calc'] = accrued_div_calc['lead_transformed_rate_price'] * accrued_div_calc['projection']
        df_1 = pd.merge(
            df_1,
            accrued_div_calc[['source', 'ic220_priceid', 'div_calc']].groupby(['source', 'ic220_priceid']).agg('sum').reset_index(),
            how='left',
            on=['source', 'ic220_priceid'],
        )
        df_1['div_calc'] = df_1['div_calc'].fillna(0)
    
    else:
        accrued_div_calc = pd.DataFrame()
        df_1['div_calc'] = 0
    ## END ACCRUED DIVIDENDS LOGIC ##

    ## START FEE LOGIC ##
    if date_df["fee_dt_ind"].max() == 1:
        df_1 = apply_fees(df_1, date_df, partic_proj)

    else:
        df_1['proj_fee_amt'] = 0
        df_1['proj_fee_cash'] = 0
    ## END FEE LOGIC ##

    # NEW TRANSACTION LOGIC START #
    date_df["available_balance"] = date_df["prev_day_proj"]*date_df["return_perc"]

    df_1 = pd.merge(df_1,date_df[["source","ic220_priceid","available_balance"]].groupby(["source","ic220_priceid"]).agg('sum').reset_index(), how='left', on=["source","ic220_priceid"])

    if date_df["t381_cash_trans_ind"].unique()[0] == 1:
        t381_cash_trans_df = query_t381_cash_trans_df(date_df["pin"].unique()[0], date_df["plan"].unique()[0], date_df["sub_plan"].unique()[0], today)
        t381_cash_trans_df = pd.merge(t381_cash_trans_df, df_1[["ic220_priceid", "source", "available_balance","debit_t381_cash_trans"]], left_on = ["debit_ticker", "source"], right_on = ["ic220_priceid", "source"], how='left')
        t381_cash_trans_df = pd.merge(t381_cash_trans_df, df_1[["ic220_priceid", "source", "credit_t381_cash_trans"]], left_on = ["credit_ticker", "source"], right_on = ["ic220_priceid", "source"], how='left')
        t381_cash_trans_df["t381_debit_capping_chk"] = np.where(
            t381_cash_trans_df["debit_t381_cash_trans"] != 0,
            np.where(t381_cash_trans_df["debit_t381_cash_trans"] > t381_cash_trans_df["available_balance"], 1, 0 ),
            0
        )
        t381_cash_trans_df["new_debit_t381_cash_trans"] = np.where(t381_cash_trans_df["t381_debit_capping_chk"] == 1, t381_cash_trans_df["available_balance"], t381_cash_trans_df["debit_t381_cash_trans"])
        t381_cash_trans_df["credit_t381_cash_trans_adjustment"] = np.where(t381_cash_trans_df["t381_debit_capping_chk"]==1, t381_cash_trans_df["available_balance"] / t381_cash_trans_df["debit_t381_cash_trans"], 1)
        t381_cash_trans_df["new_credit_t381_cash_trans"] = t381_cash_trans_df["credit_t381_cash_trans"]*t381_cash_trans_df["credit_t381_cash_trans_adjustment"]

        df_1 = pd.merge(df_1, t381_cash_trans_df[["credit_ticker", "source","new_credit_t381_cash_trans"]].drop_duplicates(), left_on = ["ic220_priceid", "source"], right_on = ["credit_ticker", "source"], how='left')
        df_1 = pd.merge(df_1, t381_cash_trans_df[["debit_ticker","source","new_debit_t381_cash_trans"]].drop_duplicates(), left_on = ["ic220_priceid", "source"], right_on = ["debit_ticker", "source"], how='left')
        df_1["new_credit_t381_cash_trans"] = df_1["new_credit_t381_cash_trans"].fillna(0)
        df_1["new_debit_t381_cash_trans"] = df_1["new_debit_t381_cash_trans"].fillna(0)
    else:
        df_1["new_credit_t381_cash_trans"] = 0
        df_1["new_debit_t381_cash_trans"] = 0

    if read_in == False and baseline == False and today > date_df['proj_end_dt'].unique()[0]:
        df_1["total_trans_cash"] = df_1['div_calc'] + df_1["proj_fee_cash"]
    else:
        if (baseline and date_df['order_of_ops_bl_ind'].unique()[0]) or (not baseline and date_df['order_of_ops_cr_ind'].unique()[0]):
            df_1["total_trans_cash"] = df_1['div_calc'] + df_1["proj_fee_cash"] + df_1["cash_sum"]
        else:
            df_1["total_trans_cash"] = df_1["proj_t114_1"] + df_1["proj_t381_4"] + df_1["proj_t394_12"] + df_1['div_calc'] + df_1["proj_fee_cash"] + df_1["cash_sum"] + df_1["new_debit_t381_cash_trans"] + df_1["new_credit_t381_cash_trans"]
    df_1["residual"] = df_1["available_balance"] + df_1["total_trans_cash"]
    df_1["neg_proj_ind"] = np.where(df_1["residual"]<0, 1, 0)
    df_1["pos_proj_ind"] = 1-df_1["neg_proj_ind"]
    df_1['pos_residual'] = df_1["pos_proj_ind"]*df_1["residual"]
    df_1['neg_residual'] = df_1["neg_proj_ind"]*df_1["residual"]
    df_1 = pd.merge(df_1, df_1[["source","pos_residual","neg_residual"]].groupby("source").agg('sum').reset_index(), suffixes=("","_src_total"), how='left', on='source')
    df_1["positive_pct"] = df_1['pos_residual']/df_1['pos_residual_src_total']
    df_1["redist_neg"] = df_1['neg_residual_src_total']*df_1["positive_pct"]
    df_1["final_bal"] = np.where((df_1["residual"]+df_1["redist_neg"])<0, 0, df_1["residual"]+df_1["redist_neg"])
    df_1["final_bal"] = df_1["final_bal"].fillna(0)
    df_1["final_bal_neg_proj_chk"] = df_1["final_bal"].copy()

    df_1["total_trans_calc"] = df_1["final_bal"] - df_1["available_balance"]

    residual_chk = df_1[["source","available_balance","final_bal","total_trans_cash"]].groupby("source").agg('sum').reset_index()
    residual_chk["trans_residual"] = residual_chk["available_balance"] + residual_chk["total_trans_cash"] - residual_chk["final_bal"]
    ## NEW TRANSACTION LOGIC END ##

    # Calculate cash projections.
    df_1["final_bal_total"] = 0 
    df_1["proj_t381_cash"] = 0
    df_1["proj_t381_rebal_cash"] = 0
    # Calculate all transaction cash for trad only.
    if (baseline and date_df['order_of_ops_bl_ind'].unique()[0]) or (not baseline and date_df['order_of_ops_cr_ind'].unique()[0] and today <= date_df['proj_end_dt'].unique()[0]):
        df_1["aait_trad_trans_cash"] = np.where(df_1["ic220_priceid"].isin(TRAD_TICKERS), 1, 0) * (df_1["total_trans_calc"] + df_1["proj_t114_1"] + df_1["proj_t381_4"] + df_1["proj_t394_12"] + df_1["new_debit_t381_cash_trans"] + df_1["new_credit_t381_cash_trans"])
    else:
        df_1["aait_trad_trans_cash"] = np.where(df_1["ic220_priceid"].isin(TRAD_TICKERS), 1, 0) * df_1["total_trans_calc"]

    # Separate the active vintage from all other vintages and agg sum on source.
    df_2 = date_df.loc[date_df["ic220_priceid"].isin(TRAD_TICKERS),["calendar_dates","ic220_priceid","source","balance","return_perc","contract_vintage","contract_open","contract_close","prev_day_proj", "available_balance"]]
    df_2["active_vintage"] = np.where((df_2["calendar_dates"]>=df_2["contract_open"]) & (df_2["calendar_dates"]<=df_2["contract_close"]),1,0)
    df_2 = pd.merge(df_2, df_1[["source","aait_trad_trans_cash"]].groupby("source").agg('sum').reset_index(), how='left', on='source')
    df_2 = pd.merge(df_2, df_2[["source","prev_day_proj"]].groupby("source").agg('sum').reset_index(), suffixes=("","_src_total"), how='left', on='source')

    # Set the sell and buy values for trad.
    df_3 = df_1.loc[df_1["ic220_priceid"].isin(TRAD_TICKERS),["source","proj_t381_rebal_cash"]].drop_duplicates()
    df_3["sell"] = np.where(df_3["proj_t381_rebal_cash"]<0, df_3["proj_t381_rebal_cash"], 0)
    df_3["buy"] = np.where(df_3["proj_t381_rebal_cash"]>0, df_3["proj_t381_rebal_cash"], 0)

    # Apply excel logic to get prorate and new vintage values.
    if df_3["buy"].sum()<=df_3["sell"].sum()*-1:
        df_3["prorate"] = df_3["buy"]+df_3["sell"]
        df_3["new_vintage"] = 0
    else:
        df_3["prorate"] = df_3["sell"] + ((df_3["buy"]/df_3["buy"].sum())*df_3["sell"].sum()*-1)
        df_3["new_vintage"] = (df_3["buy"]/df_3["buy"].sum())*(df_3["sell"].sum()+df_3["buy"].sum())

    # Join calculated values and aggregate sum by source.
    df_2 = pd.merge(df_2, df_3[["source","buy","sell","prorate","new_vintage"]], how='left', on='source')
    df_2 = pd.merge(df_2, df_2[["source", "available_balance"]].groupby('source').agg('sum').reset_index(), how='left', on='source', suffixes=("","_src_total"))
    df_2["pre_rebal_bal"] = np.where(
        df_2["aait_trad_trans_cash"] >= 0,
        np.where(
            df_2["active_vintage"]==1,
            (df_2["prev_day_proj"]*df_2["return_perc"]) + df_2["aait_trad_trans_cash"],
            df_2["prev_day_proj"]*df_2["return_perc"]
        ),
        (df_2["prev_day_proj"]*df_2["return_perc"]) + df_2["aait_trad_trans_cash"]*(df_2["available_balance"]/df_2["available_balance_src_total"])
    )
    df_2 = pd.merge(df_2, df_2[["source","pre_rebal_bal"]].groupby("source").agg('sum').reset_index(), suffixes=("","_src_sum"), how='left', on="source")

    # Calculate prorate out and aggregate sum by vintage.
    df_2["prorate_out"] = np.where(df_2["prorate"]<0, ((df_2["pre_rebal_bal_src_sum"]+df_2["prorate"])/df_2["pre_rebal_bal_src_sum"])*df_2["pre_rebal_bal"], df_2["pre_rebal_bal"] )
    df_2 = pd.merge(df_2, df_2[["contract_vintage","pre_rebal_bal","prorate_out"]].groupby(["contract_vintage"]).agg('sum').reset_index(), suffixes=("","_vint_sum"), how='left', on="contract_vintage")

    # Calculate prorate in and calculate the trad transaction amount.
    df_2["prorate_in"] = np.where(df_2["prorate"]>0, df_2["prorate_out"] + (df_2["prorate"]/df_3["sell"].sum()*-1) * (df_2["pre_rebal_bal_vint_sum"]-df_2["prorate_out_vint_sum"]), df_2["prorate_out"] )
    df_2["new_vintage_post_rebal"] = df_2["prorate_in"]+ (df_2["new_vintage"]*df_2["active_vintage"])
    df_2["trad_trans"] = df_2["new_vintage_post_rebal"]-df_2["pre_rebal_bal"]
    df_2["trad_trans"] = df_2["trad_trans"].fillna(0)
    # Calculate the projections for vintages (active and inactive logic differ).
    df_2["projection"] = df_2["pre_rebal_bal"] + df_2["trad_trans"]
    df_2["projection"] = df_2["projection"].fillna(0)

    # Calculate projections for df_1.
    if (baseline and date_df['order_of_ops_bl_ind'].unique()[0]) or (not baseline and date_df['order_of_ops_cr_ind'].unique()[0] and today <= date_df['proj_end_dt'].unique()[0]):
        df_1["projection"] = df_1["available_balance"] + df_1["total_trans_calc"] + df_1["proj_t381_rebal_cash"] + df_1["proj_t114_1"] + df_1["proj_t381_4"] + df_1["proj_t394_12"] + df_1["new_debit_t381_cash_trans"] + df_1["new_credit_t381_cash_trans"]
    else:
        df_1["projection"] = df_1["available_balance"] + df_1["total_trans_calc"] + df_1["proj_t381_rebal_cash"]
    df_1["projection"] = df_1["projection"].fillna(0)

    # Separate non-trad tickers.
    df_4 = df_1[~df_1["ic220_priceid"].isin(TRAD_TICKERS)]
    
    # Set the pin-plan-subplans based on the participant level input DataFrame.
    df_1["pin"] = date_df["pin"].unique()[0]
    df_1["plan"] = date_df["plan"].unique()[0]
    df_1["sub_plan"] = date_df["sub_plan"].unique()[0]
    df_1["scenario_id"] = date_df["scenario_id"].unique()[0]
    df_1["calendar_dates"] = date_df["calendar_dates"].unique()[0]
    df_2["pin"] = date_df["pin"].unique()[0]
    df_2["plan"] = date_df["plan"].unique()[0]
    df_2["sub_plan"] = date_df["sub_plan"].unique()[0]
    df_2["scenario_id"] = date_df["scenario_id"].unique()[0]

    # Recombine trad and non-trad DataFrames.
    df_5 = pd.concat((df_2[["ic220_priceid","source","contract_vintage","projection"]], df_4[["ic220_priceid","source","projection"]]), axis=0)
    date_df = pd.merge(date_df, df_5, how='left', on=["ic220_priceid","source","contract_vintage"])

    return date_df, df_2, df_1, residual_chk, accrued_div_calc, pd.DataFrame(), pd.DataFrame(), t444_4_date

def projection_start(date_df):
    """Calculates the daily projection on the first day of the projection.

    This function is called only on the first day of the projection. Its purpose
    is to instance the necessary columns that are needed for the projection to
    occur. These initial instances are usually 0 or null. This is also why the
    BTR data is prepared 1 day before the first model misalignment rebalancing
    event occurs. Thus, we also always expect the following day of the
    projection to be a rebalancing event and call the projection_rebal function.

    Args:
        date_df: The BTR data filtered on the specific partic_id and current
          date of the projection.
    
    Returns:
        A tuple of DataFrames corresponding to main projection output, and extra
        debugging table output.
            date_df: Contains the prepped BTR data, but also now will constantly
              update with the new fields pertaining to the projection.
            df_2: Contains the separated projection data for trad assets. This
              is recombined into the date_df final output in the function but is
              also kept separate for debugging due to different treatment of
              trad and non-trad assets.
            df_1: Contains the separated projection data for non-trad assets.
              This is recombined into the date_df final output in the function
              but is also kept separate for debugging due to separate treatment
              of trad and non-trad assets.
            residual_chk: A DataFrame that checks for residual cash between the
              calculated final balance and the sum of total cash transactions
              and available balance.
            accrued_div_calc: Debugging table that tracks the data throughout
              the accrued dividends logic.
            rebal_cash_hist: Debugging table that tracks the rebalancing cash
              history thoughout the key fund logic and cash rebalancing process.
            two_step_rebal_chk: Debugging table that tracks the data througout
              the two step rebalancing logic.
            t444_4_date: An indicator that is stored and updates for when a
              t444_4 debit transaction is encountered. This is to keep track of
              the most recent transaction date throughout the projection which
              is needed for properly applying the accrued dividends logic.
    """
    # Separate trad and non-trad tickers from the participant DataFrame.
    is_aait_trad = date_df["ic220_priceid"].isin(TRAD_TICKERS)
    non_trad_df_1 = date_df.loc[~is_aait_trad, ["ic220_priceid","source","prev_day_proj","balance","t114_1", "t381_4", "t394_12", "cash_sum"]].groupby(["ic220_priceid","source"]).agg('sum').reset_index()
    non_trad_df_2 = date_df.loc[~is_aait_trad, ["ic220_priceid","t815_pct"]].drop_duplicates()
    non_trad_df_2 = pd.merge(non_trad_df_1,non_trad_df_2, how='left', on="ic220_priceid")
    non_trad_df_2 = pd.merge(non_trad_df_2,date_df.loc[~is_aait_trad, ["source","ic220_priceid","fee_rate"]].drop_duplicates(), how='left', on=["source", "ic220_priceid"])
    
    trad_df_1 = date_df.loc[is_aait_trad, ["ic220_priceid","source","prev_day_proj","balance"]].groupby(["ic220_priceid","source"]).agg('sum').reset_index()
    trad_df_2 = date_df.loc[is_aait_trad, ["ic220_priceid","source","t114_1", "t381_4", "t394_12", "cash_sum","t815_pct", "fee_rate"]].drop_duplicates()
    trad_df_2 = pd.merge(trad_df_1,trad_df_2, how='left', on=["ic220_priceid","source"])
    # Join the separated versions and set the t815 and t381 calcs to 0.
    df_1 = pd.concat((non_trad_df_2,trad_df_2), axis=0)
    df_1["t381_calc"] = 0
    df_1["t815_calc"] = 0
    
    df_1 = pd.merge(df_1, df_1[["source","t114_1", "t381_4", "t394_12"]].groupby("source").agg('sum').reset_index(), suffixes=("","_total"), how='left', on='source')
    df_1["proj_t114_1"] = 0
    df_1["proj_t381_4"] = 0
    df_1["proj_t394_12"] = 0
    df_1['div_calc'] = 0

    # Set transaction fee cash to 0.
    df_1['proj_fee_amt'] = 0
    df_1['proj_fee_cash'] = 0

    # NEW TRANSACTION LOGIC START #
    date_df["available_balance"] = 0

    df_1 = pd.merge(df_1,date_df[["source","ic220_priceid","available_balance"]].groupby(["source","ic220_priceid"]).agg('sum').reset_index(), how='left', on=["source","ic220_priceid"])

    df_1["new_credit_t381_cash_trans"] = 0
    df_1["new_debit_t381_cash_trans"] = 0

    df_1["total_trans_cash"] =  0
    df_1["residual"] = 0
    df_1["neg_proj_ind"] = 0
    df_1["pos_proj_ind"] = 0
    df_1['pos_residual'] = 0
    df_1['neg_residual'] = 0
    df_1 = pd.merge(df_1, df_1[["source","pos_residual","neg_residual"]].groupby("source").agg('sum').reset_index(), suffixes=("","_src_total"), how='left', on='source')
    df_1["positive_pct"] = 0
    df_1["redist_neg"] = 0
    df_1["final_bal"] = 0
    df_1["final_bal_neg_proj_chk"] = 0

    df_1["total_trans_calc"] = 0

    residual_chk = df_1[["source","available_balance","final_bal","total_trans_cash"]].groupby("source").agg('sum').reset_index()
    residual_chk["trans_residual"] = 0
    ## NEW TRANSACTION LOGIC END ##

    # Calculate cash projections.
    df_1["final_bal_total"] = 0 
    df_1["proj_t381_cash"] = 0
    df_1["proj_t381_rebal_cash"] = 0
    df_1["aait_trad_trans_cash"] = 0

    # Separate the active vintage from all other vintages and agg sum on source.
    df_2 = date_df.loc[date_df["ic220_priceid"].isin(TRAD_TICKERS),["calendar_dates","ic220_priceid","source","balance","return_perc","contract_vintage","contract_open","contract_close","prev_day_proj", "available_balance"]]
    df_2["active_vintage"] = np.where((df_2["calendar_dates"]>=df_2["contract_open"]) & (df_2["calendar_dates"]<=df_2["contract_close"]),1,0)
    df_2 = pd.merge(df_2, df_1[["source","aait_trad_trans_cash"]].groupby("source").agg('sum').reset_index(), how='left', on='source')
    df_2 = pd.merge(df_2, df_2[["source","prev_day_proj"]].groupby("source").agg('sum').reset_index(), suffixes=("","_src_total"), how='left', on='source')

    # Set the sell and buy values for trad.
    df_3 = df_1.loc[df_1["ic220_priceid"].isin(TRAD_TICKERS),["source","proj_t381_rebal_cash"]].drop_duplicates()
    df_3["sell"] = 0
    df_3["buy"] = 0

    # Apply excel logic to get prorate and new vintage values.
    if df_3["buy"].sum()<=df_3["sell"].sum()*-1:
        df_3["prorate"] = 0
        df_3["new_vintage"] = 0
    else:
        df_3["prorate"] = 0
        df_3["new_vintage"] = 0

    # Join calculated values and aggregate sum by source.
    df_2 = pd.merge(df_2, df_3[["source","buy","sell","prorate","new_vintage"]], how='left', on='source')
    df_2["pre_rebal_bal"] = 0
    df_2 = pd.merge(df_2, df_2[["source","pre_rebal_bal"]].groupby("source").agg('sum').reset_index(), suffixes=("","_src_sum"), how='left', on="source")

    # Calculate prorate out and aggregate sum by vintage.
    df_2["prorate_out"] = 0
    df_2 = pd.merge(df_2, df_2[["contract_vintage","pre_rebal_bal","prorate_out"]].groupby(["contract_vintage"]).agg('sum').reset_index(), suffixes=("","_vint_sum"), how='left', on="contract_vintage")

    # Calculate prorate in and calculate the trad transaction amount.
    df_2["prorate_in"] = 0
    df_2["new_vintage_post_rebal"] = 0
    df_2["trad_trans"] = 0

    # Calculate projections for df_2.
    df_2["projection"] = df_2["balance"]
    df_2["projection"] = df_2["projection"].fillna(0)

    # Calculate projections for df_1.
    df_1["total_trans_calc"] = 0
    df_1["projection"] = df_1["balance"]
    df_1["projection"] = df_1["projection"].fillna(0)

    # Separate non-trad tickers.
    df_4 = df_1[~df_1["ic220_priceid"].isin(TRAD_TICKERS)]
    
    # Set the pin-plan-subplans based on the participant level input DataFrame.
    df_1["pin"] = date_df["pin"].unique()[0]
    df_1["plan"] = date_df["plan"].unique()[0]
    df_1["sub_plan"] = date_df["sub_plan"].unique()[0]
    df_1["scenario_id"] = date_df["scenario_id"].unique()[0]
    df_1["calendar_dates"] = date_df["calendar_dates"].unique()[0]
    df_1["t815_read_in_ind"] = 0
    df_1["recalc_t381_pct"] = 0
    df_1["t381_rebal_cash"] = 0
    df_2["pin"] = date_df["pin"].unique()[0]
    df_2["plan"] = date_df["plan"].unique()[0]
    df_2["sub_plan"] = date_df["sub_plan"].unique()[0]
    df_2["scenario_id"] = date_df["scenario_id"].unique()[0]
    
    # Recombine trad and non-trad DataFrames.
    df_5 = pd.concat((df_2[["ic220_priceid","source","contract_vintage","projection"]], df_4[["ic220_priceid","source","projection"]]), axis=0)
    date_df = pd.merge(date_df, df_5, how='left', on=["ic220_priceid","source","contract_vintage"])

    return date_df, df_2, df_1, residual_chk, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def log_pin_error(pin, plan, subplan, date, sid, arg):
    """Creates a log for a PIN that raises an error during projections.

    Args:
        pin: The participant id for the current projection.
        plan: The plan id for the current projection.
        subplan: The subplan id for the current projection.
        date: The current date of the current projection.
        sid: The scenario_id of the current projection.
        arg: The error raised during the PIN's projection.
    
    Returns:
        A DataFrame containing all of the input data. This error data is
        appended to the error_table DataFrame and all errored PINs are
        eventually exported and uploaded as metadata tables along with all
        other results.
    """
    error_msg = (f'[ERROR]: Pin: {pin} | Plan: {plan} | Subplan: {subplan} | Date: {date} | Scenario: {sid} | Error = {arg}')
    print(error_msg)
    log = open(ERROR_LOG_PATH + 'roll_fwd_error_log.txt', 'a+')
    log.write(error_msg)
    log.close()

    return pd.DataFrame({
        'pin': [pin],
        'plan': [plan],
        'sub_plan': [subplan],
        'calendar_dates': [date],
        'scenario_id': [sid],
        'error': [str(arg)],
    })

def query_model_weight(id, sid, baseline):
    """Query the stg_model_weights_actual df from the SQL DB.

    Args:
        id: partic_id containing a concatination of pin, plan, and sub_plan.
        sid: The scenario_id of the current projection.
        baseline: Boolean value specifying if the run is baseline or corrected.
    
    Returns:
        A DataFrame of the stg_model_weights_actual df with the '#'
        sign removed from the trad ticker TICP1. This is filtered for each
        participant and each scenario.
    """
    if baseline:
        suffix = 'actual'
    else:
        suffix = 'correct'
    
    query = f'''
    select *
    from {SCHEMA}.stg_model_weights_{suffix}
    where concat(pin, [plan], sub_plan) = '{id}'
    and scenario_id = {sid}
    '''
    engine = sa.create_engine(CONN_URL)
    df = pd.read_sql(query, con=engine)
    df["fund_ticker"] = df.fund_ticker.str.replace("#","")

    return df

def query_ca_ticker_map(id, baseline):
    """Query the stg_fp_ca_legacy_balance df from the SQL DB.

    Args:
        id: The concat(pin, plan, sub_plan) lookup key to filter the ticker map
          for only on participant.
        baseline: Boolean value specifying if the run is baseline or corrected.
    
    Returns:
        A DataFrame of the stg_ca_legacy_balance df which will function
        as a considered assets ticker map.
    """
    if baseline:
        prefix = ''
    else:
        prefix = 'cr_'
    query = f'''
    select *
    from {SCHEMA}.stg_{prefix}ca_legacy_balance
    where concat(pin, [plan]) = '{id[:-6]}'
    '''
    engine = sa.create_engine(CONN_URL)
    df = pd.read_sql(query, con=engine)

    return df

def create_rebal_data(date_df):
    """Creates the rebal_data table from date_df for checking overallocations.

    Args:
        date_df: The BTR data filtered on the specific partic_id and current
          date of the projection.
    
    Returns:
        A DataFrame containing all the data needed for rebalancing. This will be
        passed into the over_allocation and rebalancing gate functions.
    """
    rebal_data = date_df[[
        "pin",
        "plan",
        "sub_plan",
        "ic220_priceid",
        "prev_day_proj",
        "calendar_dates"
    ]]
    rebal_data = (
        rebal_data
        .groupby([
            "pin",
            "plan",
            "sub_plan",
            "ic220_priceid",
            "calendar_dates"
        ])
        .agg('sum')
        .reset_index()
    )
    return rebal_data.drop_duplicates()

def update_rebal_gate_hist(date_df, rebal_data, rebalance_gate):
    """Updates the rebalancing gate history function.

    Throughout the projection there are certain days that contain rebalacing
    events. When those rebalancing events occur, we must identify how assets
    are overallocated and rebalance them accordingly. Therefore each rebalancing
    event calls one of the eight rebalancing gate functions, and produces t381
    and t815 values. This function creates and updates a master list of all
    rebalancing events throughout the projection.

    Args:
        date_df: The BTR data filtered on the specific partic_id and current
          date of the projection.
        rebal_data: DataFrame containing the results from the rebalancing gate
          calculation.
        rebalance_gate: String indicating which rebalancing gate was called
          during this rebalancing event.
        
    Returns:
        A DataFrame of the updated complete gate history table with the most
        recent rebalancing event.
    """
    columns = [
        "pin",
        "plan",
        "sub_plan",
        "calendar_dates",
        "ic220_priceid",
        "t381_calc",
        "t815_calc"
    ]
    df = rebal_data[columns].drop_duplicates()
    df = pd.merge(
        df,
        date_df[["ic220_priceid","t815_pct"]].drop_duplicates(),
        how='left',
        on='ic220_priceid',
    )
    df["rebalance_gate"] = rebalance_gate
    
    return df

def update_runtimes_dtypes(df):
    """Updates the runtimes table by assigning the correct columns and dtypes.

    This function casts all of the results values from the runtimes table to
    the proper data types and returns the modified table as a DataFrame.

    Args:
        df: DataFrame representing the runtimes table.
    
    Returns:
        The runtimes table with all of the correct column names and values cast
        into the correct data types.
    """
    df.columns = [
        'pin',
        'plan',
        'sub_plan',
        'scenario_id',
        'start_time',
        'end_time',
        'run_time',
        'days_projected',
    ]
    df['pin'] = df['pin'].astype('str')
    df['plan'] = df['plan'].astype('str')
    df['sub_plan'] = df['sub_plan'].astype('str')
    df['scenario_id'] = df['scenario_id'].astype('int')
    df['start_time'] = df['start_time'].astype('str')
    df['end_time'] = df['end_time'].astype('str')
    df['run_time'] = df['run_time'].astype('float')
    df['days_projected'] = df['days_projected'].astype('int')

    return df

def add_pps_date(pin, plan, subplan, date, sid, df):
    """Inserts the pin, plan, subplan, and date into a dataframe.

    This function inserts the current pin, plan, and subplan being projected
    as well as the current date of that projection as 4 separate columns. These
    columns are used for identifying which pin-plan-subplan was being projected,
    and when in the projection the table was produced for easier debugging.

    Args:
        pin: The participant id for the current projection.
        plan: The plan id for the current projection.
        subplan: The subplan id for the current projection.
        date: The current date of the current projection.
        sid: The scenario_id of the current projection.
        df: A DataFrame containing the debugging output for certain debugging
          tables; e.g., debit_amts or rebal_cash_hist.
    
    Returns:
        The same DataFrame with columns denoting the pin, plan, subplan, and
        date of the current projection.
    """
    df.insert(0, 'scenario_id', sid, allow_duplicates=True)
    df.insert(0, 'calendar_dates', date, allow_duplicates=True)
    df.insert(0, 'sub_plan', subplan, allow_duplicates=True)
    df.insert(0, 'plan', plan, allow_duplicates=True)
    df.insert(0, 'pin', pin, allow_duplicates=True)

    return df

def get_proj_type(baseline, read_in, drift, dynamic):
    """Creates a string prefix for each type of run for files and table names.

    Args:
        baseline: Boolean value specifying if the run is baseline or corrected.
        read_in: Boolean value specifying if we are reading in rebalancing data
          or calculating it live.
        drift: Boolean value specifying if the run will include drift
          rebalancing. (Implies read_in = False)
        dynamic: Boolean value specifying if the run will include dynamic
          rebalancing. (Implies read_in = False and drift = True)
    
    Returns:
        A prefix string specifying the run type.
    """
    if baseline and read_in: # Baseline and readin.
        proj_type = 'bla'
    elif baseline and dynamic: # Baseline and dynamic; no readin.
        proj_type = 'bld'
    elif baseline and drift: # Baseline and drift; no readin, no dynamic.
        proj_type = 'blc'
    elif baseline: # Baseline; no readin, no drift, no dynamix
        proj_type = 'blb'
    else:
        proj_type = 'cr'
    
    return proj_type

def df_attach_metadata(df=None, run_id=None, version=None):
	"""Attaches run_id and version metadata fields to a pandas DataFrame object.

    Args:
        df: pandas DataFrame object.
        run_id: Run identification number for run currently being processed.
        version: Version number of source code.

    Returns:
        A DataFrame object with the run_id and version fields added.
    """
	try:
		df.insert(loc=0, column="run_id", value=run_id)
		df.insert(loc=1, column="version", value=version)
	except ValueError:
		column_run_id = df.pop("run_id")
		column_version = df.pop("version")
		df.insert(loc=0, column="run_id", value=column_run_id)
		df.insert(loc=1, column="version", value=column_version)

	return df
    
def upload_results(proj_type):
    """Uploads the results tables from local storage to the SQL DB.

    This function will iterate through all of the separately exported partic_id
    level results in their respective folders and call the bcp command to
    upload them all into their resepective tables on the SQL database.

    Args:
        proj_type: A string prefix for each type of run for files and table
          names.
        
    Returns:
        None
    """
    table_end = get_filter_pins_suffix()
    
    for table in table_config.keys():
        column_dtypes = table_config[table]
        table_name = f'result_{proj_type}_{table}_{TABLE_SUFFIX}{table_end}'

        make_table(table_name, column_dtypes)
        all_paths = glob.glob(f'{RESULT_PATH}{SCHEMA}_{proj_type}_{TABLE_SUFFIX}{table_end}/{table}/*.txt')
        for path in all_paths:
            print(f'\n[UPLOADING]: {path}')
            bcp(table_name, path)

def query_filter_table(read_in, drift):
    """Queries distinct partic_ids, with balances, that are in the filter_table.

    This function queries and returns distinct partic_ids and their respective
    proj_start_dts from the stg_consolidated_btr table.

    Args:
        None

    Returns:
        A DataFrame of all partic_ids and proj_start_dts.
    """
    query = f'''
    select distinct partic_id, proj_start_dt
    from {SCHEMA}.stg_consolidated_btr
    where partic_id in (
        select concat(pin, [plan], sub_plan)
        from {SCHEMA}.{FILTER_TABLE}
    )
    '''
    if not read_in or drift:
        query += f'and base_contract_ind = 0 '
    if FILTER_PINS:
        pins = ''.join([f"'{s}', " for s in FILTER_PINS])[:-2]
        query += f'and partic_id in ({pins}) '
    query += f'order by proj_start_dt;'
    
    engine = sa.create_engine(CONN_URL)
    df = pd.read_sql(query, con=engine)

    return df

def query_scenario_ids(unique_ids):
    """Queries distinct partic_ids-scenario combinations from stg_scenario_ids.

    This function queries and returns all partic_ids and scenarios from the
    full list of unique partic_ids.

    Args:
        unique_ids: full list of unique partic_ids being projected.

    Returns:
        A DataFrame of all partic_ids and scenarios for those partic_ids.
    """
    query = f'''
    select partic_id, scenario_id
    from {SCHEMA}.stg_scenario_ids
    where partic_id
    '''
    if len(unique_ids) == 1:
        query += f" = '{unique_ids[0]}'"
    else:
        query += 'in ('
        for id in unique_ids:
            query += f"'{id}', "
        query = query[:-2] + ') '
    
    if FILTER_SCENARIOS:
        pins = ''.join([f"'{s}', " for s in FILTER_SCENARIOS])[:-2]
        query += f'and scenario_id in ({pins}) '
    
    query += 'order by partic_id, scenario_id;'
    engine = sa.create_engine(CONN_URL)
    df = pd.read_sql(query, con=engine)
    
    return df

def create_batches(n, unique_partic):
    """Creates a sprted set of lists of partic_ids to minimize unused core-time.

    This function takes an array of unique partic_ids that are sorted from
    longest projection time to shortest projection time.
    
    E.g., given an array: [12, 11, 10, 8, 8, 5, 3, 2, 1]. Assume each element
    represents a partic_id whose projection length is the value of that element.
    This list will always be passed to this function sorted from longest to
    shortest. Our output list will be [[12, 8], [11, 5], [10, 3], [8, 2, 1]].

    This is important because our multiprocessing function will calculate
    projections for each PIN in a batch before moving on to the next, but will
    idle when they are complete and there are no other PINs to process. If 10
    PINs need to be processed with 4 cores, and the last PIN has the longest
    projection time of all the 10 PINs, then the other 3 cores will idle when
    they complete their projections "waiting" for the final PIN to complete.
    Therefore, sorting the PINs into batches like this is designed to minimize
    waiting time at the end of batches.

    Args:
        n: Number of batches in the run.
        unique_partic: The list of unique partic_ids being processed.
    
    Returns:
        A list of lists of partic_ids where each inner list contains the set of
        PINs to be run in the batch (in that order), the outer list acts as an
        iterable for all batches.
    """
    jobs = len(unique_partic)
    batches = []
    for i in range(n):
        batches.append([
            unique_partic[n * j + i]
            for j in range(jobs // n)
        ])
    if (jobs % n) != 0:
        batches[-1].extend(unique_partic[-(jobs % n):])

    return batches

def query_btr(id):
    """Queries the stg_consolidated_btr table for one partic_id and scenario_id.

    This function queries and returns all data in the stg_consolidated_btr table
    and returns the data as a DataFrame.

    Args:
        id: partic_id containing a concatination of pin, plan, and sub_plan.
    
    Returns:
        A DataFrame of all the btr data corresponding to a pin-plan-subplan
        combination.
    """
    query = f'''
        select * from {SCHEMA}.stg_consolidated_btr where partic_id = '{id}';
    '''
    engine = sa.create_engine(CONN_URL)
    df = pd.read_sql(query, con=engine)
    df["balance"] = df["balance"].fillna(0)

    return df

def update_ctrl_projection_times(df, proj_type, upload_time):
    """Updates the ctrl_projection_times table at the end of the run.

    The ctrl_projection_times table is a metadata table that tracks the
    timestamps and total run times of different components of the run:
    read_time - the time it takes to query the input data from the SQL DB,
    proj_time - the time it takes to calculate the projections,
    write_time - the time it takes to export results from memory to storage.
    upload_time - the time it takes to upload from storage to the SQL DB.

    These times are calculated per-batch, and aggregate at the end of the run
    to provide a total time spent doing each category of processing.

    Args:
        df: the ctrl_projection_times DataFrame before the aggregated fields
          have been calculated and added.
        proj_type: A string prefix for each type of run for files and table
          names.
        upload_time: The calculated total upload time in minutes.
    
    Returns:
        None
    """
    run_id_query = f'''
    select run_id
    from {SCHEMA}.ctrl_projection_times
    order by run_id;
    '''
    engine = sa.create_engine(CONN_URL)
    run_id = int(pd.read_sql(run_id_query, con=engine).iloc[-1, 0])

    df['total_read_time'] = df['read_time'].sum()
    df['total_proj_time'] = df['proj_time'].sum()
    df['total_write_time'] = df['write_time'].sum()
    df['total_upload_time'] = upload_time
    
    df.insert(4, 'end_timestamp', datetime.now(), allow_duplicates=True)
    df.insert(0, 'run_type', proj_type, allow_duplicates=True)
    df.insert(0, 'version', VERSION, allow_duplicates=True)
    df.insert(0, 'run_id', run_id+1, allow_duplicates=True)

    print('[UPDATING]: ctrl_projection_times')
    df.to_sql(
        'ctrl_projection_times',
        con=CREDS.engine,
        index=False,
        schema=SCHEMA,
        if_exists='append'
    )

def clear_errored_results(results, today, e):
    """Log an errored PIN and reset its results tables to empty DataFrames
    
    Args:
        results: A dictionary containing metadata and results DataFrames for
          the PIN that raised the error.
        today: The current date of the projection in YYYY-MM-dd format.
        e: The exception raised by the PIN that was projecting.
    
    Returns:
        results DataFrame with it's results overwritten with empty DataFrames.
    """
    # Metadata tables
    results['error_table'] = log_pin_error(
        results['pin'],
        results['plan'],
        results['subplan'],
        today,
        results['sid'],
        e,
    )
    
    # Proj tables
    results['proj'] = pd.DataFrame()
    results['proj_trad'] = pd.DataFrame()
    results['proj_non_trad'] = pd.DataFrame()

    # Debugging tables
    results['debit_amts'] = pd.DataFrame()
    results['accrued_div_calc'] = pd.DataFrame()
    results['rebal_cash_hist'] = pd.DataFrame()
    results['two_step_rebal_chk'] = pd.DataFrame()

    # Rebalancing gate history tables
    results['g1_hist'] = pd.DataFrame()
    results['g2_hist'] = pd.DataFrame()
    results['g3_hist'] = pd.DataFrame()
    results['g4_hist'] = pd.DataFrame()
    results['g5_hist'] = pd.DataFrame()
    results['g6_hist'] = pd.DataFrame()
    results['g7_hist'] = pd.DataFrame()
    results['g8_hist'] = pd.DataFrame()
    results['gate_hist_complete'] = pd.DataFrame()

    return results

def add_sid_to_gate_hist(results):
    """Adds the scenario_id to the gate history tables.

    Args:
        results: A dictionary containing metadata and results DataFrames for
          the PIN that raised the error.
    
    Returns:
        results DataFrame with scenario_ids added to the gate history tables.
    """
    if not results['g1_hist'].empty:
        results['g1_hist']['scenario_id'] = results['sid']
    if not results['g2_hist'].empty:
        results['g2_hist']['scenario_id'] = results['sid']
    if not results['g3_hist'].empty:
        results['g3_hist']['scenario_id'] = results['sid']
    if not results['g4_hist'].empty:
        results['g4_hist']['scenario_id'] = results['sid']
    if not results['g5_hist'].empty:
        results['g5_hist']['scenario_id'] = results['sid']
    if not results['g6_hist'].empty:
        results['g6_hist']['scenario_id'] = results['sid']
    if not results['g7_hist'].empty:
        results['g7_hist']['scenario_id'] = results['sid']
    if not results['g8_hist'].empty:
        results['g8_hist']['scenario_id'] = results['sid']
    if not results['gate_hist_complete'].empty:
        results['gate_hist_complete']['scenario_id'] = results['sid']
    
    return results

def create_runtimes_table(results, start_time):
    """Creates a table for tracking the runtimes of each partic_id-scenario.

    At the beginning of each partic_id's projection a start time stamp is
    created. Once the projection is complete this function is called so
    a runtime calculation can be formed. This is a metadata table that can
    be used for optimizations insights, or to gain an understanding for certain
    partic_ids that may take more or less time to project.

    Args:
        results: A dictionary of DataFrames and metadata used to store all
          projection results.
        start_time: A datetime representing the start of this particular
          partic_id's projection.
    
    Returns:
        A DataFrame containing the runtime metadata for one partic_id.
    """
    # Concat pin runtime data to runtimes.
    end_time = datetime.now()
    runtimes = pd.DataFrame([
        results['pin'],
        results['plan'],
        results['subplan'],
        results['sid'],
        start_time,
        end_time,
        round((end_time - start_time).total_seconds() / 60, 2),
        results['total_proj_days'],
    ]).transpose()
    runtimes = update_runtimes_dtypes(runtimes)

    return runtimes

def map_trad_to_cref_trad(date_tm, date_df):
    """Creates a mapping between trad and cref_trad tickers in the date_tm.

    Args:
        date_tm: A ticker map for considered assets at the date level.
        date_df: The BTR data filtered on the specific partic_id and current
          date of the projection.
        
    Returns:
        The date_tm DataFrame with the 'mf_ticker_new' column added as the new
        mapping.
    """
    trad_tickers = date_df[date_df['ic220_priceid'].isin(TRAD_TICKERS)]['ic220_priceid'].unique()

    if len(trad_tickers) > 0:
        trad_ticker = trad_tickers[0]
    else:
        trad_ticker = None
    date_tm['mf_ticker_new'] = np.where(
        date_tm['is_aait_trad'] == 1, trad_ticker, date_tm['mf_ticker']
    )
    
    return date_tm

def calc_t381_t815(rebal_data, date_df, date_tm, model_weight, date_tm_trad_legacy_bal, baseline):
    """Calculates t381 and t815 allocations.

    This function is used to calculate the new t381 and t815 allocations during
    a rebalancing event. This function is only called, though, to determine if
    a drift rebalancing event has been triggered or not. If a drift rebalancing
    is triggered, then these t381 andd t815 values can be referenced instead of
    recalculating them in the rebalancing portion of the calc_pojections_day
    function.

    Args:
        rebal_data: DataFrame containing the results from the rebalancing gate
          calculation.
        date_df: The BTR data filtered on the specific partic_id and current
          date of the projection.
        date_tm: A ticker map for considered assets at the date level.
        model_weight: Identifies the target model weights.
        date_tm_trad_legacy_bal: The sum of all trad raw legacy balances.
        baseline: Boolean value specifying if the run is baseline or corrected.
    
    Returns:
        A tupe of results, the first being the rebal_data passed into the
        function. The second is the gate_hist DataFrame, which contains the
        gX_hist data, where x is a number 1-8 depending on which rebalancing
        gate logic was triggered. Third is rebal_gate, which is a string that
        represents the rebalancing gate logic that was triggered as well.
    """
    if not baseline:
        illiquid_ind = date_df['cr_mms_illiquid_ind'].unique()[0]
    else:
        illiquid_ind = date_df['bl_mms_illiquid_ind'].unique()[0]
    rebal_gate, overalloc_data, trad_active_pct, trad_target_pct = identify_overalloc(
        rebal_data,
        date_df['implicit_ind'].unique()[0],
        illiquid_ind,
        date_tm_trad_legacy_bal,
        date_tm,
        model_weight,
    )

    if rebal_gate == GATE_NAMES["gate_1"]:
        gate_hist, t381, t815 = calc_gate_1(overalloc_data, trad_active_pct, trad_target_pct)
        rebal_data["t381_calc"] = t381
        rebal_data["t815_calc"] = t815

    elif rebal_gate == GATE_NAMES["gate_2"]:
        gate_hist, t381, t815 = calc_gate_2(overalloc_data, trad_active_pct, trad_target_pct)
        rebal_data["t381_calc"] = t381
        rebal_data["t815_calc"] = t815

    elif rebal_gate == GATE_NAMES["gate_3"]:
        gate_hist, t381, t815 = calc_gate_3(overalloc_data, trad_active_pct, trad_target_pct)
        rebal_data["t381_calc"] = t381
        rebal_data["t815_calc"] = t815

    elif rebal_gate == GATE_NAMES["gate_4"]:
        gate_hist, t381, t815 = calc_gate_4(overalloc_data, trad_active_pct, trad_target_pct)
        rebal_data["t381_calc"] = t381
        rebal_data["t815_calc"] = t815

    elif rebal_gate == GATE_NAMES["gate_5"]:
        gate_hist, t381, t815 = calc_gate_5(overalloc_data)
        rebal_data["t381_calc"] = t381
        rebal_data["t815_calc"] = t815
                                
    elif rebal_gate == GATE_NAMES["gate_6"]:
        gate_hist, t381, t815 = calc_gate_6(overalloc_data, trad_active_pct, trad_target_pct)
        rebal_data["t381_calc"] = t381
        rebal_data["t815_calc"] = t815
    
    elif rebal_gate == GATE_NAMES["gate_7"]:
        gate_hist, t381, t815 = calc_gate_7(overalloc_data)
        rebal_data["t381_calc"] = t381
        rebal_data["t815_calc"] = t815
    
    elif rebal_gate == GATE_NAMES["gate_8"]:
        gate_hist, t381, t815 = calc_gate_8(overalloc_data, trad_active_pct, trad_target_pct)
        rebal_data["t381_calc"] = t381
        rebal_data["t815_calc"] = t815
    
    gate_hist['rebalance_gate'] = rebal_gate

    return rebal_data, gate_hist, rebal_gate

def determine_drift(rebal_data, date_df, model_weight, results):
    """Determines if a drift rebalancing event is triggered on the current date.

    This function considers the currently calculated t381s and t815s to those
    of the previous rebalancing event to determine if the true allocations have
    drifted beyond the drift threshold.

    Args:
        rebal_data: DataFrame containing the results from the rebalancing gate
          calculation.
        date_df: The BTR data filtered on the specific partic_id and current
          date of the projection.
        model_weight: Identifies the target model weights.
        results: A dictionary of DataFrames and metadata used to store all
          projection results.
    """
    g_hist_complete = results['gate_hist_complete']
    if g_hist_complete.empty:
        last_rebal_dt = None
    else:
        last_rebal_dt = g_hist_complete['calendar_dates'].sort_values(ascending=False).unique()[0]

    drift_chk = rebal_data[[
        "pin",
        "plan",
        "sub_plan",
        "ic220_priceid",
        "calendar_dates",
        "t381_calc",
        "t815_calc",
        "prev_day_proj",
    ]]
    drift_chk = df_attach_metadata(drift_chk, run_id=RUN_ID, version=VERSION)
    
    if last_rebal_dt != None:
        last_rebal_data = g_hist_complete.loc[
            g_hist_complete["calendar_dates"] == last_rebal_dt,
            ['ic220_priceid', 't381_calc', 't815_calc']
        ]
    else:
        last_rebal_data = pd.DataFrame()
        last_rebal_data['ic220_priceid'] = drift_chk['ic220_priceid']
        last_rebal_data['prev_t381_calc'] = 0
        last_rebal_data['prev_t815_calc'] = 0
    
    last_rebal_data.columns = ['ic220_priceid', 'prev_t381_calc', 'prev_t815_calc']

    drift_thresh = date_df['drift_threshold'].unique()[0] / 100
    drift_chk = pd.merge(
        drift_chk,
        last_rebal_data,
        how='left',
        on=['ic220_priceid']
    )
    drift_chk = pd.merge(
        drift_chk,
        model_weight[["pin", "plan", "sub_plan", "calendar_dates", "fund_ticker", "fund_weight"]],
        how='left', 
        left_on=["pin","plan","sub_plan","calendar_dates","ic220_priceid"], 
        right_on=["pin","plan","sub_plan","calendar_dates","fund_ticker"],
    )
    drift_chk['fund_weight'] = drift_chk['fund_weight'] / 100
    drift_chk["is_aait_trad"] = np.where(
        drift_chk["ic220_priceid"].isin(TRAD_TICKERS), 1, 0
    )
    drift_chk['drift_threshold'] = drift_thresh
    drift_chk['prev_day_proj_weight'] = drift_chk['prev_day_proj'] / drift_chk['prev_day_proj'].sum()
    
    drift_chk['t381_diff'] = abs(round(drift_chk['t381_calc'] - drift_chk['prev_day_proj_weight'], 10))
    drift_chk['model_weight_diff'] = round(drift_chk['t381_calc'] - drift_chk['fund_weight'], 10)

    drift_chk['t381_diff_threshold_chk'] = np.where(
        drift_chk['t381_diff'] > drift_chk['drift_threshold'], 1, 0
    )
    drift_chk['model_weight_diff_threshold_chk'] = np.where(
        drift_chk['model_weight_diff'] > drift_chk['drift_threshold'], 1, 0
    )
    drift_chk['t815_diff_threshold_chk'] = np.where(
        drift_chk['is_aait_trad'] == 1,
        np.where(
            drift_chk['t815_calc'] == 0,
            np.where(
                drift_chk['prev_t815_calc'] > 0,
                1,
                0
            ),
            0,
        ),
        0
    )

    if drift_chk['t381_diff_threshold_chk'].sum() > 0:
        drift_rebal_ind = 1
    
    elif (drift_chk['model_weight_diff_threshold_chk'] * drift_chk['is_aait_trad']).sum() > 0:
        if (drift_chk['t815_diff_threshold_chk'] * drift_chk['is_aait_trad']).sum() > 0:
            drift_rebal_ind = 1
        else:
            drift_rebal_ind = 0

    else:
        drift_rebal_ind = 0
    
    drift_chk['drift_rebal_ind'] = drift_rebal_ind
    
    return drift_rebal_ind, drift_chk

def update_gate_hist(rebal_data, date_df, gate_name):
    """Updates the complete rebal gate history table.

    This function creates a DataFrame with all of the updated gate_hist_complete
    data for a particular rebalancing event. The resulting dataframe is then
    appended to the results DataFrame and stored for enventual output.

    Args:
        rebal_data: DataFrame used as the input data to determine the
          rebalancing gate type and to calculate the t381s and t815s.
        date_df: The BTR data filtered on the specific partic_id and current
          date of the projection.
        gate_name: A string specifying which rebalancing gate logic was
          triggered during this rebalancing event.
    
    Returns:
        DataFrame constaining the updated information. All columns will match
        the gate_hist_complete column configuration in table_config.yaml.
    """
    columns = [
        "pin",
        "plan",
        "sub_plan",
        "calendar_dates",
        "ic220_priceid",
        "t381_calc",
        "t815_calc"
    ]
    df = rebal_data[columns].drop_duplicates()
    df = pd.merge(
        df,
        date_df[["ic220_priceid","t815_pct"]].drop_duplicates(),
        how='left',
        on='ic220_priceid',
    )
    df["rebalance_gate"] = gate_name
    
    return df

def query_cr_rebal_dts(id, sid):
    """Queries stg_cr_rebal_readin_dt_final for one partic_id and scenario_id.

    This function queries and returns all data in the
    stg_cr_rebal_readin_dt_final table and returns the data as a DataFrame.

    Args:
        id: partic_id containing a concatination of pin, plan, and sub_plan.
        sid: The scenario_id of the current projection.

    Returns:
        A DataFrame of all the data corresponding to a pin-plan-subplan-scenario
        combination.
    """
    query = f'''
    select distinct
        concat(pin, [plan], sub_plan) as partic_id,
        pin, [plan], sub_plan, new_rebal_dt as cr_rebal_readin_dt
    from {SCHEMA}.stg_cr_rebal_readin_dt_final
    where concat(pin, [plan], sub_plan) = '{id}'
        and scenario_id = {sid};
    '''
    engine = sa.create_engine(CONN_URL)
    df = pd.read_sql(query, con=engine)
    
    return df

def query_cr_illiquid_ind(id, sid):
    """Queries stg_rebal_cr_illiquid_ind for one partic_id and scenario_id.

    This function queries and returns all data in the stg_rebal_cr_illiquid_ind
    table and returns the data as a DataFrame.

    Args:
        id: partic_id containing a concatination of pin, plan, and sub_plan.
        sid: The scenario_id of the current projection.

    Returns:
        A DataFrame of all the data corresponding to a pin-plan-subplan-scenario
        combination.
    """
    query = f'''
    select distinct
        concat(pin, [plan], sub_plan) as partic_id,
        pin, [plan], sub_plan, trade_dt
    from {SCHEMA}.stg_rebal_cr_illiquid_ind
    where concat(pin, [plan], sub_plan) = '{id}'
    and scenario_id = {sid};
    '''
    engine = sa.create_engine(CONN_URL)
    df = pd.read_sql(query, con=engine)
    
    return df

def calc_projections_day(baseline, read_in, drift, dynamic, results, btr,
    partic_tm, model_weight, today, yest):
    """Calculates and stores the projection for a single day.

    This function is called for every day of the projection period, passing in
    the previous day's projection values and current day's btr data. All results
    are appended to the results dictionary which is a dictionary of DataFrames
    and metadata used to store all projection results.

    Args:
        baseline: Boolean value specifying if the run is baseline or corrected.
        read_in: Boolean value specifying if we are reading in rebalancing data
          or calculating it live.
        drift: Boolean value specifying if the run will include drift
          rebalancing. (Implies read_in = False)
        dynamic: Boolean value specifying if the run will include dynamic
          rebalancing. (Implies read_in = False and drift = True)
        results: A dictionary of DataFrames and metadata used to store all
          projection results.
        btr: The full BTR data filtered on one specific partic_id.
        partic_tm: A ticker map for considered assets at the participant level.
        model_weight: Identifies the target model weights.
        today: The current date of the projection in YYYY-MM-DD fomat.
        yest: The current date of the projection -1 in YYYY-MM-DD fomat.
    
    Returns:
        The updated results dictionary with the current day's results appended
        to the projection, debugging, and metadata tables.
    """
    id, pin, plan, subplan, sid, t444_4_date = (
        results['partic_id'],
        results['pin'],
        results['plan'],
        results['subplan'],
        results['sid'],
        results['t444_4_date'],
    )
    # Create a subset of the data filtered on each date.
    date_df = btr[btr["calendar_dates"] == today]
    date_df = date_df.sort_values(['source', 'ic220_priceid', 'contract_vintage'])
    if not read_in:
        date_tm = partic_tm[partic_tm["date"] == today - timedelta(days=1)]
        date_tm_trad_legacy_bal = (date_tm['is_aait_trad'] * date_tm['raw_legacy_balance'].fillna(0)).sum()
        if not date_tm.empty:
            date_tm = map_trad_to_cref_trad(date_tm, date_df)
            date_tm = (
                date_tm[["mf_ticker_new", "raw_legacy_balance"]]
                .groupby("mf_ticker_new")
                .agg("sum")
                .reset_index()
            )
        else:
            date_tm['mf_ticker_new'] = None
            date_tm = date_tm[["mf_ticker_new", "raw_legacy_balance"]]

    if DEBUG:
        print(today)
    
    # CASE 1: First day of projections
    if yest == None:
        # Set previous day's projection.
        date_df["prev_day_proj"] = 0

        # Calculate projection.
        a, b, c, d, e, f, g = projection_start(date_df)

        drift_chk = pd.DataFrame()
    else:
        # Set previous day's projection.
        date_df["prev_day_proj"] = results['proj'].loc[
                results['proj']["calendar_dates"] == yest, ["projection"]
            ].values
    
    # Perform drift check
    if (
        (
            (baseline and drift)
            or (not baseline and today < date_df['proj_end_dt'].unique()[0])
        )
        and date_df['drift_ind'].unique()[0] == 1
        and date_df['working_day_ind'].unique()[0] == 1
        and date_df['prev_day_proj'].sum() > 10000
        # and not date_df['blc_rebal_readin_dt_ind'].unique()[0] == 1
        and yest != None 
        and today != date_df['proj_start_dt'].unique()[0]
    ):
        rebal_data = create_rebal_data(date_df)
        rebal_data, g_hist, gate_name = calc_t381_t815(rebal_data, date_df, date_tm, model_weight, date_tm_trad_legacy_bal, baseline)
        drift_rebal_ind, drift_chk = determine_drift(rebal_data, date_df, model_weight, results)
        drift_chk['rebalance_gate'] = gate_name
        drift_chk['scenario_id'] = sid
        if drift_rebal_ind == 1:
            if DEBUG:
                print(f'DRIFT TRIGGERED: {gate_name}')
            
            if gate_name == GATE_NAMES["gate_1"]:
                results['g1_hist'] = pd.concat([results['g1_hist'], g_hist])
            elif gate_name == GATE_NAMES["gate_2"]:
                results['g2_hist'] = pd.concat([results['g2_hist'], g_hist])
            elif gate_name == GATE_NAMES["gate_3"]:
                results['g3_hist'] = pd.concat([results['g3_hist'], g_hist])
            elif gate_name == GATE_NAMES["gate_4"]:
                results['g4_hist'] = pd.concat([results['g4_hist'], g_hist])
            elif gate_name == GATE_NAMES["gate_5"]:
                results['g5_hist'] = pd.concat([results['g5_hist'], g_hist])
            elif gate_name == GATE_NAMES["gate_6"]:
                results['g6_hist'] = pd.concat([results['g6_hist'], g_hist])
            elif gate_name == GATE_NAMES["gate_7"]:
                results['g7_hist'] = pd.concat([results['g7_hist'], g_hist])
            elif gate_name == GATE_NAMES["gate_8"]:
                results['g8_hist'] = pd.concat([results['g8_hist'], g_hist])
            rebal_gate_hist = update_gate_hist(rebal_data, date_df, gate_name)
            results['gate_hist_complete'] = pd.concat([results['gate_hist_complete'], rebal_gate_hist])
            
            a, b, c, d, e, f, g, t444_4_date = projection_rebal(
                date_df,
                results['proj'],
                rebal_data,
                read_in,
                baseline,
                drift_rebal_ind,
                id,
                today,
                t444_4_date,
            )
        else:
            rebal_gate_hist = pd.DataFrame()
            a, b, c, d, e, f, g, t444_4_date = projection_else(
                date_df,
                results['proj'],
                results['proj_non_trad'],
                today,
                yest,
                read_in,
                baseline,
                t444_4_date,
            )

    else:
        drift_chk = pd.DataFrame()
        drift_rebal_ind = 0
    
    # CASE 2: Rebalancing event.
    if (
        (
            (
                (baseline and not (drift or dynamic)) # if blb
                and date_df["rebal_dt"].notnull().max()
                # and date_df["blc_rebal_readin_dt_ind"].unique()[0]
            )
            or (
                (baseline and drift and not dynamic) # if blc
                and date_df["blc_rebal_readin_dt_ind"].unique()[0]
                and drift_rebal_ind == 0
            )
            or (
                (baseline and dynamic) # if bld
                and date_df["bld_rebal_readin_dt_ind"].unique()[0]
                and drift_rebal_ind == 0
            )
            or (
                not baseline # if cr
                # and date_df["cr_rebal_readin_dt"].notnull().max()
                and date_df["blc_rebal_readin_dt_ind"].unique()[0]
                and drift_rebal_ind == 0
            )
        )
        and yest != None
    ):
        # Create rebal_data DataFrame
        rebal_data = create_rebal_data(date_df)

        # Execute live rebalancing if read_in = False
        if not read_in:
            if not baseline:
                illiquid_ind = date_df['cr_mms_illiquid_ind'].unique()[0]
            else:
                illiquid_ind = date_df['bl_mms_illiquid_ind'].unique()[0]
            # Identify overallocations and the correct gates.
            rebal_gate, overalloc_data, trad_active_pct, trad_target_pct = identify_overalloc(
                rebal_data,
                date_df['implicit_ind'].unique()[0],
                illiquid_ind,
                date_tm_trad_legacy_bal,
                date_tm,
                model_weight,
            )
            if rebal_gate == GATE_NAMES["gate_1"]:
                gate_hist, t381, t815 = calc_gate_1(overalloc_data, trad_active_pct, trad_target_pct)
                rebal_data["t381_calc"] = t381
                rebal_data["t815_calc"] = t815

                gate_hist["rebalance_gate"] = rebal_gate
                results['g1_hist'] = pd.concat([results['g1_hist'], gate_hist])

            elif rebal_gate == GATE_NAMES["gate_2"]:
                gate_hist, t381, t815 = calc_gate_2(overalloc_data, trad_active_pct, trad_target_pct)
                rebal_data["t381_calc"] = t381
                rebal_data["t815_calc"] = t815

                gate_hist["rebalance_gate"] = rebal_gate
                results['g2_hist'] = pd.concat([results['g2_hist'], gate_hist])

            elif rebal_gate == GATE_NAMES["gate_3"]:
                gate_hist, t381, t815 = calc_gate_3(overalloc_data, trad_active_pct, trad_target_pct)
                rebal_data["t381_calc"] = t381
                rebal_data["t815_calc"] = t815

                gate_hist["rebalance_gate"] = rebal_gate
                results['g3_hist'] = pd.concat([results['g3_hist'], gate_hist])

            elif rebal_gate == GATE_NAMES["gate_4"]:
                gate_hist, t381, t815 = calc_gate_4(overalloc_data, trad_active_pct, trad_target_pct)
                rebal_data["t381_calc"] = t381
                rebal_data["t815_calc"] = t815

                gate_hist["rebalance_gate"] = rebal_gate
                results['g4_hist'] = pd.concat([results['g4_hist'], gate_hist])

            elif rebal_gate == GATE_NAMES["gate_5"]:
                gate_hist, t381, t815 = calc_gate_5(overalloc_data)
                rebal_data["t381_calc"] = t381
                rebal_data["t815_calc"] = t815

                gate_hist["rebalance_gate"] = rebal_gate
                results['g5_hist'] = pd.concat([results['g5_hist'], gate_hist])
                                        
            elif rebal_gate == GATE_NAMES["gate_6"]:
                gate_hist, t381, t815 = calc_gate_6(overalloc_data, trad_active_pct, trad_target_pct)
                rebal_data["t381_calc"] = t381
                rebal_data["t815_calc"] = t815

                gate_hist["rebalance_gate"] = rebal_gate
                results['g6_hist'] = pd.concat([results['g6_hist'], gate_hist])
            
            elif rebal_gate == GATE_NAMES["gate_7"]:
                gate_hist, t381, t815 = calc_gate_7(overalloc_data)
                rebal_data["t381_calc"] = t381
                rebal_data["t815_calc"] = t815

                gate_hist["rebalance_gate"] = rebal_gate
                results['g7_hist'] = pd.concat([results['g7_hist'], gate_hist])
            
            elif rebal_gate == GATE_NAMES["gate_8"]:
                gate_hist, t381, t815 = calc_gate_8(overalloc_data, trad_active_pct, trad_target_pct)
                rebal_data["t381_calc"] = t381
                rebal_data["t815_calc"] = t815

                gate_hist["rebalance_gate"] = rebal_gate
                results['g8_hist'] = pd.concat([results['g8_hist'], gate_hist])

            if DEBUG:
                print(f'REBAL TRIGGERED: {rebal_gate}')
            
            # Update the rebal_gate_history for the participant.
            rebal_gate_hist = update_rebal_gate_hist(date_df, rebal_data, rebal_gate)
            results['gate_hist_complete'] = pd.concat([results['gate_hist_complete'], rebal_gate_hist])

        # If read_in = True we set t815 and t381 calc values to 0.
        else:
            rebal_data["t381_calc"] = 0
            rebal_data["t815_calc"] = 0

        # Calculate projection.
        a, b, c, d, e, f, g, t444_4_date = projection_rebal(
            date_df,
            results['proj'],
            rebal_data,
            read_in,
            baseline,
            drift_rebal_ind,
            id,
            today,
            t444_4_date,
        )

    # CASE 3: Else
    elif yest != None and drift_rebal_ind == 0:
        # Check if today is the proj start date and it's a blb, blb, or cr run.
        if today == date_df['proj_start_dt'].unique()[0]:
            rebal_data = create_rebal_data(date_df)

            if not read_in:
                rebal_data, g_hist, gate_name = calc_t381_t815(
                    rebal_data,
                    date_df,
                    date_tm,
                    model_weight,
                    date_tm_trad_legacy_bal,
                    baseline
                )

            else:
                rebal_data["t381_calc"] = 0
                rebal_data["t815_calc"] = 0
            
            # Overwrite previous day's t815_calc.
            temp = pd.merge(results['proj_non_trad']['ic220_priceid'], rebal_data[['ic220_priceid', 't815_calc']], on='ic220_priceid')['t815_calc']
            results['proj_non_trad']['t815_calc'] = temp

        # Calculate projection.
        a, b, c, d, e, f, g, t444_4_date = projection_else(
            date_df,
            results['proj'],
            results['proj_non_trad'],
            today,
            yest,
            read_in,
            baseline,
            t444_4_date,
        )

    # Update projection DataFrames
    daily_proj = a
    daily_proj_trad = b
    daily_proj_non_trad = c
    debit_amts = add_pps_date(pin, plan, subplan, today, sid, d)
    accrued_div_calc = e
    rebal_cash_hist = add_pps_date(pin, plan, subplan, today, sid, f)
    two_step_rebal_chk = add_pps_date(pin, plan, subplan, today, sid, g)

    results['proj'] = pd.concat(
        [results['proj'], daily_proj]
    )
    results['proj_trad'] = pd.concat(
        [results['proj_trad'], daily_proj_trad]
    )
    results['proj_non_trad'] = pd.concat(
        [results['proj_non_trad'], daily_proj_non_trad]
    )
    results['debit_amts'] = pd.concat(
        [results['debit_amts'], debit_amts]
    )
    results['accrued_div_calc'] = pd.concat(
        [results['accrued_div_calc'], accrued_div_calc]
    )
    results['rebal_cash_hist'] = pd.concat(
        [results['rebal_cash_hist'], rebal_cash_hist]
    )
    results['two_step_rebal_chk'] = pd.concat(
        [results['two_step_rebal_chk'], two_step_rebal_chk]
    )
    if drift:
        results['drift_chk'] = pd.concat([results['drift_chk'], drift_chk])

    results['t444_4_date'] = t444_4_date

    return results

def calc_projections_worker(baseline, read_in, drift, dynamic, id, sid, i, btr):
    """Calculates and stores the full projection for one partic_id.

    This function calculates the full projection period of one partic_id. The
    results are appended to the results dictionary at the end of each day's
    projection as a dictionary of DataFrames and metadata.
    
    Although the code's logic iterates through scenario ids, it is worth noting
    that for optimization purposes, each function call of
    calc_projections_worker instead also passes a scenario_id for a particular
    participant. This allows participants with a high number of scnearios to be
    processed in parallel instead of in series.

    Args:
        baseline: Boolean value specifying if the run is baseline or corrected.
        read_in: Boolean value specifying if we are reading in rebalancing data
          or calculating it live.
        drift: Boolean value specifying if the run will include drift
          rebalancing. (Implies read_in = False)
        dynamic: Boolean value specifying if the run will include dynamic
          rebalancing. (Implies read_in = False and drift = True)
        id: partic_id containing a concatination of pin, plan, and sub_plan.
        sid: The scenario_id of the current projection.
        i: The id number of the current partic_id-scenario being projected. This
          is an identifier for logs and terminal output.
        btr: The full BTR data filtered on one specific partic_id.

    Returns:
        scenario_results which is a dictionary of results. The keys are
        scenario_ids and the values are results dictionaries. Each
        scenario_results dictionary will only have one key-value pair since
        this function processes each partic_id-scenario_id combination
        separately.
    """
    scenario_ids = [sid]
    partic_tm = query_ca_ticker_map(id, baseline)

    if not baseline:
        cr_rebal_dts = query_cr_rebal_dts(id, sid)
        btr = pd.merge(
            btr,
            cr_rebal_dts,
            how='left',
            left_on=['partic_id', 'pin', 'plan', 'sub_plan', 'calendar_dates'],
            right_on=['partic_id', 'pin', 'plan', 'sub_plan', 'cr_rebal_readin_dt'],
        )
        cr_illiquid_ind = query_cr_illiquid_ind(id, sid)
        btr = pd.merge(
            btr,
            cr_illiquid_ind,
            how='left',
            left_on=['partic_id', 'pin', 'plan', 'sub_plan', 'calendar_dates'],
            right_on=['partic_id', 'pin', 'plan', 'sub_plan', 'trade_dt'],
        )
        btr['cr_mms_illiquid_ind'] = np.where(btr['trade_dt'].isnull(), 1, 0)
        btr['cr_omni_illiquid_ind'] = btr['cr_mms_illiquid_ind']
    else:
        btr['cr_rebal_readin_dt'] = np.nan
        btr['trade_dt'] = np.nan
        btr['cr_mms_illiquid_ind'] = np.nan
        btr['cr_omni_illiquid_ind'] = btr['cr_mms_illiquid_ind']

    scenario_results = dict()
    for sid in scenario_ids:
        start_time = datetime.now()
        btr['scenario_id'] = sid
        model_weight = query_model_weight(id, sid, baseline)

        unique_dates = btr["calendar_dates"].unique().tolist()
        unique_dates.sort()
        total_proj_days = len(unique_dates)

        results = {
            # Identifier data
            "partic_id": id,
            "job_id": i,
            "pin": btr["pin"].iloc[0],
            "plan": btr["plan"].iloc[0],
            "subplan": btr["sub_plan"].iloc[0],
            "sid": sid,

            # Proj parameters
            "total_proj_days": total_proj_days,
            "proj_type": get_proj_type(baseline, read_in, drift, dynamic),
            "t444_4_date": 0,

            # Metadata tables
            "runtimes": pd.DataFrame(),
            "error_table": pd.DataFrame(),

            # Projection tables
            "proj": pd.DataFrame(),
            "proj_trad": pd.DataFrame(),
            "proj_non_trad": pd.DataFrame(),

            # Debugging tables
            "debit_amts": pd.DataFrame(),
            "accrued_div_calc": pd.DataFrame(),
            "rebal_cash_hist": pd.DataFrame(),
            "two_step_rebal_chk": pd.DataFrame(),
            "drift_chk": pd.DataFrame(),
            
            # Rebalancing gate history tables
            "g1_hist": pd.DataFrame(),
            "g2_hist": pd.DataFrame(),
            "g3_hist": pd.DataFrame(),
            "g4_hist": pd.DataFrame(),
            "g5_hist": pd.DataFrame(),
            "g6_hist": pd.DataFrame(),
            "g7_hist": pd.DataFrame(),
            "g8_hist": pd.DataFrame(),
            "gate_hist_complete": pd.DataFrame(),
        }
        
        print(
            f"[PROJECTING]: {i+1} | " +
            f"S{results['sid']}-" +
            f"{results['pin']}-" +
            f"{results['plan']}-" +
            f"{results['subplan']} | " +
            f"{datetime.now()}"
        )
        # Calculates the full projection for a pin-plan-sublpan-sid
        for j in range(total_proj_days):
            # Create references for the current and previous date.
            try:
                today = unique_dates[j]
                if j == 0:
                    yest = None 
                else:
                    yest = unique_dates[j - 1]
                    
                results = calc_projections_day(
                    baseline,
                    read_in,
                    drift,
                    dynamic,
                    results,
                    btr,
                    partic_tm,
                    model_weight,
                    today,
                    yest,
                )
            
            except Exception as e:
                if DEBUG:
                    raise e
                results = clear_errored_results(results, today, e)
                break

        results = add_sid_to_gate_hist(results)
        results['runtimes'] = create_runtimes_table(results, start_time)

        # Exporting the results for the current partic.
        scenario_results[sid] = results

    return scenario_results

def export_single_pin(worker_results):
    """Exports to local storage each partic_id-scenario_id combination.

    This function exports each partic_id-scenario_id results as a separate .txt
    file. The result will be a folder structure organized by:
    OUTPUT_PATH/RUN_NAME/TABLE_NAME/{partic_id_results}.txt

    OUTPUT_PATH is specifed in the config file.
    RUN_NAME is generated dynamically based on other config parameters.
        E.g., lec_phase2_bla_v80, lec_phase3_cr_v83_12pins
    TABLE_NAME is the same as one of the projection, debug, or metadata tables.

    Args:
        worker_results: A dictionary of all results for partic_id-scenario_ids
          in the batch. I.e., A batch size of 100 implies worker_results has 100
          elements
    
    Returns:
        None
    """
    for sid in worker_results:
        scenario_dict = worker_results[sid]
        id = scenario_dict['partic_id']
        proj_type = scenario_dict['proj_type']
        print(f"[EXPORTING]: S{sid}-{id} | {datetime.now()}")
        tables = {
            # Proj tables
            'proj': scenario_dict['proj'],
            'proj_trad': scenario_dict['proj_trad'],
            'proj_non_trad': scenario_dict['proj_non_trad'],

            # Debugging tables
            'debit_amts': scenario_dict['debit_amts'],
            'accrued_div_calc': scenario_dict['accrued_div_calc'],
            'rebal_cash_hist': scenario_dict['rebal_cash_hist'],
            'two_step_rebal_chk': scenario_dict['two_step_rebal_chk'],
            'drift_chk': scenario_dict['drift_chk'],

            # Metadeta tables
            'runtimes': scenario_dict['runtimes'],
            'error_table': scenario_dict['error_table'],
            
            # Reablancing gate history tables
            'g1_hist': scenario_dict['g1_hist'],
            'g2_hist': scenario_dict['g2_hist'],
            'g3_hist': scenario_dict['g3_hist'],
            'g4_hist': scenario_dict['g4_hist'],
            'g5_hist': scenario_dict['g5_hist'],
            'g6_hist': scenario_dict['g6_hist'],
            'g7_hist': scenario_dict['g7_hist'],
            'g8_hist': scenario_dict['g8_hist'],
            'gate_hist_complete': scenario_dict['gate_hist_complete'],
        }
        table_end = get_filter_pins_suffix()
        if sid == 0:
            pin_end = f"_{id}"
        else:
            pin_end = f'_{id}_{sid}'
        prefix = f"result_{proj_type}"
        suffix = f'{TABLE_SUFFIX}{pin_end}'
        
        if TO_CSV:
            result_path = f'{RESULT_PATH}{SCHEMA}_{proj_type}_{TABLE_SUFFIX}{table_end}'
            os.makedirs(result_path, exist_ok=True)
            for table_name in tables:
                table_path = f'{result_path}/{table_name}'
                os.makedirs(table_path, exist_ok=True)
                table = tables[table_name]
                if not table.empty:
                    table = df_attach_metadata(table, run_id=RUN_ID, version=VERSION)
                    table.to_csv(
                        f'{table_path}/{SCHEMA}_{prefix}_{table_name}_{suffix}.txt',
                        index=False,
                        sep='|',
                    )

def run_roll_fwd_mp(baseline, read_in, drift, dynamic, n_batches, use_multiprocess, n_cores):
    """Executes Roll Forward for Model Points.

    This takes in parameters that determine if it is one of five run types:
        BLA (baseline read-in)
        BLB (baseline live)
        BLC (baseline drift)
        BLD (baseline dynamic)
        CR (corrected run)

    Args:
        baseline: Boolean value specifying if the run is baseline or corrected.
        read_in: Boolean value specifying if we are reading in rebalancing data
          or calculating it live.
        drift: Boolean value specifying if the run will include drift
          rebalancing. (Implies read_in = False)
        dynamic: Boolean value specifying if the run will include dynamic
          rebalancing. (Implies read_in = False and drift = True)
        n_batches: Specifies the number of batches all paricipants will be
          subdivided into.
        use_multiprocess: Indicator for if projections will be run in parallel
          or exclusively in series.
        n_cores: Specified the number of cores Python can allocate to a
          multiprocessing run.

    Returns:
        None
    """    
    full_run_start_time = datetime.now()

    print()
    print(f'[BEGINNING RUN]: Version = {VERSION} | Baseline = {baseline} | Read-in = {read_in} | Drift = {drift} | Dynamic = {dynamic} | {datetime.now()}')
    print()

    # Retrieve unique list of model points for processing and a list of batches.
    unique_partic = query_filter_table(read_in, drift)
    if not baseline:
        scenario_ids = query_scenario_ids(unique_partic['partic_id'].tolist())
        unique_partic = pd.merge(unique_partic, scenario_ids, how='left', on=['partic_id'])
        unique_partic = [f"{row['partic_id']}_{row['scenario_id']}" for i, row in unique_partic.iterrows()]
    else:
        unique_partic = unique_partic['partic_id'].tolist()

    batches = create_batches(n_batches, unique_partic)
    complete_jobs = 0
    read_times, proj_times, write_times, end_times = [], [], [], []
    for i in range(n_batches):
        # Instance the start time for data read.
        read_times.append(datetime.now())

        # Create a container for the results of each worker process
        results_arraydict = []

        # Get all pins in this batch.
        batch_partic = batches[i]
        n_jobs = len(batch_partic)
        print(f'[BEGINNING BATCH]: {i+1} / {n_batches} | {n_jobs} PINs | {datetime.now()}')

        # Create a cartesian-product parameter grid of args used for each job.
        params = []
        for job_id in range(n_jobs):
            if baseline:
                partic_id = batch_partic[job_id]
                sid = 0
            else:
                partic_id = batch_partic[job_id].split('_')[0]
                sid = batch_partic[job_id].split('_')[1]
            
            params.append((
                baseline,
                read_in,
                drift,
                dynamic,
                partic_id,
                sid,
                complete_jobs + job_id,
                query_btr(partic_id)
            ))

        # Instance the start time for projections.
        proj_times.append(datetime.now())

        # Create an mp pool if multiprocess is enabled and there is > 1 job.
        if use_multiprocess and n_jobs > 1:
            n_processes = min(mp.cpu_count(), n_cores, n_jobs)
            pool = mp.Pool(n_processes)

            for worker_results in pool.starmap_async(calc_projections_worker, params).get():
                # results_arraydict.append(worker_results)
                export_single_pin(worker_results)

        # Otherwise execute PINs in series.
        else:
            for job_id in range(n_jobs):
                worker_results = calc_projections_worker(
                    baseline,
                    read_in,
                    drift,
                    dynamic,
                    partic_id,
                    sid,
                    complete_jobs + job_id,
                    query_btr(partic_id),
                )
                # results_arraydict.append(worker_results)
                export_single_pin(worker_results)
                del worker_results

        # Instance the start time for data write to local storage.
        write_times.append(datetime.now())
        # export_run_results(results_arraydict)
        end_times.append(datetime.now())
        complete_jobs += n_jobs

        if TEXT_NOTIFICATIONS:
            run = get_proj_type(baseline, read_in, drift, dynamic)
            
            if i == n_batches - 1 and TO_SQL:
                message = fr'Batch {i+1} / {n_batches} is complete for {run}. '
                message += 'Continuing to data upload.'
            else:
                message = fr'Batch {i+1} / {n_batches} is complete for {run}.'
            
            send_text_notif(message)

    projection_times = pd.DataFrame({
        'total_pins': [complete_jobs for i in range(n_batches)],
        'batch': [i+1 for i in range(n_batches)],
        'start_timestamp': [full_run_start_time for i in range(n_batches)],
        'read_time': [
            round((proj_times[i] - read_times[i]).total_seconds() / 60, 2)
            for i in range(n_batches)
        ],
        'proj_time': [
            round((write_times[i] - proj_times[i]).total_seconds() / 60, 2)
            for i in range(n_batches)
        ],
        'write_time': [
            round((end_times[i] - write_times[i]).total_seconds() / 60, 2)
            for i in range(n_batches)
        ],
    })

    proj_type = get_proj_type(baseline, read_in, drift, dynamic)
    upload_start = datetime.now()
    if TO_SQL:
        print()
        print(f'[BEGINNING UPLOAD]: {datetime.now()}')
        upload_results(proj_type)
    
    upload_time = round((datetime.now() - upload_start).total_seconds() / 60, 2)
    if not DEBUG:
        update_ctrl_projection_times(projection_times, proj_type, upload_time)
    
    print()
    print(f"[COMPLETE]: {datetime.now()}\n")

def check_base_contract_bla_version_exists():
    """Checks if the results table for the BASE_CONTRACT_BLA_VERSION exists.

    Args:
        None

    Returns:
        Boolean values based on whether or not the results table for the 
        BASE_CONTRACT_BLA_VERSION exists.
    """
    query = f'select top 1 * from {SCHEMA}.result_bla_proj_{BASE_CONTRACT_BLA_VERSION}'
    engine = sa.create_engine(CONN_URL)

    try:
        df = pd.read_sql(query, con=engine)
        return True
    
    except:
        return False

def update_ca_queries():
    """
    """
    module = SourceFileLoader("05_data_prep", f"{dirname(__file__)}/05_data_prep.py").load_module()
    create_ca_queries = module.create_ca_queries
    
    stg_balance_daily_ca_query = f'''
    with base_contract_results as (
        select
            [pin],
            [plan],
            [sub_plan],
            '' as found_msg,
            '' as client_id,
            '' as client,
            '' as dob_ph050,
            '' as curr_model_ph641,
            [calendar_dates] as [date],
            '' as invest_id,
            '' as investment,
            '' as ticker,
            [ic220_priceid],
            sum(projection) as balance,
            '' as shares
        from {SCHEMA}.result_bla_proj_{BASE_CONTRACT_BLA_VERSION}
        where base_contract_ind = 1
        group by pin, [plan], sub_plan, calendar_dates, ic220_priceid
    )
    select *
    into {SCHEMA}.stg_balance_daily_ca
    from {SCHEMA}.clean_balance_daily_ca
    union
    select * from base_contract_results
    ;
    '''
    update_query = {'stg_balance_daily_ca': stg_balance_daily_ca_query}
    
    arc_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    loop_queries([update_query, create_ca_queries(prefix='stg')], arc_suffix)

def main():
    if TEXT_NOTIFICATIONS:
        runs = ''
        if RUN_BASELINE_READIN:
            runs += 'bla '
        if RUN_BASELINE_LIVE:
            runs += 'blb '
        if RUN_BASELINE_DRIFT:
            runs += 'blc '
        if RUN_BASELINE_DYNAMIC:
            runs += 'bld '
        if RUN_CORRECTED:
            runs += 'cr '
        runs = runs[:-1]

        end = ''
        if FILTER_PINS:
            end += f' {len(FILTER_PINS)}pins'
        if FILTER_SCENARIOS:
            end += f' {len(FILTER_SCENARIOS)}scenarios'
        
        message = fr"Beginning run(s) for {runs}, {TABLE_SUFFIX}{end}."
        send_text_notif(message)
    
    try:
        if RUN_BASELINE_READIN:
            run_roll_fwd_mp(
                baseline=True,
                read_in=True,
                drift=False,
                dynamic=False,
                n_batches=N_BATCHES,
                use_multiprocess=USE_MULTIPROCESS,
                n_cores=N_CORES,
            )
            if TEXT_NOTIFICATIONS:
                message = "SUCCESS! The baseline read-in run (bla) is complete."
                send_text_notif(message)

        if RUN_BASELINE_LIVE:
            assert_string = f'BLA {BASE_CONTRACT_BLA_VERSION} must exist.'
            assert check_base_contract_bla_version_exists() == True, assert_string
            update_ca_queries()

            run_roll_fwd_mp(
                baseline=True,
                read_in=False,
                drift=False,
                dynamic=False,
                n_batches=N_BATCHES,
                use_multiprocess=USE_MULTIPROCESS,
                n_cores=N_CORES,
            )
            if TEXT_NOTIFICATIONS:
                message = "SUCCESS! The baseline live run (blb) is complete."
                send_text_notif(message)

        if RUN_BASELINE_DRIFT:
            assert_string = f'BLA {BASE_CONTRACT_BLA_VERSION} must exist.'
            assert check_base_contract_bla_version_exists() == True, assert_string
            update_ca_queries()

            run_roll_fwd_mp(
                baseline=True,
                read_in=False,
                drift=True,
                dynamic=False,
                n_batches=N_BATCHES,
                use_multiprocess=USE_MULTIPROCESS,
                n_cores=N_CORES,
            )
            if TEXT_NOTIFICATIONS:
                message = "SUCCESS! The baseline drift run (blc) is complete."
                send_text_notif(message)
        
        if RUN_CORRECTED:
            run_roll_fwd_mp(
                baseline=False,
                read_in=False,
                drift=True,
                dynamic=True,
                n_batches=N_BATCHES,
                use_multiprocess=USE_MULTIPROCESS,
                n_cores=N_CORES,
            )
            if TEXT_NOTIFICATIONS:
                message = "SUCCESS! The corrected run (cr) is complete."
                send_text_notif(message)
    
    except Exception as e:
        if TEXT_NOTIFICATIONS:
            message = "WARNING! The current run was halted."
            send_text_notif(message)
        raise e

if __name__ == '__main__':
    main()
