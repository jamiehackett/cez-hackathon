import os
import numpy as np
import pandas as pd
import holidays
import matplotlib.pyplot as plt

#----------------------------------------------------------------------------------------------- Preprocessing -----------------------------------------------------------------------------
def create_seasonality_features(df):
    df = df.copy()
    ds_index = df.index

    hour_float = ds_index.hour + ds_index.minute / 60.0
    df['hour_sin'] = np.sin(2 * np.pi * hour_float / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour_float / 24)

    dow = ds_index.weekday
    df['dow_sin'] = np.sin(2 * np.pi * dow / 7)
    df['dow_cos'] = np.cos(2 * np.pi * dow / 7)

    doy = ds_index.dayofyear
    df['doy_sin'] = np.sin(2 * np.pi * doy / 365)
    df['doy_cos'] = np.cos(2 * np.pi * doy / 365)

    month = ds_index.month
    df['month_sin'] = np.sin(2 * np.pi * month / 12)
    df['month_cos'] = np.cos(2 * np.pi * month / 12)

    df['year'] = ds_index.year

    return df

def add_holiday_columns(df, cz_holidays):
    """
    Add Czech holiday dummy variables to the DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with a DatetimeIndex.
    cz_holidays : holidays.HolidayBase
        Holidays object for Czechia.

    Returns
    -------
    pandas.DataFrame
        DataFrame with holiday dummies and renamed columns.
    """
    unique_holidays = set(holiday for _, holiday in cz_holidays.items())
    df["is_holiday"] = df.index.map(lambda x: 1 if x in cz_holidays else 0)

    for holiday in unique_holidays:
        col = f'is_{holiday.replace(" ", "_").upper()}'
        df[col] = 0

    for date in df.index:
        if date in cz_holidays:
            holiday_name = cz_holidays[date]
            col = f'is_{holiday_name.replace(" ", "_").upper()}'
            if col in df.columns:
                df.at[date, col] = 1

    rename_map = {
        "is_2._SVÁTEK_VÁNOČNÍ": "is_2_svatek_vanocni",
        "is_DEN_SLOVANSKÝCH_VĚROZVĚSTŮ_CYRILA_A_METODĚJE": "is_den_slovanskych_verovestu_cyrila_a_metodeje",
        "is_DEN_ČESKÉ_STÁTNOSTI": "is_den_ceske_statnosti",
        "is_DEN_VZNIKU_SAMOSTATNÉHO_ČESKOSLOVENSKÉHO_STÁTU": "is_den_vzniku_samostatneho_ceskoslovenskeho_statu",
        "is_1._SVÁTEK_VÁNOČNÍ": "is_1_svatek_vanocni",
        "is_DEN_VÍTĚZSTVÍ": "is_den_vitezstvi",
        "is_ŠTĚDRÝ_DEN": "is_stedry_den",
        "is_DEN_UPÁLENÍ_MISTRA_JANA_HUSA": "is_den_upaleni_mistra_jana_husa",
        "is_VELKÝ_PÁTEK": "is_velky_patek",
        "is_DEN_BOJE_ZA_SVOBODU_A_DEMOKRACII": "is_den_boje_za_svobodu_a_demokracii",
        "is_SVÁTEK_PRÁCE": "is_svatek_prace",
        "is_VELIKONOČNÍ_PONDĚLÍ": "is_velikonocni_pondeli",
        "is_DEN_OBNOVY_SAMOSTATNÉHO_ČESKÉHO_STÁTU": "is_den_obnovy_samostatneho_ceskeho_statu",
    }

    df.rename(columns=rename_map, inplace=True)
    return df

def add_christmas_period(df, years=None):
    """
    Add a dummy variable for the Christmas period (27–31 December) for specified years.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with a DatetimeIndex.
    years : list, set, or None
        Years to mark for Christmas period. If None, defaults to 2020–2030.
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with IS_CHRISTMAS dummy variable.
    """
    if years is None:
        years = set(range(2005, 2031))  # default years 2005–2030 inclusive
    mask = (
        (df.index.month == 12) &
        (df.index.day >= 27) &
        (df.index.day <= 31) &
        (df.index.year.isin(years))
    )
    df["is_christmas"] = mask.astype(int)
    return df

def add_working_day_columns(df):
    """
    Add dummy variables for working days and non-working days.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with holiday columns.

    Returns
    -------
    pandas.DataFrame
        DataFrame with working day-related dummy variables.
    """
    df["is_working_day"] = ((df.index.weekday < 5) & (df["is_holiday"] == 0)).astype(int)
    df["is_non_working_day"] = 1 - df["is_working_day"]
    return df

def map_holiday_and_collapse(df, drop_original=True, result_col="holiday_type"):
    """
    Maps multiple binary holiday indicators into a single categorical holiday label
    and optionally drops the original binary columns.

    The input DataFrame must contain the following binary columns (0/1 values):
        - 'is_den_slovanskych_verovestu_cyrila_a_metodeje'
        - 'is_den_ceske_statnosti'
        - 'is_den_vzniku_samostatneho_ceskoslovenskeho_statu'
        - 'is_den_vitezstvi'
        - 'is_den_upaleni_mistra_jana_husa'
        - 'is_velky_patek'
        - 'is_den_boje_za_svobodu_a_demokracii'
        - 'is_svatek_prace'
        - 'is_velikonocni_pondeli'
        - 'is_2_svatek_vanocni'
        - 'is_1_svatek_vanocni'
        - 'is_stedry_den'
        - 'is_den_obnovy_samostatneho_ceskeho_statu'
        - 'is_christmas'

    Only one column should be active per row; if none are active, the resulting column
    will be 'no_holiday'.

    Args:
        df (pd.DataFrame): DataFrame containing the binary holiday columns.
        drop_original (bool, optional): Whether to drop the original binary columns. Defaults to True.
        result_col (str, optional): Name of the resulting categorical column. Defaults to "holiday".

    Returns:
        pd.DataFrame: DataFrame with the new categorical column and optional dropped binary columns.
    """
    def map_holiday(row):
        if row['is_den_slovanskych_verovestu_cyrila_a_metodeje']:
            return 'cyril_methodius'
        elif row['is_den_ceske_statnosti']:
            return 'czech_statehood'
        elif row['is_den_vzniku_samostatneho_ceskoslovenskeho_statu']:
            return 'independence_day'
        elif row['is_den_vitezstvi']:
            return 'victory_day'
        elif row['is_den_upaleni_mistra_jana_husa']:
            return 'jan_hus_day'
        elif row['is_velky_patek']:
            return 'good_friday'
        elif row['is_den_boje_za_svobodu_a_demokracii']:
            return 'freedom_day'
        elif row['is_svatek_prace']:
            return 'labour_day'
        elif row['is_velikonocni_pondeli']:
            return 'easter_monday'
        elif row['is_2_svatek_vanocni']:
            return 'xmas_26'
        elif row['is_1_svatek_vanocni']:
            return 'xmas_25'
        elif row['is_stedry_den']:
            return 'christmas_eve'
        elif row['is_den_obnovy_samostatneho_ceskeho_statu']:
            return 'renewal_day'
        elif row['is_christmas']:
            return 'christmas'
        else:
            return 'no_holiday'

    df[result_col] = df.apply(map_holiday, axis=1)

    if drop_original:
        holiday_cols = [
            'is_den_slovanskych_verovestu_cyrila_a_metodeje',
            'is_den_ceske_statnosti',
            'is_den_vzniku_samostatneho_ceskoslovenskeho_statu',
            'is_den_vitezstvi',
            'is_den_upaleni_mistra_jana_husa',
            'is_velky_patek',
            'is_den_boje_za_svobodu_a_demokracii',
            'is_svatek_prace',
            'is_velikonocni_pondeli',
            'is_2_svatek_vanocni',
            'is_1_svatek_vanocni',
            'is_stedry_den',
            'is_den_obnovy_samostatneho_ceskeho_statu',
            'is_christmas'
        ]
        df = df.drop(columns=holiday_cols)

    return df

def map_peak_and_collapse(df, drop_original=True, result_col="peak_type"):
    """
    Collapses multiple binary peak indicators into a single categorical time-of-day label
    and optionally drops the original binary columns.

    The input DataFrame must contain the following binary columns:
        - 'morning_peak'
        - 'noon_peak'
        - 'evening_peak'
        - 'night_drop'
    Each column should contain 0 or 1 to indicate whether the peak is active. 
    Only one column should be active per row; if none are active, the resulting column
    will be 'off_peak'.

    Args:
        df (pd.DataFrame): DataFrame containing the binary peak columns.
        drop_original (bool, optional): Whether to drop the original binary columns. Defaults to True.
        result_col (str, optional): Name of the resulting categorical column. Defaults to "peak_period".

    Returns:
        pd.DataFrame: DataFrame with the new categorical column and optional dropped binary columns.
    """
    def map_peak(row):
        if row['morning_peak']:
            return 'morning'
        elif row['noon_peak']:
            return 'noon'
        elif row['evening_peak']:
            return 'evening'
        elif row['night_drop']:
            return 'night'
        else:
            return 'off_peak'

    df[result_col] = df.apply(map_peak, axis=1)

    if drop_original:
        df = df.drop(columns=["morning_peak", "noon_peak", "evening_peak", "night_drop"])

    return df

def categorize_holiday_type(df, column='holiday_type'):
    """
    Converts the specified column of a DataFrame into a categorical type 
    with a predefined order for holiday types.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the column to categorize.
    column (str): The name of the column to convert (default is 'holiday_type').

    Returns:
    pd.DataFrame: The DataFrame with the column converted to a categorical type.
    """
    df[column] = pd.Categorical(df[column], categories=[
        'cyril_methodius', 'czech_statehood', 'independence_day', 'victory_day',
        'jan_hus_day', 'good_friday', 'freedom_day', 'labour_day', 'easter_monday',
        'xmas_26', 'xmas_25', 'christmas_eve', 'renewal_day', 'christmas', 'no_holiday'
    ])
    return df

def add_time_features(df):
    """
    Add time-based dummy variables (DST, HDO, and peak hours) with lowercase column names.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with a DatetimeIndex.

    Returns
    -------
    pandas.DataFrame
        DataFrame with time-related dummy variables.
    """
    df["hour"] = df.index.hour
    
    df["morning_peak"] = df["hour"].between(5, 8).astype(int)
    df["noon_peak"] = df["hour"].between(10, 12).astype(int)
    df["evening_peak"] = df["hour"].between(17, 19).astype(int)
    df["night_drop"] = ((df["hour"] >= 22) | (df["hour"] <= 4)).astype(int)
    return df.drop(["hour"], axis=1)

def categorize_peak_type(df, column='peak_type'):
    """
    Converts the specified column of a DataFrame into a categorical type 
    with a predefined order for peak times.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the column to categorize.
    column (str): The name of the column to convert (default is 'PEAK_TYPE').

    Returns:
    pd.DataFrame: The DataFrame with the column converted to a categorical type.
    """
    df[column] = pd.Categorical(df[column], categories=[
        'morning', 'noon', 'evening', 'night', 'off_peak'
    ])
    return df

def combine_binary_columns(
    df: pd.DataFrame,
    binary_cols=['is_working_day', 'is_non_working_day'],
    new_col='day_type',
    labels={'is_working_day': 'working', 'is_non_working_day': 'non_working'},
    drop_binary_cols: bool = False
) -> pd.DataFrame:
    """
    Combine two binary columns into a single categorical column with custom labels.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing binary columns.
    - binary_cols (list of str): List of two binary column names.
    - new_col (str): Name of the new column to create.
    - labels (dict): Mapping from column name to desired label.
    - drop_binary_cols (bool): Whether to drop the original binary columns (default False).

    Returns:
    - pd.DataFrame: A copy of the input DataFrame with the new combined column added.
    """
    df = df.copy()
    
    col1, col2 = binary_cols

    # Validate mutually exclusive binary flags
    invalid_rows = df[(df[col1] + df[col2]) != 1]
    if not invalid_rows.empty:
        raise ValueError(f"Rows with invalid flags:\n{invalid_rows.index.tolist()}")

    df[new_col] = df[col1].map({1: labels[col1]}).fillna(labels[col2])

    if drop_binary_cols:
        df = df.drop(columns=binary_cols)

    return df

def add_day_transition_column(df: pd.DataFrame, holidays: list) -> pd.DataFrame:
    """
    Add a 'day_transition' column to a DataFrame with a datetime index.

    The 'day_transition' column indicates the transition from the previous day to the current day
    for each row, formatted as "<current_day_type>_after_<previous_day_type>".
    All rows within the same calendar day will have the same value.

    The previous day for the first row is determined using calendar logic:
    - If the previous day is a weekend or in the holidays list, it is 'non_working'.
    - Otherwise, it is 'working'.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with:
        - datetime index
        - 'day_type' column containing 'working' or 'non_working'
    holidays : list of datetime.date
        Dates to be treated as non-working days.

    Returns
    -------
    pd.DataFrame
        Copy of the input DataFrame with an added 'day_transition' column.
    """

    # --- Checks ---
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        raise TypeError("DataFrame index must be of datetime type")

    if 'day_type' not in df.columns:
        raise ValueError("DataFrame must contain 'day_type' column")

    valid_day_types = {'working', 'non_working'}
    if not set(df['day_type'].unique()).issubset(valid_day_types):
        raise ValueError(f"'day_type' column contains invalid values. Allowed: {valid_day_types}")

    # --- Processing ---
    df = df.copy()
    df['date'] = df.index.floor('D')

    # Get unique days and their day_type
    daily = df[['date', 'day_type']].drop_duplicates().sort_values('date')

    # Infer previous day_type for all days except first
    daily['prev_day_type'] = daily['day_type'].shift(1)

    # Handle first day using calendar logic
    first_day = daily['date'].iloc[0]
    prev_day = first_day - pd.Timedelta(days=1)

    if prev_day.date() in holidays or prev_day.weekday() >= 5:
        first_prev_day_type = 'non_working'
    else:
        first_prev_day_type = 'working'

    daily.loc[daily.index[0], 'prev_day_type'] = first_prev_day_type

    # Create day_transition column
    daily['day_transition'] = daily['day_type'] + '_after_' + daily['prev_day_type']

    # Join back to original df
    df = df.merge(daily[['date', 'day_transition']], on='date', how='left')
    df = df.drop(columns=['date'])

    return df

def add_lag_features(df, lags, target_columns, remove_suffix=True):
    """
    Adds lag features to a DataFrame for one or more columns.
    Automatically removes the last '_suffix' from column names for lag naming.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - lags (list of int): List of lag values to create.
    - target_columns (list of str): Columns to create lag features for.
    - remove_suffix (bool): Whether to remove the last '_suffix' in column names for lag features.

    Returns:
    - pd.DataFrame: Modified DataFrame with lag features added.
    """
    max_lag = max(lags)
    
    for col in target_columns:
        # Determine the base name for lag columns
        base_col = col
        if remove_suffix and '_' in col:
            base_col = '_'.join(col.split('_')[:-1])
        
        # Add lag features
        for lag in lags:
            df[f'lag_{lag}'] = df[col].shift(lag)

    # Drop rows at the beginning that have NaNs due to lagging
    return df.iloc[max_lag:]

def preprocess_time_series(df, ds_col="ds", holiday_years=None, cz_start_year=2010, cz_end_year=2030,
                           lags=None, target_columns=None, remove_suffix=True):
    """
    Full preprocessing pipeline for time series data including lag features.

    This function:
    - Converts the datetime column `ds_col` to pandas datetime and sets it as index.
    - Generates cyclical seasonality features (hour, day of week, day of year, month).
    - Adds a year trend feature.
    - Adds Czech holiday dummy variables.
    - Adds a Christmas period dummy (27–31 December for specified years).
    - Adds working day / non-working day indicators.
    - Maps multiple holiday dummies into a single categorical holiday label.
    - Adds time-based features (DST, peak hours, etc.).
    - Adds lag features for specified target columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing a datetime column.
    ds_col : str, default "ds"
        Name of the datetime column to be used as index.
    holiday_years : list, set, or None
        Years to mark for Christmas period. Defaults to None.
    cz_start_year : int, default 2010
        Start year for Czech holiday generation.
    cz_end_year : int, default 2030
        End year for Czech holiday generation.
    lags : list of int, optional
        List of lag values to create for target columns.
    target_columns : list of str, optional
        Columns for which to generate lag features.
    remove_suffix : bool, default True
        Whether to remove the last '_' part in column names for naming lag features.

    Returns
    -------
    pandas.DataFrame
        DataFrame with all seasonality, holiday, working-day, time, and lag features added,
        and the datetime column set as index.
    """
    df = df.copy()
    df[ds_col] = pd.to_datetime(df[ds_col])
    df = df.set_index(ds_col, drop=False)
    df = create_seasonality_features(df)
    cz_holidays = holidays.Czechia(years=range(cz_start_year, cz_end_year + 1), language="cs")
    df = add_holiday_columns(df, cz_holidays)
    df = add_christmas_period(df, years=holiday_years)
    df = add_working_day_columns(df)
    df = map_holiday_and_collapse(df)
    df = categorize_holiday_type(df)
    df = add_time_features(df)
    df = map_peak_and_collapse(df)
    df = categorize_peak_type(df)
    df = combine_binary_columns(df, drop_binary_cols=True)
    df = add_day_transition_column(df, holidays=list(cz_holidays.keys()))
    df = add_lag_features(df, lags=lags, target_columns=target_columns, remove_suffix=remove_suffix)
    return df

#----------------------------------------------------------------------------------------------- Training -----------------------------------------------------------------------------
def filter_dataframe_by_date(df: pd.DataFrame, date_str: str, column: str = 'ds', inclusive: bool = True) -> pd.DataFrame:
    """
    Filter DataFrame rows by a specified datetime column based on the given date string.
    
    Args:
        df (pd.DataFrame): Input DataFrame with a datetime column.
        date_str (str): Date string to filter by, e.g. '2021-12-31 23:59:59'.
        column (str): Name of the datetime column to filter on. Default is 'ds'.
        inclusive (bool): If True, keep rows with column >= date_str,
                          else keep rows with column > date_str.
    
    Returns:
        pd.DataFrame: Filtered DataFrame copy.
    """
    if inclusive:
        filtered_df = df[df[column] >= date_str].copy()
    else:
        filtered_df = df[df[column] > date_str].copy()
    return filtered_df

def train_val_test_split(
    df: pd.DataFrame, 
    val_split_date: str, 
    test_split_date: str = None, 
    encoder_length: int = 0, 
    freq: str = '15min', 
    test_dataset: bool = True, 
    column: str = 'ds'
) -> tuple:
    """
    Splits a DataFrame into train, validation, and optional test sets for time series, 
    including encoder history and ensuring train and validation do not overlap.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with a datetime column.
    val_split_date : str or pd.Timestamp
        Start of validation period.
    test_split_date : str or pd.Timestamp, optional
        Start of test period; required if `test_dataset=True`.
    encoder_length : int, default 0
        Number of steps to include before val/test start for encoder history.
    freq : str, default '15min'
        Time series frequency for encoder calculation.
    test_dataset : bool, default True
        Whether to create a test dataset.
    column : str, default 'ds'
        Name of the datetime column.

    Returns
    -------
    tuple
        (train_df, val_df) or (train_df, val_df, test_df) if `test_dataset=True`.

    Notes
    -----
    - Train: strictly before `val_split_date - encoder_length*freq`.  
    - Validation: from `val_split_date - encoder_length*freq` to `test_split_date` or end.  
    - Test: from `test_split_date - encoder_length*freq` to end (if `test_dataset=True`).  
    - Train and validation do not overlap; validation and test may share encoder rows.
    """
    df = df.drop_duplicates().copy()
    if column not in df.columns:
        raise KeyError(f"The DataFrame does not contain a '{column}' column.")
    
    df[column] = pd.to_datetime(df[column])
    val_split_date = pd.to_datetime(val_split_date)
    if test_dataset and test_split_date is None:
        raise ValueError("test_split_date must be provided when test_dataset is True.")
    test_split_date = pd.to_datetime(test_split_date) if test_dataset else None

    encoder_delta = encoder_length * pd.Timedelta(freq)

    df_train = df[df[column] < val_split_date - encoder_delta]
    df_val = df[(df[column] >= val_split_date - encoder_delta) & 
                (df[column] < (test_split_date if test_dataset else df[column].max() + pd.Timedelta(freq)))]
    
    if test_dataset:
        df_test = df[df[column] >= test_split_date - encoder_delta]
        return df_train, df_val, df_test

    return df_train, df_val

def save_train_config(training, subdirectory_name):
    """
    Save the training object to the specified subdirectory.

    Args:
        training: The training object with a save method.
        subdirectory_name: The directory where files will be saved.
    """
    training.save(f"{subdirectory_name}/training_config.pkl")

def plot_loss(trainer, by="step", output_file=None):
    """
    Plots training and validation loss from the metrics logged by the trainer.

    Args:
        trainer: PyTorch Lightning Trainer object with a logger.
        by: "step" or "epoch" to choose the x-axis.
        output_file: Path to save the plot as an image file. Defaults to "loss_by_step.png" or "loss_by_epoch.png".
    """
    if by not in ["step", "epoch"]:
        raise ValueError('`by` must be either "step" or "epoch"')

    metrics_path = f"{trainer.logger.log_dir}/metrics.csv"
    metrics = pd.read_csv(metrics_path)

    train_col = f"train_loss_{by}"
    x_col = by

    # Drop rows where all values are NaN except x_col
    metrics = metrics.dropna(subset=[train_col, "val_loss"], how="all")

    plt.figure(figsize=(10, 6))

    if train_col in metrics.columns:
        train_loss = metrics[metrics[train_col].notnull()]
        plt.plot(train_loss[x_col], train_loss[train_col], label="Train Loss")

    if "val_loss" in metrics.columns:
        val_loss = metrics[metrics["val_loss"].notnull()]
        plt.plot(val_loss[x_col], val_loss["val_loss"], label="Validation Loss")

    plt.xlabel(by.capitalize())
    plt.ylabel("Loss")
    plt.title(f"Training and Validation Loss per {by.capitalize()}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if output_file is None:
        output_file = f"loss_by_{by}.png"

    plt.savefig(output_file, dpi=300)
    plt.close()

