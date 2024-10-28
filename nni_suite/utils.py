import os
import pandas as pd
import sqlite3
from pathlib import Path
import json


class SearchSpaceUpdater(object):
    def __init__(self, *initial_data, **kwargs):
        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])


def retrieve_nni_results(
        exp_name,
        exp_id,
        metrics,
        trial_id=None,
        max_trial_num=10000,
        nni_default_path=True,
        export_best_params=True,
        working_directory='..'
):

    db_path = os.path.expanduser(f"{nni_default_path}/{exp_name}/{exp_id}/db")

    params = []

    base_query = """
    SELECT m.timestamp, t.trialJobId, t.data AS params, m.type, m.data AS results
    FROM TrialJobEvent AS t
    INNER JOIN MetricData AS m ON t.trialJobId = m.trialJobId
    WHERE m.type = 'FINAL' AND t.event = 'RUNNING' 
    """

    if trial_id is not None:
        base_query += " AND t.trialJobId = (?);"
        params.append(trial_id)
    else:
        base_query += "AND t.sequenceId <= (?);"
        params.append(max_trial_num)

    con = sqlite3.connect(os.path.join(db_path, "nni.sqlite"))  # sqlite connector

    # Load the data into a DataFrame
    trial_data = pd.read_sql_query(base_query, con, params=params)

    # Process top10 results of default = test accuracy
    results_data = pd.json_normalize(trial_data['results'].map(eval).map(eval))
    params_data = pd.json_normalize(trial_data['params'].map(eval))

    df_trial = pd.concat([trial_data.drop(['results', 'params'], axis=1), params_data, results_data], axis=1)

    top = df_trial.sort_values(by=[metrics], ascending=False).head(1)

    parameter_columns = df_trial.filter(regex='^parameters').columns
    df_parameters = top[parameter_columns]
    df_parameters.columns = [col.replace('parameters.', '') for col in df_parameters.columns]
    json_strings = [row.to_json() for _, row in df_parameters.iterrows()]
    json_string = '\n'.join(json_strings)

    # Parse each JSON string into a JSON object
    params = [json.loads(js) for js in json_strings][0]

    # Print or save the JSON string
    if export_best_params:
        folder_path = Path(f'{working_directory}/best_parameters/{exp_name}')
        folder_path.mkdir(parents=True, exist_ok=True)
        filename = f'parameters_{exp_id}_{top.iloc[0]["trialJobId"]}.json'
        with open(folder_path / filename, 'w') as file:
            file.write(json_string)

    con.close()

    return top.iloc[0]["trialJobId"], top.iloc[0][metrics], params
