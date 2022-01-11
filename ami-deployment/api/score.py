import json
import os.path

import joblib

import pandas as pd
import numpy as np
import datetime
import math
import shap

VERSION = "0.1"

VIEW_VERSION_INFO_CMD = 'VIEW VERSION INFO'  # Used by client to view required inputs

# The biomarker names below are derived from the conversions.yml in ami_packages.
REQUIRED_BIOMARKERS = {
    'hsTnI'
}

REQUESTED_BIOMARKERS = {
    'BUN',
    'lymph_leu_non_variant',
    'glome_M.p',
    'Basophils.val',
    'Eos',
    'Neu/PMN/polys',
    'RDW.val',
    'BnP',
    'heart_rate',
    'body_weight',
    'diastolic_bp',
}

# The formula for feature extraction expects a group with tests consisting of the type given by the key. It is sorted by abs_time_since_first_trp ascending
# The first element of the tuple is the function that extracts the feature, the second is the name of the feature.
EXTRACTION_FORMULA_DICT = {
    'hsTnI': (lambda x: x.iloc[0].result_value, 'FIRST_TRP'),
    'BUN': (lambda x: x.iloc[0].result_value, 'BUN'),
    'lymph_leu_non_variant': (lambda x: x.iloc[0].result_value, 'lymph_leu_non_variant'),
    'glome_M.p': (lambda x: x.iloc[0].result_value, 'glome_M.p'),
    'Basophils.val': (lambda x: x.iloc[0].result_value, 'Basophils.val'),
    'Eos': (lambda x: x.iloc[0].result_value, 'Eos'),
    'Neu/PMN/polys': (lambda x: x.iloc[0].result_value, 'Neu/PMN/polys'),
    'RDW.val': (lambda x: x.iloc[0].result_value, 'RDW.val'),
    'BnP': (lambda x: x.iloc[0].result_value, 'BnP'),
    'heart_rate': (lambda x: np.max(x.result_value), 'HEART_RATE_MAX'),
    'body_weight': (lambda x: np.min(x.result_value), 'BODY_WEIGHT_MIN'),
    'diastolic_bp': (lambda x: np.max(x.result_value), 'DIASTOLIC_BLOOD_PRESSURE_MAX'),
}


def generate_specification_json():
    """Generate the input format required by the client. Also provides loinc/biomarker map for convinience
    """
    out_dict = {'version': VERSION, 'required_biomarkers': list(REQUIRED_BIOMARKERS),
                'optional_biomarkers': list(REQUESTED_BIOMARKERS)}
    sub_bio_df = biomarker_df[biomarker_df.alias.apply(lambda x: x in REQUESTED_BIOMARKERS.union(REQUIRED_BIOMARKERS))]
    out_dict['loinc_table'] = {}

    for row in sub_bio_df.itertuples():
        if row.alias not in out_dict['loinc_table']:
            out_dict['loinc_table'][row.alias] = set()
        out_dict['loinc_table'][row.alias].add(row.loinc_code)
    for bio in sub_bio_df.alias.unique():
        out_dict['loinc_table'][bio] = list(out_dict['loinc_table'][bio])

    return json.dumps(out_dict)


def verify_json(raw_input, out_dict):
    """Ensures the validity of the supplied input. Errors and warnings are logged in out_dict
    """
    error_list = out_dict['CommonMessage']['Errors']
    try:
        input_dict = json.loads(raw_input)
    except:
        error_list.append('Invalid json format.')
        return None

    if 'ModelReq' not in input_dict:
        error_list.append('Key "ModelReq" not in json.')
        return None

    if not isinstance(input_dict['ModelReq'], dict):
        error_list.append('ModelReq is not a dictionary.')
    else:
        req_dict = input_dict['ModelReq']

    if 'ModelTrackerID' not in req_dict:
        error_list.append('ModelTrackerID not provided.')
    elif not isinstance(req_dict['ModelTrackerID'], str):
        error_list.append('ModelTrackerID is an invalid type.')
    else:
        out_dict['ModelRes']['ModelTrackerID'] = req_dict['ModelTrackerID']

    if 'Age' not in req_dict:
        error_list.append('Age not provided')
    elif not isinstance(req_dict['Age'], int):
        error_list.append('Age is not an int.')
    elif req_dict['Age'] < 18:
        error_list.append('Age <18')

    if 'Gender' not in req_dict:
        error_list.append('Gender not provided')
    elif not isinstance(req_dict['Gender'], str):
        error_list.append('Gender not appropriate type')

    if 'Data' not in req_dict:
        error_list.append('Data not provided.')
    elif not isinstance(req_dict['Data'], list):
        error_list.append('Data is not a list.')

    for entry in req_dict['Data']:
        if 'Code' not in entry:
            error_list.append('Code not provided for test.')
        if 'Results' not in entry:
            error_list.append('Results not provided test.')
        if not isinstance(entry['Results'], list):
            error_list.append('Provided Results is not a list.')
        else:
            for res in entry['Results']:
                if 'result_value' not in res or 'result_time' not in res or 'result_unit' not in res:
                    code = entry['Code']
                    error_list.append(f'Test with code={code} is missing a value.')

    return input_dict


def json_to_loinc_df(data_list):
    """Converts and input json file to a long dataframe.
    """
    out_dict = {'loinc_code': [], 'result_time': [], 'result_value': [], 'result_unit': []}

    for entry in data_list:
        loinc_code = entry['Code']
        for res in entry['Results']:
            out_dict['loinc_code'].append(loinc_code)
            out_dict['result_value'].append(res['result_value'])
            out_dict['result_unit'].append(res['result_unit'])
            out_dict['result_time'].append(res['result_time'])

    out_df = pd.DataFrame(out_dict).merge(biomarker_df, how='left')
    out_df.result_time = pd.to_datetime(out_df.result_time)
    return out_df


def init():
    global biomarker_df, model, n_sample
    cwd = os.path.dirname(__file__)
    n_sample = 1856
    biomarker_df = pd.read_csv(os.path.join(cwd, 'biomarker.csv'))
    model_path = os.path.join(cwd, '../models/ami_predictor_v0.0.1.pkl')
    model = joblib.load(model_path)


# wilson score interval
def wilson_ci(p, n, z=1.96):
    """
        Binomial proportion confidence interval based on Wilson method.
        https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Wilson_score_interval
        p - sampled proportion (probability)
        n - sample size (training set size)
        z - standard norm ppf corresponding to alpha, e.g. z=1.96 corresponds to alpha=0.05
            z = scipy.stats.norm.ppf(1 - alpha / 2)
        output:
        ci - confidence interval
    """
    # the center of the interval is biased and the bias reduces as the sample size grows
    ci_center = (p + (z ** 2) / (2 * n)) / (1 + (z ** 2) / n)
    # half interval size
    delta = z / (1 + (z ** 2) / n) * math.sqrt(p * (1 - p) / n + (z ** 2) / (4 * n ** 2))
    ci = ci_center + delta * np.array([-1, 1])
    return ci


def prediction(feature_df):
    score_dict = {"Name": "Score", "Value": "N/A"}
    range_dict = {"Name": "ProbabilityRange", "Value": "N/A"}
    positive_contributors_dict = {"Name": "PositiveContributors", "Value": "N/A"}
    negative_contributors_dict = {"Name": "NegativeContributors", "Value": "N/A"}

    feature_names = model.get_booster().feature_names
    # align to the same order of features
    feature_df = feature_df.reindex(columns=feature_names)
    result = model.predict_proba(feature_df)

    # probability for positive class
    probability = result[0, 1]
    score_dict['Value'] = '{0:.2f}%'.format(probability * 100)

    probability_range = wilson_ci(probability, n_sample)
    range_dict['Value'] = '({0:.2f}%, {1:.2f}%)'.format(probability_range[0] * 100, probability_range[1] * 100)

    # shap value explanation
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(feature_df)[0]
    contributions = shap_values / sum(abs(shap_values))

    positive_contributors, negative_contributors = [], []
    desc_idx = shap_values.argsort()[::-1]
    for order, idx in enumerate(desc_idx[:3]):
        if shap_values[idx] > 0:
            positive_contributors.append(" {0}: {1}%".format(feature_names[idx], int(contributions[idx] * 100)))
    positive_contributors_dict['Value'] = ','.join(positive_contributors).strip()
    asc_idx = shap_values.argsort()
    for order, idx in enumerate(asc_idx[:3]):
        if shap_values[idx] < 0:
            negative_contributors.append(" {0}: {1}%".format(feature_names[idx], int(contributions[idx] * 100)))
    negative_contributors_dict['Value'] = ','.join(negative_contributors).strip()

    return [score_dict, range_dict, positive_contributors_dict, negative_contributors_dict]


def run(raw_input):
    raw_input = str(raw_input)
    if raw_input == VIEW_VERSION_INFO_CMD:
        return generate_specification_json()

    out_dict = {
        'CommonMessage': {
            'TimeStamp': str(datetime.datetime.utcnow()),
            'TrackingID': "????",
            'Errors': [],
            'Warnings': []
        },
        'ModelRes': {
            "ModelTrackerID": "",
            "Version": VERSION,
            "Data": []
        }
    }

    input_dict = verify_json(raw_input, out_dict)

    if len(out_dict['CommonMessage']['Errors']) > 0:
        return json.dumps(out_dict)

    l_df = json_to_loinc_df(input_dict['ModelReq']['Data'])

    # Provide warnings when filtering rows with missing values.
    for row in l_df.itertuples():
        if pd.isnull(row.result_value) or pd.isnull(row.result_unit) or pd.isnull(row.result_time):
            out_dict['CommonMessage']['Warnings'].append(
                f'Test missing values. Entry will be ignored. loinc_code={row.loinc_code} time={row.result_time}')
    l_df = l_df[~l_df.result_value.isna() & ~l_df.result_unit.isna() & ~l_df.result_unit.isna()]

    # Stop and report a fatal error if an essential biomarker is missing
    for bio in REQUIRED_BIOMARKERS:
        if bio not in l_df.alias.unique():
            out_dict['CommonMessage']['Errors'].append(f'Biomarker {bio} not found.')
            return json.dumps(out_dict)

    # Provide warnings while filtering rows outside of valid time frames.
    first_trp_time = l_df[l_df.alias == 'hsTnI'].sort_values('result_time').iloc[0].result_time
    l_df['time_since_first_trp'] = (l_df.result_time - first_trp_time).apply(lambda x: x.total_seconds() / 3600)
    for row in l_df.itertuples():
        if row.time_since_first_trp < -2 or row.time_since_first_trp > 0.5:
            msg = f'Test with loinc code {row.loinc_code} at time {row.result_time} outside of input range. Will be ignored.'
            out_dict['CommonMessage']['Warnings'].append(msg)
    l_df = l_df[(l_df.time_since_first_trp <= 0.5) & (l_df.time_since_first_trp >= -2)]

    # Ensure filtering didnt remove required biomarkers.
    for bio in REQUIRED_BIOMARKERS:
        if bio not in l_df.alias.unique():
            out_dict['CommonMessage']['Errors'].append(f'Biomarker {bio} not found after time filtering.')
            return json.dumps(out_dict)

    # Report all missing, optional biomarkers.
    for bio in REQUESTED_BIOMARKERS:
        if bio not in l_df.alias.unique():
            out_dict['CommonMessage']['Warnings'].append(f'Biomarker {bio} not found.')

    def to_features(group_df):
        """Converts and alias group to a feature df. Input should already be sorted by abs_time_since_first_trp
        """
        alias = group_df.iloc[0].alias
        feat_func, new_name = EXTRACTION_FORMULA_DICT[alias]
        value = feat_func(group_df)
        return pd.DataFrame({'feature': [new_name], 'value': [value]})

    l_df['abs_time_since_first_trp'] = np.abs(l_df.time_since_first_trp)
    feature_df = l_df.sort_values('abs_time_since_first_trp').groupby('alias').apply(to_features).reset_index(
        drop=True).set_index('feature').T

    # Add empty columns to missing values
    all_features = {v[1] for k, v in EXTRACTION_FORMULA_DICT.items()}
    for feat in all_features:
        if feat not in feature_df.columns:
            feature_df[feat] = np.nan

    feature_df['AGE'] = input_dict['ModelReq']['Age']
    feature_df['GENDER'] = 1 if input_dict['ModelReq']['Gender'] == 'F' else 0

    data_dict = prediction(feature_df)
    out_dict['ModelRes']['Data'] = data_dict

    return json.dumps(out_dict)
