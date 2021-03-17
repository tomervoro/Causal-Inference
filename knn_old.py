import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm

DATASET_PATH = r"archive\covid.csv"


features = [#'age',
            'diabetes', 'copd', 'asthma',
            'inmsupr', 'hypertension', 'other_disease',
            'cardiovascular', 'obesity', 'renal_chronic',
            'tobacco']

final_outcomes = ['patient_type',
                 'intubed',
                 'pneumonia',
                 'icu',
                'date_died'
                 ]
date_outcomes = ['entry_date', 'date_symptoms']


def calc_error(row, row_not):
    # age_1 = row['age']
    # age_2 = row_not['age']
    # age_dif = age_1 - age_2
    # age_err = np.power(age_dif,2)
    row = row.to_numpy()
    row_not = row_not.to_numpy()
    err = (row != row_not).sum()
    return err # + age_err


def search_for_closest(row, not_pregnant, K):
    row_ft = row.loc[features]
    #age = row.loc["age"]

    lst_errors = []
    for index, row_not_ft in not_pregnant.iterrows():
        err = calc_error(row_ft, row_not_ft)
        if len(lst_errors) < K:
            lst_errors.append([err, index])
            lst_errors.sort(key=lambda x: x[0])
        else:
            if err >= lst_errors[-1][0]:
                continue
            lst_errors.pop(len(lst_errors)-1)
            lst_errors.append([err, index])
            lst_errors.sort(key=lambda x: x[0])


    return lst_errors



def calc_avg(lst_errors, not_pregnant):
    try:
        size = len(final_outcomes)
        ITE = np.zeros(size)
        cnt = np.zeros(size)

        for err, index in lst_errors:
            outcomes = not_pregnant.loc[index]
            outcomes = outcomes.loc[final_outcomes]
            outcomes = outcomes.to_numpy()
            for idx, feat in enumerate(outcomes):
                if feat == 97 or feat == 99 or feat == 98:
                    print(f" Encountered Weird Value! feat = {feat}")
                    continue
                if feat not in [1,2]:
                    print(f"bad feat {feat}")
                ITE[idx] += feat
                cnt[idx] += 1

        for idx, c in enumerate(cnt):
            if c == 0:
                cnt[idx] = 1

        ITE = np.divide(ITE, cnt)
        return ITE

    except:
        return None


def print_and_log(st: str):
    with open("log.txt", "a") as f:
        f.write(st+"\n")
        print(st)


def calc_ATE(pregnant, not_pregnant, K):
    not_pregnant_features = not_pregnant[features]
    cont = np.zeros(len(final_outcomes))
    cnt = 0
    ATE = np.zeros(len(final_outcomes))
    for index, row in tqdm(pregnant.iterrows()):
        lst_errors = search_for_closest(row, not_pregnant_features, K)
        ite = calc_avg(lst_errors, not_pregnant)
        # print(f" The current ite is: {ite}")
        if ite is None:
            continue
        pregnant_outcome = row.loc[final_outcomes]
        pregnant_outcome = pregnant_outcome.to_numpy()
        poc_ite = np.zeros(len(final_outcomes))

        for idx, feat in enumerate(pregnant_outcome):
            if feat == 97 or feat == 99 or feat == 98:
                continue
            if feat not in [1, 2]:
                print(f"bad feat {feat}")
            poc_ite[idx] += feat - ite[idx]
            cont[idx] += 1

        for idx, c in enumerate(cont):
            if c == 0:
                cont[idx] = 1
        ATE = np.add(ATE, poc_ite)
        cnt += 1
        if cnt % 10 == 0:
            temp = np.divide(ATE, cont)
            print(f"Current ATE: {temp}")
    ATE = np.divide(ATE, cont)
    return ATE

def main():
    K = 1
    # def main(K):
    df = pd.read_csv(DATASET_PATH)
    df = df.loc[df['sex'] == 1]  # take only females
    df = df.loc[df['covid_res'] == 1]  # take only positives
    df = df.drop('sex', axis=1)  # drop sex
    df = df.drop('id', axis=1)  # drop id
    df = df.loc[(df['age'] <= 45) & (df['age'] >= 18)]
    # age_range = 45 - 18
    # means = df.mean(axis=0)
    # age_avarage = means['age']
    # df['age'] = df['age'] - age_avarage
    # df['age'] = df['age'].div(age_range)
    #  Not pregnant data adaptation
    not_pregnant = df.loc[(df['pregnancy'] == 2)]
    not_pregnant.loc[not_pregnant.date_died == "9999-99-99", 'date_died'] = 2
    not_pregnant.loc[not_pregnant.date_died != 2, 'date_died'] = 1
    not_pregnant.loc[not_pregnant.intubed == 97, 'intubed'] = 2

    #  Pregnant data adaptation
    pregnant = df.loc[(df['pregnancy'] == 1)]
    pregnant.loc[pregnant.date_died == "9999-99-99", 'date_died'] = 2
    pregnant.loc[pregnant.date_died != 2, 'date_died'] = 1
    pregnant.loc[pregnant.intubed == 97, 'intubed'] = 2
    not_pregnant = not_pregnant.loc[(not_pregnant['patient_type'].isin([1, 2])) & (not_pregnant['intubed'].isin([1, 2]))
                                    & (not_pregnant['pneumonia'].isin([1, 2])) & (not_pregnant['icu'].isin([1, 2]))]
    pregnant = pregnant.loc[(pregnant['patient_type'].isin([1, 2])) & (pregnant['intubed'].isin([1, 2]))
                            & (pregnant['pneumonia'].isin([1, 2])) & (pregnant['icu'].isin([1, 2]))]
    ATE = np.zeros(len(final_outcomes))
    cnt = 0
    num_rows = pregnant.shape[0]

    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print_and_log("\n\n---------------")
    print_and_log(f"Time: {dt_string}")
    print_and_log(f"K: {K}")
    print_and_log(f"Total pregnant rows: {num_rows}")
    print_and_log(f"Total not_pregnant rows: {not_pregnant.shape[0]}")
    not_pregnant_features = not_pregnant[features]
    cont = np.zeros(len(final_outcomes))

    ATE = calc_ATE(pregnant=pregnant, not_pregnant=not_pregnant, K=K)
    ATE += calc_ATE(pregnant=not_pregnant, not_pregnant=pregnant, K=K)
    """for index, row in tqdm(pregnant.iterrows()):
        lst_errors = search_for_closest(row, not_pregnant_features, K)
        ite = calc_avg(lst_errors, not_pregnant)
        # print(f" The current ite is: {ite}")
        if ite is None:
            continue
        pregnant_outcome = row.loc[final_outcomes]
        pregnant_outcome = pregnant_outcome.to_numpy()
        poc_ite = np.zeros(len(final_outcomes))

        for idx, feat in enumerate(pregnant_outcome):
            if feat == 97 or feat == 99 or feat == 98:
                continue
            if feat not in [1, 2]:
                print(f"bad feat {feat}")
            poc_ite[idx] += feat - ite[idx]
            cont[idx] += 1

        for idx, c in enumerate(cont):
            if c == 0:
                cont[idx] = 1
        ATE = np.add(ATE, poc_ite)
        cnt += 1
        if cnt % 10 == 0:
            temp = np.divide(ATE, cont)
            print(f"Current ATE: {temp}")

    pregnant_features = pregnant[features]
    for index, row in tqdm(pregnant.iterrows()):
        lst_errors = search_for_closest(row, pregnant_features, K)
        ite = calc_avg(lst_errors, pregnant)
        # print(f" The current ite is: {ite}")
        if ite is None:
            continue
        not_pregnant_outcome = row.loc[final_outcomes]
        not_pregnant_outcome = not_pregnant_outcome.to_numpy()
        poc_ite = np.zeros(len(final_outcomes))
        for idx, feat in enumerate(not_pregnant_outcome):
            if feat == 97 or feat == 99 or feat == 98:
                continue
            if feat not in [1, 2]:
                print(f"bad feat {feat}")
            poc_ite[idx] += feat - ite[idx]
            cont[idx] += 1

        for idx, c in enumerate(cont):
            if c == 0:
                cont[idx] = 1
        ATE = np.add(ATE, poc_ite)
        cnt += 1
        if cnt % 10 == 0:
            temp = np.divide(ATE, cont)
            print(f"Current ATE: {temp}")
            # print(f"temp outcome ({cnt}/{num_rows}: {temp}")

    ATE = np.divide(ATE, cont)"""
    print_and_log(f"final outcome: {ATE}")

if __name__ == "__main__":
    main()
