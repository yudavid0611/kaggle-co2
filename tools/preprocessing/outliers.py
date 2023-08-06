import pandas as pd
from collections import defaultdict


# 이상치 상한, 하한 얻기
def get_limits(df, columns):
    limits = defaultdict(list)

    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        upper_limit = q3 + 1.5 * iqr
        lower_limit = q1 - 1.5 * iqr

        limits[col].extend([upper_limit, lower_limit])

    return limits


# 이상치 제거
def delete_outliers(df, columns, limits):
    deleted_indices = set()

    for col in columns:
        upper_limit, lower_limit = limits[col]
        indices = df.loc[(df[col] > upper_limit) | (df[col] < lower_limit)].index
        deleted_indices.update(list(indices))
    
    df_deleted = df.drop(deleted_indices)
    return df_deleted, deleted_indices


# 이상치 수정 함수
def replace_outliers(x, upper_limit, lower_limit):
    if x > upper_limit:
        return upper_limit
    elif x < lower_limit:
        return lower_limit
    else:
        return x
    

# 이상치 대체(by flooring and capping)
def impute_outliers(df, columns, limits):
    for col in columns:
        upper_limit, lower_limit = limits[col]
        df[col] = df[col].apply(replace_outliers, args=[upper_limit, lower_limit])

    return df