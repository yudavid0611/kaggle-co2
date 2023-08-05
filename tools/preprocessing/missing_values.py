import pandas as pd


# 결측값 비율 확인
def get_missing_raio(df, threshold=0):
    # 전체 행의 개수
    n_rows = df.shape[0]

    # 칼럼별 결측값 개수
    n_missing_values = df.isna().sum()
    ratio_missing_values = n_missing_values / n_rows
    ratio_missing_values = ratio_missing_values[ratio_missing_values >= threshold]
    
    return ratio_missing_values


# 결측값 비율이 높은 칼럼 삭제
def delete_columns(df, threshold, target=None):
    ratio_missing_values = get_missing_raio(df)

    over_threshold_columns = list(ratio_missing_values[ratio_missing_values > threshold].index)

    # target 칼럼은 삭제할 칼럼 리스트에서 제거
    if target and target in over_threshold_columns:
        over_threshold_columns.remove(target)

    df = df.drop(over_threshold_columns, axis=1)

    return df


# 결측값 대체(numeric)
def impute(df, method=None, value=0):
    basic_methods = ['bfill', 'ffill', 'mean', 'median', 'value']
    
    df_numeric = df.select_dtypes(exclude=['O'])
    df_object = df.select_dtypes(include=['O'])

    if method in basic_methods:
        if method == 'mean':
            mean = df_numeric.mean()
            df_numeric.fillna(mean, inplace=True)
        elif method == 'median':
            median = df_numeric.median()
            df_numeric.fillna(median, inplace=True)
        elif method == 'value':
            df_numeric.fillna(value, inplace=True)
        else:
            df_numeric.fillna(method=method, inplace=True)
    
    # 선형 보간
    elif method == 'linear':
        df_numeric = df_numeric.interpolate(method=method, limit_direction='both')
    else:
        raise Exception("유효한 method가 아닙니다.")


    df_columns = list(df.columns)
    final_df = pd.concat([df_numeric, df_object], axis=1)
    final_df = final_df[df_columns]

    return final_df