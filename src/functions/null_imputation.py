import pickle 


def impute_null_data(df, path=None):
    # Filter out continuous columns with cardinality > 9
    continuous_cols = df.select_dtypes(exclude=['object'])
    selected_cont_cols = continuous_cols.columns[continuous_cols.nunique() > 9]

    # Filter out categorical columns with cardinality < 9
    categorical_cols = df.select_dtypes(include=['int64', 'object'])
    selecte_cat_cols = categorical_cols.columns[categorical_cols.nunique() < 9]

    # Impute missing values in continuous columns with median
    df[selected_cont_cols] = df[selected_cont_cols].fillna(df[selected_cont_cols].median())

    # Impute missing values in categorical columns with mode
    for col in selecte_cat_cols:
        mode_val = df[col].mode().iloc[0]
        df[col] = df[col].fillna(mode_val)

    # Save the new data frame
    if path:
        with open(path, 'wb') as file:
            pickle.dump(df, file)
            
    return df