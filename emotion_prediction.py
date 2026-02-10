import pandas as pd
import matplotlib as mp

emo_df = pd.read_csv('EmoSounds-3.csv')
iad_df = pd.read_csv('IADSED-2.csv')

def dropNaNs(df):
    df = df.dropna()
    print("Dropped NaN values.")
    return df

def cleanNaNs(df1, df2):
    print("=====cleanNaNs called========")

    df1_check = df1.isnull().values.any()
    df2_check = df2.isnull().values.any()

    if df1_check:
        df1 = dropNaNs(df1)

    if df2_check:
        df2 = dropNaNs(df2)

    print("Dataset cleaned.")
    print("=============================")
    return df1, df2

def checkForNaNs(df1, df2):
    print("=====checkForNaNs called=====")
    df1_check = df1.isnull().values.any()
    df2_check = df2.isnull().values.any()

    print(f"EmoSounds: {df1_check}")
    print(f"IADSED: {df2_check}")
    print("=============================")

def force_numeric(df, df_target):
    for c in df_target:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

emo_df, iad_df = cleanNaNs(emo_df, iad_df)

checkForNaNs(emo_df,iad_df)

# after checking for NaNs, declare target variables and make sure to handle non-numeric values
emo_targets = ["arousal", "valence"]
iad_targets = ["arousal", "valence", "dominance"]

force_numeric(emo_df,emo_targets)
force_numeric(iad_df, iad_targets)


