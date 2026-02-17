import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

def dropNaNs(df):
    df = df.dropna()
    print("Dropped NaN values.")
    return df

def cleanNaNs(df1, df2):
    df1_check = df1.isnull().values.any()
    df2_check = df2.isnull().values.any()

    if df1_check:
        df1 = dropNaNs(df1)

    if df2_check:
        df2 = dropNaNs(df2)

    print("Dataset cleaned.")
    return df1, df2

def checkForNaNs(df1, df2):
    df1_check = df1.isnull().values.any()
    df2_check = df2.isnull().values.any()

    print(f"EmoSounds: {df1_check}")
    print(f"IADSED: {df2_check}")

def force_numeric(df, df_target):
    for c in df_target:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# Load datasets
emo_df = pd.read_csv('EmoSounds-3.csv')
iad_df = pd.read_csv('IADSED-2.csv')

# Specify target variables before checking for NaNs
emo_targets = ["arousal", "valence"]
iad_targets = ["arousal", "valence", "dominance"]

# Force numeric values for target variables before checking for NaNs to ensure that any non-numeric values are converted to NaN and can be handled properly
emo_df = force_numeric(emo_df, emo_targets)
iad_df = force_numeric(iad_df, iad_targets)

# Clean NaNs (drops rows with non-numeric targets)
emo_df, iad_df = cleanNaNs(emo_df, iad_df)

# Check for any remaining NaNs
checkForNaNs(emo_df, iad_df)

print("Creating Visualizations...")

fig, axes = plt.subplots(2, 3)

# EmoSounds visualizations
axes[0, 0].scatter(emo_df['arousal'], emo_df['valence'])
axes[0, 0].set_xlabel('Arousal')
axes[0, 0].set_ylabel('Valence')
axes[0, 0].set_title('EmoSounds: Arousal vs Valence')
axes[0, 0].grid(True)

axes[0, 1].hist(emo_df['arousal'], bins=30)
axes[0, 1].set_xlabel('Arousal')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('EmoSounds: Arousal Distribution')
axes[0, 1].grid(True)

axes[0, 2].hist(emo_df['valence'], bins=30)
axes[0, 2].set_xlabel('Valence')
axes[0, 2].set_ylabel('Frequency')
axes[0, 2].set_title('EmoSounds: Valence Distribution')
axes[0, 2].grid(True)

# IADSED visualizations
axes[1, 0].scatter(iad_df['arousal'], iad_df['valence'])
axes[1, 0].set_xlabel('Arousal')
axes[1, 0].set_ylabel('Valence')
axes[1, 0].set_title('IADSED: Arousal vs Valence')
axes[1, 0].grid(True)

axes[1, 1].hist(iad_df['arousal'], bins=30)
axes[1, 1].set_xlabel('Arousal')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('IADSED: Arousal Distribution')
axes[1, 1].grid(True)

axes[1, 2].hist(iad_df['dominance'], bins=30)
axes[1, 2].set_xlabel('Dominance')
axes[1, 2].set_ylabel('Frequency')
axes[1, 2].set_title('IADSED: Dominance Distribution')
axes[1, 2].grid(True)

plt.savefig('emotion_visualization.png')
print("Visualization saved as 'emotion_visualization.png'")

# Print insights
print("\nInsights:")
print(f"EmoSounds - Arousal mean: {emo_df['arousal'].mean():.3f}, std: {emo_df['arousal'].std():.3f}")
print(f"EmoSounds - Valence mean: {emo_df['valence'].mean():.3f}, std: {emo_df['valence'].std():.3f}")
print(f"IADSED - Arousal mean: {iad_df['arousal'].mean():.3f}, std: {iad_df['arousal'].std():.3f}")
print(f"IADSED - Valence mean: {iad_df['valence'].mean():.3f}, std: {iad_df['valence'].std():.3f}")
print(f"IADSED - Dominance mean: {iad_df['dominance'].mean():.3f}, std: {iad_df['dominance'].std():.3f}")

# Preprocessing with StandardScaler
print("\nPreprocessing Datasets...")

# EmoSounds preprocessing
non_feature_cols_emo = ['dataset', 'fnames', 'genre', 'splits', 'vocals', 'arousal', 'valence']
feature_cols_emo = [col for col in emo_df.select_dtypes(include=['number']).columns
                    if col not in non_feature_cols_emo]

emo_preprocessed = emo_df.copy()
scaler_emo = StandardScaler()
emo_preprocessed[feature_cols_emo] = scaler_emo.fit_transform(emo_df[feature_cols_emo])

print(f"EmoSounds: Normalized {len(feature_cols_emo)} acoustic features")

# IADSED preprocessing
non_feature_cols_iad = ['source', 'description', 'category', 'fname', 'BE_Classification',
                        'arousal', 'valence', 'dominance']
feature_cols_iad = [col for col in iad_df.select_dtypes(include=['number']).columns
                    if col not in non_feature_cols_iad]

iad_preprocessed = iad_df.copy()
scaler_iad = StandardScaler()
iad_preprocessed[feature_cols_iad] = scaler_iad.fit_transform(iad_df[feature_cols_iad])

print(f"IADSED: Normalized {len(feature_cols_iad)} acoustic features")

emo_preprocessed.to_csv('EmoSounds_preprocessed.csv', index=False)
iad_preprocessed.to_csv('IADSED_preprocessed.csv', index=False)
print("EmoSounds_preprocessed.csv saved")
print("IADSED_preprocessed.csv saved")
