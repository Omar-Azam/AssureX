"""
AssureX Claims Feature Engineering & Processing Pipeline
========================================================

Executes the exact feature engineering, categorical encoding, and export pipeline
defined in notebooks/claims_feature_engineering_eda.ipynb:
1. Loads train.csv, val.csv, test.csv
2. Computes 6 derived features:
   - product_age_days
   - remaining_warranty_days
   - missing_document_count
   - days_to_reporting_deadline
   - repair_count
   - has_any_contradiction
3. Encodes categorical variables:
   - product_category (One-Hot)
   - damage_type (Frequency & Target Risk Encoding fitted on train)
   - retailer (Frequency & Missing Indicator fitted on train)
4. Generates and saves:
   - train_features.csv
   - val_features.csv
   - test_features.csv
   Both in dataset/ and current working directory.
5. Saves visualization figures if matplotlib/seaborn are available.
"""

import os
import re
import argparse
import pandas as pd
import numpy as np
from claim_metrics import compute_claim_metrics, parse_date


def find_data_file(filename: str, dataset_dir: str = "dataset") -> str:
    """Locate data CSV across multiple candidate directories."""
    candidates = [
        os.path.join(dataset_dir, filename),
        filename,
        os.path.join("..", dataset_dir, filename),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(f"Could not find {filename} in {candidates}")


def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes domain-informed derived features:
    - product_age_days: purchase_date to claim_submission_date
    - remaining_warranty_days: warranty_expiry_date to fault_occurrence_date
    - missing_document_count: sum of missing checklist documents (0-4)
    - days_to_reporting_deadline: (fault_occurrence_date + 7d) - claim_submission_date
    - repair_count: integer extracted from repair_history
    - has_any_contradiction: boolean composite flag
    """
    df = df.copy()

    dt_purchase = pd.to_datetime(df['purchase_date'], errors='coerce')
    dt_fault = pd.to_datetime(df['fault_occurrence_date'], errors='coerce')
    dt_expiry = pd.to_datetime(df['warranty_expiry_date'], errors='coerce')
    dt_claim = pd.to_datetime(df['claim_submission_date'], errors='coerce')

    # 1 & 2: Canonical product_age_days and remaining_warranty_days (sole single source of truth)
    metrics_list = [
        compute_claim_metrics(row['purchase_date'], row['claim_submission_date'], row['warranty_duration_months'])
        for _, row in df.iterrows()
    ]
    df['product_age_days'] = [m.product_age_days for m in metrics_list]
    df['remaining_warranty_days'] = [m.remaining_warranty_days for m in metrics_list]

    # 3. missing_document_count
    doc_flags = ['has_receipt', 'has_warranty_card', 'has_product_image', 'has_serial_evidence']
    df['missing_document_count'] = 0
    for col in doc_flags:
        df['missing_document_count'] += (~df[col].astype(bool)).astype(int)

    # 4. days_to_reporting_deadline: (fault_date + 7 days) - claim_submission_date
    reporting_deadline = dt_fault + pd.to_timedelta(7, unit='D')
    df['days_to_reporting_deadline'] = (reporting_deadline - dt_claim).dt.days

    # 5. repair_count
    df['repair_count'] = df['repair_history'].astype(str).str.extract(r'(\d+)')[0].fillna(0).astype(int)

    # 6. has_any_contradiction: boolean flag
    chronology_mismatch = (dt_claim < dt_fault) | (dt_fault < dt_purchase) | (dt_claim < dt_purchase)

    sn_unit = df['serial_number'].astype(str).str.strip().str.upper()
    sn_rcpt = df['serial_number_on_receipt'].astype(str).str.strip().str.upper()
    serial_mismatch = (
        df['has_receipt'].astype(bool) &
        (df['serial_number_on_receipt'].isna() | (sn_rcpt == 'NAN') | (sn_rcpt == '') | (sn_rcpt != sn_unit))
    )

    prior_replacement_flag = df['prior_replacement'].astype(bool)

    desc_txt = df['fault_description'].astype(str).str.lower()
    type_txt = df['damage_type'].astype(str).str.lower()
    narrative_mismatch = (
        type_txt.str.contains('glitch|unspecified|wear', na=False) &
        desc_txt.str.contains('water|pool|spill|tea|drop|shatter|cracked', na=False)
    )

    df['has_any_contradiction'] = (
        chronology_mismatch | serial_mismatch | prior_replacement_flag | narrative_mismatch
    ).astype(bool)

    return df


def fit_and_apply_encodings(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame
):
    """
    Fits categorical encoders strictly on train_df to prevent data leakage,
    then transforms train, val, and test dataframes.
    """
    # 1. product_category: One-Hot Encoding
    known_categories = sorted(train_df['product_category'].dropna().unique())

    # 2. damage_type: Frequency & Target Risk Encoding
    damage_freq_map = (train_df['damage_type'].value_counts() / len(train_df)).to_dict()
    global_invalid_prior = (train_df['class_label'] == 'Invalid Claim').mean()
    damage_risk_map = (
        train_df.groupby('damage_type')
        .apply(lambda g: (g['class_label'] == 'Invalid Claim').sum() / len(g))
        .to_dict()
    )

    # 3. retailer: Frequency Encoding & Missing Flag
    retailer_freq_map = (train_df['retailer'].value_counts(dropna=True) / len(train_df)).to_dict()

    def transform(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for cat in known_categories:
            col_name = f"category_{re.sub(r'[^a-zA-Z0-9]+', '_', cat)}"
            df[col_name] = (df['product_category'] == cat).astype(int)

        df['damage_type_freq'] = df['damage_type'].map(damage_freq_map).fillna(0.0)
        df['damage_type_invalid_risk'] = df['damage_type'].map(damage_risk_map).fillna(global_invalid_prior)

        df['retailer_freq'] = df['retailer'].map(retailer_freq_map).fillna(0.0)
        df['retailer_is_missing'] = df['retailer'].isna().astype(int)
        return df

    return transform(train_df), transform(val_df), transform(test_df), known_categories


def export_feature_matrices(
    train_encoded: pd.DataFrame,
    val_encoded: pd.DataFrame,
    test_encoded: pd.DataFrame,
    output_dirs=["dataset", "."]
):
    """Assembles and writes finalized feature matrices to disk."""
    one_hot_cols = [c for c in train_encoded.columns if c.startswith("category_")]

    feature_columns = [
        'claim_id',
        'purchase_price',
        'warranty_duration_months',
        'product_age_days',
        'remaining_warranty_days',
        'missing_document_count',
        'days_to_reporting_deadline',
        'repair_count',
        'has_receipt',
        'has_warranty_card',
        'has_product_image',
        'has_serial_evidence',
        'prior_replacement',
        'has_any_contradiction',
        'damage_type_freq',
        'damage_type_invalid_risk',
        'retailer_freq',
        'retailer_is_missing',
        *one_hot_cols,
        'class_label'
    ]

    train_out = train_encoded[feature_columns].copy()
    val_out = val_encoded[feature_columns].copy()
    test_out = test_encoded[feature_columns].copy()

    # Cast boolean flags to 0/1 integers
    bool_cols = ['has_receipt', 'has_warranty_card', 'has_product_image', 'has_serial_evidence',
                 'prior_replacement', 'has_any_contradiction']
    for b in bool_cols:
        train_out[b] = train_out[b].astype(int)
        val_out[b] = val_out[b].astype(int)
        test_out[b] = test_out[b].astype(int)

    for out_dir in output_dirs:
        os.makedirs(out_dir, exist_ok=True)
        train_out.to_csv(os.path.join(out_dir, "train_features.csv"), index=False)
        val_out.to_csv(os.path.join(out_dir, "val_features.csv"), index=False)
        test_out.to_csv(os.path.join(out_dir, "test_features.csv"), index=False)

    print(f"[+] Feature matrices successfully exported:")
    print(f"    - train_features.csv: {train_out.shape[0]} rows, {train_out.shape[1]} cols")
    print(f"    - val_features.csv:   {val_out.shape[0]} rows, {val_out.shape[1]} cols")
    print(f"    - test_features.csv:  {test_out.shape[0]} rows, {test_out.shape[1]} cols")

    return train_out, val_out, test_out


def generate_plots_if_available(train_df: pd.DataFrame, figures_dir: str = "reports/figures"):
    """Render EDA plots as PNGs if matplotlib and seaborn are available."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        os.makedirs(figures_dir, exist_ok=True)
        sns.set_theme(style="whitegrid", palette="muted")
        class_colors = {'Valid Claim': '#10B981', 'Invalid Claim': '#EF4444', 'Manual Review': '#F59E0B'}

        # Plot 1: Class Balance
        plt.figure(figsize=(8, 5))
        ax = sns.countplot(x='class_label', data=train_df, palette=class_colors,
                           order=['Valid Claim', 'Invalid Claim', 'Manual Review'])
        plt.title("AssureX Training Set Class Distribution (N=1,050)", fontsize=13, weight='bold', pad=12)
        plt.xlabel("Decision Class", fontsize=11)
        plt.ylabel("Claims Count", fontsize=11)
        for p in ax.patches:
            h = int(p.get_height())
            ax.annotate(f"{h}", (p.get_x() + p.get_width() / 2., h / 2),
                        ha='center', va='center', color='white', weight='bold', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "class_balance.png"), dpi=200)
        plt.close()

        # Plot 2: Correlation Heatmap
        corr_df = train_df.copy()
        corr_df['is_valid'] = (corr_df['class_label'] == 'Valid Claim').astype(int)
        corr_df['is_invalid'] = (corr_df['class_label'] == 'Invalid Claim').astype(int)
        corr_df['is_manual_review'] = (corr_df['class_label'] == 'Manual Review').astype(int)
        cols = ['product_age_days', 'remaining_warranty_days', 'missing_document_count',
                'days_to_reporting_deadline', 'repair_count', 'has_any_contradiction',
                'purchase_price', 'warranty_duration_months', 'is_valid', 'is_invalid', 'is_manual_review']
        plt.figure(figsize=(12, 9))
        sns.heatmap(corr_df[cols].astype(float).corr(), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1.0, vmax=1.0)
        plt.title("Numeric Features Correlation Heatmap vs. Decisions", fontsize=13, weight='bold', pad=12)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "correlation_heatmap.png"), dpi=200)
        plt.close()

        # Plot 3: Boxplot of Product Age
        plt.figure(figsize=(9, 6))
        sns.boxplot(x='class_label', y='product_age_days', data=train_df, palette=class_colors,
                    order=['Valid Claim', 'Invalid Claim', 'Manual Review'])
        plt.title("Product Age (Days) by Claim Decision Class", fontsize=13, weight='bold', pad=12)
        plt.xlabel("Decision Class", fontsize=11)
        plt.ylabel("Product Age (Days)", fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "product_age_boxplot.png"), dpi=200)
        plt.close()

        print(f"[+] EDA plots generated and saved to {figures_dir}/")
    except ImportError:
        print("[!] Matplotlib/Seaborn not installed in local environment; plots skipped locally (available in Colab).")


def main():
    parser = argparse.ArgumentParser(description="Run AssureX Claims Feature Engineering")
    parser.add_argument("--dataset-dir", type=str, default="dataset", help="Input dataset directory")
    args = parser.parse_args()

    train_path = find_data_file("train.csv", args.dataset_dir)
    val_path = find_data_file("val.csv", args.dataset_dir)
    test_path = find_data_file("test.csv", args.dataset_dir)

    print(f"[*] Loading datasets from {args.dataset_dir}...")
    train_raw = pd.read_csv(train_path)
    val_raw = pd.read_csv(val_path)
    test_raw = pd.read_csv(test_path)

    print("[*] Computing derived features...")
    train_fe = compute_derived_features(train_raw)
    val_fe = compute_derived_features(val_raw)
    test_fe = compute_derived_features(test_raw)

    print("[*] Applying categorical encodings...")
    train_enc, val_enc, test_enc, _ = fit_and_apply_encodings(train_fe, val_fe, test_fe)

    print("[*] Exporting finalized feature matrices...")
    export_feature_matrices(train_enc, val_enc, test_enc, output_dirs=["dataset", "."])

    print("[*] Checking visualization capabilities...")
    generate_plots_if_available(train_fe)

    print("\n[SUCCESS] Feature engineering pipeline completed successfully!")


if __name__ == "__main__":
    main()
