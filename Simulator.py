# =============================================================================
# DBbun LLC — Executable Publication Layer
# Tool      : paper_to_simulator_builder  v3.4.0
# Generated : 2026-05-28T15:38:11.297654Z
# Run ID    : 70c65e7e-d234-4a15-b669-337d4bbf1bbc
#
# © 2026 DBbun LLC. All rights reserved.  |  dbbun.com
# CAGE: 16VU3  |  UEI: QY39Y38E6WG8  |  Cambridge, MA, USA
#
# This simulator, synthetic datasets, and all derived intellectual property
# are the exclusive property of DBbun LLC.  Unauthorised reproduction,
# distribution, or commercial use is prohibited without prior written consent.
# =============================================================================
#
"""
Measurement Uncertainty in EMR-Derived Clinical AI Pipelines Simulator

Paper titles:
  - Reproducibility in Clinical AI Requires Modeling Measurement Error
  - Measurement Uncertainty in EMR-Derived Clinical AI Pipelines (Figure 1)
  - HL7 AI Challenge 2026 Simulator Prompt: Measurement Uncertainty in Clinical AI

This simulator models how pre-analytical errors, calibration drift, and threshold-based
diagnostic definitions propagate through EMR pipelines to corrupt AI training labels.
"""

import argparse
import csv
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ─────────────────────────────────────────────
# MODEL PROFILE
# ─────────────────────────────────────────────
MODEL_PROFILE = {
    "parameters": {
        "potassium_critical_threshold": {"value": 6.0, "unit": "mmol/L", "source": "extracted",
            "description": "Critical high potassium alert threshold triggering urgent clinical evaluation"},
        "potassium_analytical_sd": {"value": 0.15, "unit": "mmol/L", "source": "assumed",
            "description": "SD of analytical random error for potassium measurement"},
        "hemolysis_probability": {"value": 0.08, "unit": "fraction", "source": "assumed",
            "description": "Prevalence of hemolyzed specimens in routine phlebotomy"},
        "hemolysis_bias_mean": {"value": 1.8, "unit": "mmol/L", "source": "assumed",
            "description": "Mean upward potassium bias from hemolysis"},
        "hemolysis_bias_sd": {"value": 0.5, "unit": "mmol/L", "source": "assumed",
            "description": "SD of hemolysis-induced potassium bias"},
        "hba1c_diabetes_threshold": {"value": 6.5, "unit": "%", "source": "extracted",
            "description": "ADA diagnostic threshold for diabetes based on HbA1c"},
        "hba1c_analytical_cv": {"value": 0.02, "unit": "fraction", "source": "assumed",
            "description": "CV for HbA1c inter-assay analytical error"},
        "hba1c_calibration_bias": {"value": 0.1, "unit": "%", "source": "assumed",
            "description": "Systematic inter-platform calibration bias for HbA1c"},
        "hba1c_physiologic_sd": {"value": 0.15, "unit": "%", "source": "assumed",
            "description": "Intra-individual biological variation of HbA1c"},
        "sbp_hypertension_threshold": {"value": 130.0, "unit": "mmHg", "source": "extracted",
            "description": "ACC/AHA 2017 Stage 1 hypertension systolic BP threshold"},
        "sbp_device_sd": {"value": 5.0, "unit": "mmHg", "source": "assumed",
            "description": "SD of oscillometric BP device random measurement error"},
        "sbp_white_coat_mean": {"value": 14.0, "unit": "mmHg", "source": "assumed",
            "description": "Mean white-coat systolic BP elevation"},
        "sbp_white_coat_sd": {"value": 6.0, "unit": "mmHg", "source": "assumed",
            "description": "Inter-individual SD of white-coat BP effect"},
        "white_coat_probability": {"value": 0.2, "unit": "fraction", "source": "assumed",
            "description": "Fraction of patients exhibiting white-coat hypertension"},
        "sbp_physiologic_sd": {"value": 8.0, "unit": "mmHg", "source": "assumed",
            "description": "Intra-individual physiologic variability of systolic BP"},
        "calibration_drift_rate": {"value": 0.001, "unit": "per draw", "source": "assumed",
            "description": "Linear drift rate of instrument calibration factor per measurement"},
        "n_patients": {"value": 2000, "unit": "patients", "source": "assumed",
            "description": "Number of simulated patients per scenario"},
        "n_serial_draws": {"value": 20, "unit": "draws", "source": "assumed",
            "description": "Number of serial measurement occasions per patient"},
        "borderline_window_k": {"value": 0.5, "unit": "mmol/L", "source": "assumed",
            "description": "Half-width of borderline zone around potassium threshold"},
        "borderline_window_hba1c": {"value": 0.3, "unit": "%", "source": "assumed",
            "description": "Half-width of borderline zone around HbA1c threshold"},
        "borderline_window_sbp": {"value": 10.0, "unit": "mmHg", "source": "assumed",
            "description": "Half-width of borderline zone around SBP threshold"},
        "n_sites": {"value": 4, "unit": "sites", "source": "assumed",
            "description": "Number of simulated clinical sites"},
        "site_calibration_bias_sd": {"value": 0.08, "unit": "fraction", "source": "assumed",
            "description": "SD of site-level multiplicative calibration bias factors"},
    },
    "fhir_constants": {
        "LOINC_K": "2823-3",
        "LOINC_HBA1C": "4548-4",
        "LOINC_SBP": "8480-6",
        "UCUM_K": "mmol/L",
        "UCUM_HBA1C": "%",
        "UCUM_SBP": "mm[Hg]",
        "SNOMED_DIABETES": "73211009",
        "SNOMED_HYPERTENSION": "38341003",
    },
    "scenarios": [
        {
            "label": "Scenario_1_False_Critical_Potassium",
            "param_overrides": {
                "true_potassium_mean": 4.2, "true_potassium_sd": 0.4,
                "hemolysis_probability": 0.08, "hemolysis_bias_mean": 1.8,
                "hemolysis_bias_sd": 0.5, "potassium_analytical_sd": 0.15,
                "potassium_critical_threshold": 6.0, "n_patients": 2000,
            }
        },
        {
            "label": "Scenario_2_Borderline_HbA1c_Diabetes_Label",
            "param_overrides": {
                "true_hba1c_mean": 6.35, "true_hba1c_sd": 0.25,
                "hba1c_analytical_cv": 0.02, "hba1c_physiologic_sd": 0.15,
                "hba1c_calibration_bias": 0.1, "hba1c_diabetes_threshold": 6.5,
                "n_serial_draws": 20, "n_patients": 2000,
            }
        },
        {
            "label": "Scenario_3_Borderline_SBP_Hypertension_Label",
            "param_overrides": {
                "true_sbp_mean": 128.0, "true_sbp_sd": 8.0,
                "sbp_device_sd": 5.0, "sbp_physiologic_sd": 8.0,
                "white_coat_probability": 0.20, "sbp_white_coat_mean": 14.0,
                "sbp_white_coat_sd": 6.0, "sbp_hypertension_threshold": 130.0,
                "n_serial_draws": 20, "n_patients": 2000,
            }
        },
        {
            "label": "Scenario_4_Calibration_Drift_Cross_Site",
            "param_overrides": {
                "n_sites": 4, "site_calibration_bias_sd": 0.08,
                "true_hba1c_mean": 6.35, "true_hba1c_sd": 0.25,
                "hba1c_analytical_cv": 0.02, "hba1c_diabetes_threshold": 6.5,
                "n_patients": 2000,
            }
        },
        {
            "label": "Scenario_5_Sensitivity_Analysis_Threshold_Perturbation",
            "param_overrides": {
                "potassium_threshold_min": 5.5, "potassium_threshold_max": 6.5,
                "hba1c_threshold_min": 6.0, "hba1c_threshold_max": 7.0,
                "sbp_threshold_min": 120.0, "sbp_threshold_max": 140.0,
                "n_threshold_steps": 20, "n_patients": 2000,
            }
        },
    ]
}

# FHIR constants
LOINC_K = "2823-3"
LOINC_HBA1C = "4548-4"
LOINC_SBP = "8480-6"
UCUM_K = "mmol/L"
UCUM_HBA1C = "%"
UCUM_SBP = "mm[Hg]"
SNOMED_DIABETES = "73211009"
SNOMED_HYPERTENSION = "38341003"

# ─────────────────────────────────────────────
# HELPER: clamp
# ─────────────────────────────────────────────
def clamp(x, lo, hi):
    return float(max(lo, min(hi, x)))


def clamp_arr(arr, lo, hi):
    return np.clip(arr, lo, hi)


# ─────────────────────────────────────────────
# SCENARIO SIMULATIONS
# ─────────────────────────────────────────────

def run_scenario_1(params: Dict, seed: int) -> List[Dict]:
    np.random.seed(seed)
    n = int(params.get("n_patients", 2000))
    true_k_mean = params.get("true_potassium_mean", 4.2)
    true_k_sd = params.get("true_potassium_sd", 0.4)
    hem_prob = params.get("hemolysis_probability", 0.08)
    hem_bias_mean = params.get("hemolysis_bias_mean", 1.8)
    hem_bias_sd = params.get("hemolysis_bias_sd", 0.5)
    anal_sd = params.get("potassium_analytical_sd", 0.15)
    k_thresh = params.get("potassium_critical_threshold", 6.0)
    drift_rate = params.get("calibration_drift_rate", 0.001)
    n_draws = int(params.get("n_serial_draws", 1))

    # For Scenario 1, one draw per patient (draw_index=0)
    true_k = clamp_arr(np.random.normal(true_k_mean, true_k_sd, n), 3.0, 5.8)
    hemolysis = (np.random.uniform(0, 1, n) < hem_prob).astype(float)
    hem_bias_raw = np.random.normal(hem_bias_mean, hem_bias_sd, n)
    hem_bias = np.where(hemolysis == 1, clamp_arr(hem_bias_raw, 0, 3.5), 0.0)
    anal_noise = np.random.normal(0, anal_sd, n)
    draw_index = 0
    calib_k = clamp(1.0 + drift_rate * draw_index, 0.9, 1.1)
    measured_k = clamp_arr((true_k + hem_bias + anal_noise) * calib_k, 2.5, 9.0)

    label_k_true = (true_k >= k_thresh).astype(int)
    label_k_emr = (measured_k >= k_thresh).astype(int)
    label_err_k = np.abs(label_k_true - label_k_emr)

    # HbA1c defaults (neutral, single draw)
    true_h = clamp_arr(np.random.normal(6.2, 0.3, n), 4.0, 12.0)
    hba1c_phys_noise = np.random.normal(0, 0.15, n)
    hba1c_anal_noise = np.random.normal(0, true_h * 0.02, n)
    calib_h = 1.0
    measured_h = clamp_arr((true_h + hba1c_phys_noise + hba1c_anal_noise) * calib_h, 4.0, 13.0)
    h_thresh = 6.5
    label_h_true = (true_h >= h_thresh).astype(int)
    label_h_emr = (measured_h >= h_thresh).astype(int)
    label_err_h = np.abs(label_h_true - label_h_emr)

    # SBP defaults
    true_sbp = clamp_arr(np.random.normal(128.0, 8.0, n), 80.0, 200.0)
    sbp_phys = np.random.normal(0, 8.0, n)
    sbp_dev = np.random.normal(0, 5.0, n)
    wc_flag = (np.random.uniform(0, 1, n) < 0.2).astype(float)
    wc_effect_raw = np.random.normal(14.0, 6.0, n)
    wc_effect = np.where(wc_flag == 1, clamp_arr(wc_effect_raw, 0, 30), 0.0)
    measured_sbp = clamp_arr(true_sbp + sbp_phys + sbp_dev + wc_effect, 60.0, 240.0)
    sbp_thresh = 130.0
    label_sbp_true = (true_sbp >= sbp_thresh).astype(int)
    label_sbp_emr = (measured_sbp >= sbp_thresh).astype(int)
    label_err_sbp = np.abs(label_sbp_true - label_sbp_emr)

    total_errors = label_err_k + label_err_h + label_err_sbp
    n_pos_obs = label_k_emr + label_h_emr + label_sbp_emr  # across biomarkers
    ai_label_prob = clamp_arr(n_pos_obs / 3.0 + np.random.normal(0, 0.05, n), 0.0, 1.0)

    rows = []
    for i in range(n):
        row = {
            "patient_id": f"S1_P{i:05d}",
            "scenario": "Scenario_1_False_Critical_Potassium",
            "site_id": 0,
            "draw_index": 0,
            "true_potassium": round(float(true_k[i]), 4),
            "hemolysis_flag": int(hemolysis[i]),
            "hemolysis_bias": round(float(hem_bias[i]), 4),
            "analytical_noise_potassium": round(float(anal_noise[i]), 4),
            "measured_potassium": round(float(measured_k[i]), 4),
            "calibration_factor_potassium": round(calib_k, 4),
            "potassium_label_true": int(label_k_true[i]),
            "potassium_label_emr": int(label_k_emr[i]),
            "potassium_label_error": int(label_err_k[i]),
            "true_hba1c": round(float(true_h[i]), 4),
            "hba1c_physiologic_noise": round(float(hba1c_phys_noise[i]), 4),
            "hba1c_analytical_noise": round(float(hba1c_anal_noise[i]), 4),
            "hba1c_calibration_bias": round(0.0, 4),
            "measured_hba1c": round(float(measured_h[i]), 4),
            "calibration_factor_hba1c": round(calib_h, 4),
            "diabetes_label_true": int(label_h_true[i]),
            "diabetes_label_emr": int(label_h_emr[i]),
            "diabetes_label_error": int(label_err_h[i]),
            "true_sbp": round(float(true_sbp[i]), 4),
            "sbp_physiologic_noise": round(float(sbp_phys[i]), 4),
            "sbp_device_noise": round(float(sbp_dev[i]), 4),
            "white_coat_flag": int(wc_flag[i]),
            "white_coat_effect": round(float(wc_effect[i]), 4),
            "measured_sbp": round(float(measured_sbp[i]), 4),
            "hypertension_label_true": int(label_sbp_true[i]),
            "hypertension_label_emr": int(label_sbp_emr[i]),
            "hypertension_label_error": int(label_err_sbp[i]),
            "label_flip_potassium_across_draws": int(label_err_k[i]),
            "label_flip_hba1c_across_draws": int(label_err_h[i]),
            "label_flip_sbp_across_draws": int(label_err_sbp[i]),
            "total_label_errors": int(total_errors[i]),
            "fhir_loinc_potassium": LOINC_K,
            "fhir_loinc_hba1c": LOINC_HBA1C,
            "fhir_loinc_sbp": LOINC_SBP,
            "fhir_ucum_potassium": UCUM_K,
            "fhir_ucum_hba1c": UCUM_HBA1C,
            "fhir_ucum_sbp": UCUM_SBP,
            "fhir_snomed_diabetes": SNOMED_DIABETES if label_h_emr[i] else "",
            "fhir_snomed_hypertension": SNOMED_HYPERTENSION if label_sbp_emr[i] else "",
            "fhir_observation_status": "final",
            "threshold_potassium_used": k_thresh,
            "threshold_hba1c_used": h_thresh,
            "threshold_sbp_used": sbp_thresh,
            # for fig 8
            "n_positive_observations": int(n_pos_obs[i]),
            "final_ai_label_probability": round(float(ai_label_prob[i]), 4),
            "true_positive_status": int((label_k_true[i] + label_h_true[i] + label_sbp_true[i]) > 0),
        }
        rows.append(row)
    return rows


def run_scenario_2(params: Dict, seed: int) -> List[Dict]:
    np.random.seed(seed)
    n = int(params.get("n_patients", 2000))
    n_draws = int(params.get("n_serial_draws", 20))
    true_h_mean = params.get("true_hba1c_mean", 6.35)
    true_h_sd = params.get("true_hba1c_sd", 0.25)
    h_cv = params.get("hba1c_analytical_cv", 0.02)
    h_phys_sd = params.get("hba1c_physiologic_sd", 0.15)
    h_cal_bias = params.get("hba1c_calibration_bias", 0.1)
    h_thresh = params.get("hba1c_diabetes_threshold", 6.5)
    drift_rate = params.get("calibration_drift_rate", 0.001)

    true_h_patients = clamp_arr(np.random.normal(true_h_mean, true_h_sd, n), 4.5, 10.0)
    label_h_true = (true_h_patients >= h_thresh).astype(int)
    wc_flag = (np.random.uniform(0, 1, n) < 0.2).astype(float)
    wc_effect_raw = np.random.normal(14.0, 6.0, n)
    wc_effect_per_patient = np.where(wc_flag == 1, clamp_arr(wc_effect_raw, 0, 30), 0.0)

    rows = []
    label_emr_matrix = np.zeros((n, n_draws), dtype=int)

    for d in range(n_draws):
        phys_noise = np.random.normal(0, h_phys_sd, n)
        anal_noise = np.random.normal(0, true_h_patients * h_cv, n)
        calib_f = clamp(1.0 + drift_rate * d, 0.9, 1.1)
        measured_h = clamp_arr((true_h_patients + phys_noise + anal_noise + h_cal_bias) * calib_f, 4.0, 13.0)
        label_h_emr = (measured_h >= h_thresh).astype(int)
        label_emr_matrix[:, d] = label_h_emr

        true_k = clamp_arr(np.random.normal(4.2, 0.4, n), 3.0, 5.8)
        hem = (np.random.uniform(0, 1, n) < 0.08).astype(float)
        hem_bias = np.where(hem == 1, clamp_arr(np.random.normal(1.8, 0.5, n), 0, 3.5), 0.0)
        anal_k = np.random.normal(0, 0.15, n)
        calib_k = clamp(1.0 + drift_rate * d, 0.9, 1.1)
        meas_k = clamp_arr((true_k + hem_bias + anal_k) * calib_k, 2.5, 9.0)
        lk_true = (true_k >= 6.0).astype(int)
        lk_emr = (meas_k >= 6.0).astype(int)

        true_sbp = clamp_arr(np.random.normal(128.0, 8.0, n), 80.0, 200.0)
        sbp_phys = np.random.normal(0, 8.0, n)
        sbp_dev = np.random.normal(0, 5.0, n)
        meas_sbp = clamp_arr(true_sbp + sbp_phys + sbp_dev + wc_effect_per_patient, 60.0, 240.0)
        ls_true = (true_sbp >= 130.0).astype(int)
        ls_emr = (meas_sbp >= 130.0).astype(int)

        n_pos = lk_emr + label_h_emr + ls_emr
        ai_prob = clamp_arr(n_pos / 3.0 + np.random.normal(0, 0.05, n), 0.0, 1.0)

        for i in range(n):
            rows.append({
                "patient_id": f"S2_P{i:05d}",
                "scenario": "Scenario_2_Borderline_HbA1c_Diabetes_Label",
                "site_id": i % 4,
                "draw_index": d,
                "true_potassium": round(float(true_k[i]), 4),
                "hemolysis_flag": int(hem[i]),
                "hemolysis_bias": round(float(hem_bias[i]), 4),
                "analytical_noise_potassium": round(float(anal_k[i]), 4),
                "measured_potassium": round(float(meas_k[i]), 4),
                "calibration_factor_potassium": round(calib_k, 4),
                "potassium_label_true": int(lk_true[i]),
                "potassium_label_emr": int(lk_emr[i]),
                "potassium_label_error": int(abs(lk_true[i] - lk_emr[i])),
                "true_hba1c": round(float(true_h_patients[i]), 4),
                "hba1c_physiologic_noise": round(float(phys_noise[i]), 4),
                "hba1c_analytical_noise": round(float(anal_noise[i]), 4),
                "hba1c_calibration_bias": round(h_cal_bias, 4),
                "measured_hba1c": round(float(measured_h[i]), 4),
                "calibration_factor_hba1c": round(calib_f, 4),
                "diabetes_label_true": int(label_h_true[i]),
                "diabetes_label_emr": int(label_h_emr[i]),
                "diabetes_label_error": int(abs(label_h_true[i] - label_h_emr[i])),
                "true_sbp": round(float(true_sbp[i]), 4),
                "sbp_physiologic_noise": round(float(sbp_phys[i]), 4),
                "sbp_device_noise": round(float(sbp_dev[i]), 4),
                "white_coat_flag": int(wc_flag[i]),
                "white_coat_effect": round(float(wc_effect_per_patient[i]), 4),
                "measured_sbp": round(float(meas_sbp[i]), 4),
                "hypertension_label_true": int(ls_true[i]),
                "hypertension_label_emr": int(ls_emr[i]),
                "hypertension_label_error": int(abs(ls_true[i] - ls_emr[i])),
                "label_flip_potassium_across_draws": 0,  # computed below
                "label_flip_hba1c_across_draws": 0,
                "label_flip_sbp_across_draws": 0,
                "total_label_errors": int(abs(lk_true[i]-lk_emr[i]) + abs(label_h_true[i]-label_h_emr[i]) + abs(ls_true[i]-ls_emr[i])),
                "fhir_loinc_potassium": LOINC_K,
                "fhir_loinc_hba1c": LOINC_HBA1C,
                "fhir_loinc_sbp": LOINC_SBP,
                "fhir_ucum_potassium": UCUM_K,
                "fhir_ucum_hba1c": UCUM_HBA1C,
                "fhir_ucum_sbp": UCUM_SBP,
                "fhir_snomed_diabetes": SNOMED_DIABETES if label_h_emr[i] else "",
                "fhir_snomed_hypertension": SNOMED_HYPERTENSION if ls_emr[i] else "",
                "fhir_observation_status": "final",
                "threshold_potassium_used": 6.0,
                "threshold_hba1c_used": h_thresh,
                "threshold_sbp_used": 130.0,
                "n_positive_observations": int(n_pos[i]),
                "final_ai_label_probability": round(float(ai_prob[i]), 4),
                "true_positive_status": int((lk_true[i] + label_h_true[i] + ls_true[i]) > 0),
            })

    # Compute label flips across draws and back-fill
    # label_emr_matrix shape = (n, n_draws)
    # For each patient, check if any draw disagrees with the true label
    flip_h = np.any(label_emr_matrix != label_h_true[:, np.newaxis], axis=1).astype(int)
    for idx, row in enumerate(rows):
        pid_idx = int(row["patient_id"].split("P")[1])
        row["label_flip_hba1c_across_draws"] = int(flip_h[pid_idx])
    return rows


def run_scenario_3(params: Dict, seed: int) -> List[Dict]:
    np.random.seed(seed)
    n = int(params.get("n_patients", 2000))
    n_draws = int(params.get("n_serial_draws", 20))
    sbp_mean = params.get("true_sbp_mean", 128.0)
    sbp_sd = params.get("true_sbp_sd", 8.0)
    sbp_dev_sd = params.get("sbp_device_sd", 5.0)
    sbp_phys_sd = params.get("sbp_physiologic_sd", 8.0)
    wc_prob = params.get("white_coat_probability", 0.2)
    wc_mean = params.get("sbp_white_coat_mean", 14.0)
    wc_sd = params.get("sbp_white_coat_sd", 6.0)
    sbp_thresh = params.get("sbp_hypertension_threshold", 130.0)
    drift_rate = params.get("calibration_drift_rate", 0.001)

    true_sbp_patients = clamp_arr(np.random.normal(sbp_mean, sbp_sd, n), 90.0, 200.0)
    label_sbp_true = (true_sbp_patients >= sbp_thresh).astype(int)
    wc_flag = (np.random.uniform(0, 1, n) < wc_prob).astype(float)
    wc_effect_raw = np.random.normal(wc_mean, wc_sd, n)
    wc_effect_per_patient = np.where(wc_flag == 1, clamp_arr(wc_effect_raw, 0, 30), 0.0)

    rows = []
    label_sbp_emr_matrix = np.zeros((n, n_draws), dtype=int)

    for d in range(n_draws):
        phys_noise = np.random.normal(0, sbp_phys_sd, n)
        dev_noise = np.random.normal(0, sbp_dev_sd, n)
        meas_sbp = clamp_arr(true_sbp_patients + phys_noise + dev_noise + wc_effect_per_patient, 60.0, 240.0)
        ls_emr = (meas_sbp >= sbp_thresh).astype(int)
        label_sbp_emr_matrix[:, d] = ls_emr

        true_k = clamp_arr(np.random.normal(4.2, 0.4, n), 3.0, 5.8)
        hem = (np.random.uniform(0, 1, n) < 0.08).astype(float)
        hem_bias = np.where(hem == 1, clamp_arr(np.random.normal(1.8, 0.5, n), 0, 3.5), 0.0)
        anal_k = np.random.normal(0, 0.15, n)
        calib_k = clamp(1.0 + drift_rate * d, 0.9, 1.1)
        meas_k = clamp_arr((true_k + hem_bias + anal_k) * calib_k, 2.5, 9.0)
        lk_true = (true_k >= 6.0).astype(int)
        lk_emr = (meas_k >= 6.0).astype(int)

        true_h = clamp_arr(np.random.normal(6.2, 0.3, n), 4.0, 12.0)
        h_phys = np.random.normal(0, 0.15, n)
        h_anal = np.random.normal(0, true_h * 0.02, n)
        calib_h = clamp(1.0 + drift_rate * d, 0.9, 1.1)
        meas_h = clamp_arr((true_h + h_phys + h_anal) * calib_h, 4.0, 13.0)
        lh_true = (true_h >= 6.5).astype(int)
        lh_emr = (meas_h >= 6.5).astype(int)

        n_pos = lk_emr + lh_emr + ls_emr
        ai_prob = clamp_arr(n_pos / 3.0 + np.random.normal(0, 0.05, n), 0.0, 1.0)

        for i in range(n):
            rows.append({
                "patient_id": f"S3_P{i:05d}",
                "scenario": "Scenario_3_Borderline_SBP_Hypertension_Label",
                "site_id": i % 4,
                "draw_index": d,
                "true_potassium": round(float(true_k[i]), 4),
                "hemolysis_flag": int(hem[i]),
                "hemolysis_bias": round(float(hem_bias[i]), 4),
                "analytical_noise_potassium": round(float(anal_k[i]), 4),
                "measured_potassium": round(float(meas_k[i]), 4),
                "calibration_factor_potassium": round(calib_k, 4),
                "potassium_label_true": int(lk_true[i]),
                "potassium_label_emr": int(lk_emr[i]),
                "potassium_label_error": int(abs(lk_true[i]-lk_emr[i])),
                "true_hba1c": round(float(true_h[i]), 4),
                "hba1c_physiologic_noise": round(float(h_phys[i]), 4),
                "hba1c_analytical_noise": round(float(h_anal[i]), 4),
                "hba1c_calibration_bias": 0.0,
                "measured_hba1c": round(float(meas_h[i]), 4),
                "calibration_factor_hba1c": round(calib_h, 4),
                "diabetes_label_true": int(lh_true[i]),
                "diabetes_label_emr": int(lh_emr[i]),
                "diabetes_label_error": int(abs(lh_true[i]-lh_emr[i])),
                "true_sbp": round(float(true_sbp_patients[i]), 4),
                "sbp_physiologic_noise": round(float(phys_noise[i]), 4),
                "sbp_device_noise": round(float(dev_noise[i]), 4),
                "white_coat_flag": int(wc_flag[i]),
                "white_coat_effect": round(float(wc_effect_per_patient[i]), 4),
                "measured_sbp": round(float(meas_sbp[i]), 4),
                "hypertension_label_true": int(label_sbp_true[i]),
                "hypertension_label_emr": int(ls_emr[i]),
                "hypertension_label_error": int(abs(label_sbp_true[i]-ls_emr[i])),
                "label_flip_potassium_across_draws": 0,
                "label_flip_hba1c_across_draws": 0,
                "label_flip_sbp_across_draws": 0,
                "total_label_errors": int(abs(lk_true[i]-lk_emr[i])+abs(lh_true[i]-lh_emr[i])+abs(label_sbp_true[i]-ls_emr[i])),
                "fhir_loinc_potassium": LOINC_K,
                "fhir_loinc_hba1c": LOINC_HBA1C,
                "fhir_loinc_sbp": LOINC_SBP,
                "fhir_ucum_potassium": UCUM_K,
                "fhir_ucum_hba1c": UCUM_HBA1C,
                "fhir_ucum_sbp": UCUM_SBP,
                "fhir_snomed_diabetes": SNOMED_DIABETES if lh_emr[i] else "",
                "fhir_snomed_hypertension": SNOMED_HYPERTENSION if ls_emr[i] else "",
                "fhir_observation_status": "final",
                "threshold_potassium_used": 6.0,
                "threshold_hba1c_used": 6.5,
                "threshold_sbp_used": sbp_thresh,
                "n_positive_observations": int(n_pos[i]),
                "final_ai_label_probability": round(float(ai_prob[i]), 4),
                "true_positive_status": int((lk_true[i]+lh_true[i]+label_sbp_true[i]) > 0),
            })

    flip_sbp = np.any(label_sbp_emr_matrix != label_sbp_true[:, np.newaxis], axis=1).astype(int)
    for idx, row in enumerate(rows):
        pid_idx = int(row["patient_id"].split("P")[1])
        row["label_flip_sbp_across_draws"] = int(flip_sbp[pid_idx])
    return rows


def run_scenario_4(params: Dict, seed: int) -> List[Dict]:
    np.random.seed(seed)
    n = int(params.get("n_patients", 2000))
    n_sites = int(params.get("n_sites", 4))
    site_bias_sd = params.get("site_calibration_bias_sd", 0.08)
    true_h_mean = params.get("true_hba1c_mean", 6.35)
    true_h_sd = params.get("true_hba1c_sd", 0.25)
    h_cv = params.get("hba1c_analytical_cv", 0.02)
    h_thresh = params.get("hba1c_diabetes_threshold", 6.5)
    drift_rate = params.get("calibration_drift_rate", 0.001)

    site_biases = np.random.normal(0, site_bias_sd, n_sites)
    true_h_all = clamp_arr(np.random.normal(true_h_mean, true_h_sd, n), 4.5, 10.0)
    label_h_true = (true_h_all >= h_thresh).astype(int)
    wc_flag = (np.random.uniform(0, 1, n) < 0.2).astype(float)
    wc_eff = np.where(wc_flag == 1, clamp_arr(np.random.normal(14.0, 6.0, n), 0, 30), 0.0)

    rows = []
    pts_per_site = n // n_sites

    for s in range(n_sites):
        start = s * pts_per_site
        end = start + pts_per_site if s < n_sites - 1 else n
        site_calib = clamp(1.0 + site_biases[s], 0.9, 1.1)
        for i in range(start, end):
            d = 0
            true_h = true_h_all[i]
            h_phys = float(np.random.normal(0, 0.15))
            h_anal = float(np.random.normal(0, true_h * h_cv))
            calib_k_factor = clamp(1.0 + drift_rate * d, 0.9, 1.1)
            meas_h = clamp(
                (true_h + h_phys + h_anal) * site_calib, 4.0, 13.0)
            lh_emr = int(meas_h >= h_thresh)
            lh_true = int(label_h_true[i])

            true_k = clamp(np.random.normal(4.2, 0.4), 3.0, 5.8)
            hem = int(np.random.uniform() < 0.08)
            hem_b = clamp(np.random.normal(1.8, 0.5), 0, 3.5) if hem else 0.0
            a_k = float(np.random.normal(0, 0.15))
            meas_k = clamp((true_k + hem_b + a_k) * calib_k_factor, 2.5, 9.0)
            lk_true = int(true_k >= 6.0)
            lk_emr = int(meas_k >= 6.0)

            true_sbp = clamp(np.random.normal(128.0, 8.0), 80.0, 200.0)
            sbp_phys = float(np.random.normal(0, 8.0))
            sbp_dev = float(np.random.normal(0, 5.0))
            wce = float(wc_eff[i])
            meas_sbp = clamp(true_sbp + sbp_phys + sbp_dev + wce, 60.0, 240.0)
            ls_true = int(true_sbp >= 130.0)
            ls_emr = int(meas_sbp >= 130.0)

            n_pos = lk_emr + lh_emr + ls_emr
            ai_prob = clamp(n_pos / 3.0 + float(np.random.normal(0, 0.05)), 0.0, 1.0)

            rows.append({
                "patient_id": f"S4_P{i:05d}",
                "scenario": "Scenario_4_Calibration_Drift_Cross_Site",
                "site_id": s,
                "draw_index": d,
                "true_potassium": round(true_k, 4),
                "hemolysis_flag": hem,
                "hemolysis_bias": round(hem_b, 4),
                "analytical_noise_potassium": round(a_k, 4),
                "measured_potassium": round(meas_k, 4),
                "calibration_factor_potassium": round(calib_k_factor, 4),
                "potassium_label_true": lk_true,
                "potassium_label_emr": lk_emr,
                "potassium_label_error": int(abs(lk_true-lk_emr)),
                "true_hba1c": round(true_h, 4),
                "hba1c_physiologic_noise": round(h_phys, 4),
                "hba1c_analytical_noise": round(h_anal, 4),
                "hba1c_calibration_bias": round(float(site_biases[s]), 4),
                "measured_hba1c": round(meas_h, 4),
                "calibration_factor_hba1c": round(site_calib, 4),
                "diabetes_label_true": lh_true,
                "diabetes_label_emr": lh_emr,
                "diabetes_label_error": int(abs(lh_true-lh_emr)),
                "true_sbp": round(true_sbp, 4),
                "sbp_physiologic_noise": round(sbp_phys, 4),
                "sbp_device_noise": round(sbp_dev, 4),
                "white_coat_flag": int(wc_flag[i]),
                "white_coat_effect": round(wce, 4),
                "measured_sbp": round(meas_sbp, 4),
                "hypertension_label_true": ls_true,
                "hypertension_label_emr": ls_emr,
                "hypertension_label_error": int(abs(ls_true-ls_emr)),
                "label_flip_potassium_across_draws": int(abs(lk_true-lk_emr)),
                "label_flip_hba1c_across_draws": int(abs(lh_true-lh_emr)),
                "label_flip_sbp_across_draws": int(abs(ls_true-ls_emr)),
                "total_label_errors": int(abs(lk_true-lk_emr)+abs(lh_true-lh_emr)+abs(ls_true-ls_emr)),
                "fhir_loinc_potassium": LOINC_K,
                "fhir_loinc_hba1c": LOINC_HBA1C,
                "fhir_loinc_sbp": LOINC_SBP,
                "fhir_ucum_potassium": UCUM_K,
                "fhir_ucum_hba1c": UCUM_HBA1C,
                "fhir_ucum_sbp": UCUM_SBP,
                "fhir_snomed_diabetes": SNOMED_DIABETES if lh_emr else "",
                "fhir_snomed_hypertension": SNOMED_HYPERTENSION if ls_emr else "",
                "fhir_observation_status": "final",
                "threshold_potassium_used": 6.0,
                "threshold_hba1c_used": h_thresh,
                "threshold_sbp_used": 130.0,
                "n_positive_observations": n_pos,
                "final_ai_label_probability": round(ai_prob, 4),
                "true_positive_status": int((lk_true+lh_true+ls_true) > 0),
            })
    return rows


def run_scenario_5(params: Dict, seed: int) -> List[Dict]:
    np.random.seed(seed)
    n = int(params.get("n_patients", 2000))
    k_min = params.get("potassium_threshold_min", 5.5)
    k_max = params.get("potassium_threshold_max", 6.5)
    h_min = params.get("hba1c_threshold_min", 6.0)
    h_max = params.get("hba1c_threshold_max", 7.0)
    s_min = params.get("sbp_threshold_min", 120.0)
    s_max = params.get("sbp_threshold_max", 140.0)
    n_steps = int(params.get("n_threshold_steps", 20))
    drift_rate = params.get("calibration_drift_rate", 0.001)

    k_thresholds = np.linspace(k_min, k_max, n_steps)
    h_thresholds = np.linspace(h_min, h_max, n_steps)
    s_thresholds = np.linspace(s_min, s_max, n_steps)

    true_k = clamp_arr(np.random.normal(4.2, 0.4, n), 3.0, 5.8)
    hem = (np.random.uniform(0, 1, n) < 0.08).astype(float)
    hem_bias = np.where(hem == 1, clamp_arr(np.random.normal(1.8, 0.5, n), 0, 3.5), 0.0)
    anal_k = np.random.normal(0, 0.15, n)
    meas_k = clamp_arr((true_k + hem_bias + anal_k) * 1.0, 2.5, 9.0)

    true_h = clamp_arr(np.random.normal(6.35, 0.25, n), 4.5, 10.0)
    h_phys = np.random.normal(0, 0.15, n)
    h_anal = np.random.normal(0, true_h * 0.02, n)
    meas_h = clamp_arr((true_h + h_phys + h_anal + 0.1) * 1.0, 4.0, 13.0)

    true_sbp = clamp_arr(np.random.normal(128.0, 8.0, n), 80.0, 200.0)
    wc_flag = (np.random.uniform(0, 1, n) < 0.2).astype(float)
    wc_eff = np.where(wc_flag == 1, clamp_arr(np.random.normal(14.0, 6.0, n), 0, 30), 0.0)
    sbp_phys = np.random.normal(0, 8.0, n)
    sbp_dev = np.random.normal(0, 5.0, n)
    meas_sbp = clamp_arr(true_sbp + sbp_phys + sbp_dev + wc_eff, 60.0, 240.0)

    rows = []
    for step_idx in range(n_steps):
        k_thresh = float(k_thresholds[step_idx])
        h_thresh = float(h_thresholds[step_idx])
        s_thresh = float(s_thresholds[step_idx])

        lk_true = (true_k >= k_thresh).astype(int)
        lk_emr = (meas_k >= k_thresh).astype(int)
        lh_true = (true_h >= h_thresh).astype(int)
        lh_emr = (meas_h >= h_thresh).astype(int)
        ls_true = (true_sbp >= s_thresh).astype(int)
        ls_emr = (meas_sbp >= s_thresh).astype(int)

        for i in range(n):
            n_pos = int(lk_emr[i] + lh_emr[i] + ls_emr[i])
            ai_prob = clamp(n_pos / 3.0 + float(np.random.normal(0, 0.05)), 0.0, 1.0)
            calib_k = clamp(1.0 + drift_rate * step_idx, 0.9, 1.1)
            rows.append({
                "patient_id": f"S5_P{i:05d}",
                "scenario": "Scenario_5_Sensitivity_Analysis_Threshold_Perturbation",
                "site_id": i % 4,
                "draw_index": step_idx,
                "true_potassium": round(float(true_k[i]), 4),
                "hemolysis_flag": int(hem[i]),
                "hemolysis_bias": round(float(hem_bias[i]), 4),
                "analytical_noise_potassium": round(float(anal_k[i]), 4),
                "measured_potassium": round(float(meas_k[i]), 4),
                "calibration_factor_potassium": round(calib_k, 4),
                "potassium_label_true": int(lk_true[i]),
                "potassium_label_emr": int(lk_emr[i]),
                "potassium_label_error": int(abs(lk_true[i]-lk_emr[i])),
                "true_hba1c": round(float(true_h[i]), 4),
                "hba1c_physiologic_noise": round(float(h_phys[i]), 4),
                "hba1c_analytical_noise": round(float(h_anal[i]), 4),
                "hba1c_calibration_bias": 0.1,
                "measured_hba1c": round(float(meas_h[i]), 4),
                "calibration_factor_hba1c": 1.0,
                "diabetes_label_true": int(lh_true[i]),
                "diabetes_label_emr": int(lh_emr[i]),
                "diabetes_label_error": int(abs(lh_true[i]-lh_emr[i])),
                "true_sbp": round(float(true_sbp[i]), 4),
                "sbp_physiologic_noise": round(float(sbp_phys[i]), 4),
                "sbp_device_noise": round(float(sbp_dev[i]), 4),
                "white_coat_flag": int(wc_flag[i]),
                "white_coat_effect": round(float(wc_eff[i]), 4),
                "measured_sbp": round(float(meas_sbp[i]), 4),
                "hypertension_label_true": int(ls_true[i]),
                "hypertension_label_emr": int(ls_emr[i]),
                "hypertension_label_error": int(abs(ls_true[i]-ls_emr[i])),
                "label_flip_potassium_across_draws": int(abs(lk_true[i]-lk_emr[i])),
                "label_flip_hba1c_across_draws": int(abs(lh_true[i]-lh_emr[i])),
                "label_flip_sbp_across_draws": int(abs(ls_true[i]-ls_emr[i])),
                "total_label_errors": int(abs(lk_true[i]-lk_emr[i])+abs(lh_true[i]-lh_emr[i])+abs(ls_true[i]-ls_emr[i])),
                "fhir_loinc_potassium": LOINC_K,
                "fhir_loinc_hba1c": LOINC_HBA1C,
                "fhir_loinc_sbp": LOINC_SBP,
                "fhir_ucum_potassium": UCUM_K,
                "fhir_ucum_hba1c": UCUM_HBA1C,
                "fhir_ucum_sbp": UCUM_SBP,
                "fhir_snomed_diabetes": SNOMED_DIABETES if lh_emr[i] else "",
                "fhir_snomed_hypertension": SNOMED_HYPERTENSION if ls_emr[i] else "",
                "fhir_observation_status": "final",
                "threshold_potassium_used": round(k_thresh, 4),
                "threshold_hba1c_used": round(h_thresh, 4),
                "threshold_sbp_used": round(s_thresh, 4),
                "n_positive_observations": n_pos,
                "final_ai_label_probability": round(ai_prob, 4),
                "true_positive_status": int((lk_true[i]+lh_true[i]+ls_true[i]) > 0),
            })
    return rows


# ─────────────────────────────────────────────
# CSV WRITERS
# ─────────────────────────────────────────────

SCHEMA_COLS = [
    "patient_id", "scenario", "site_id", "draw_index",
    "true_potassium", "hemolysis_flag", "hemolysis_bias",
    "analytical_noise_potassium", "measured_potassium", "calibration_factor_potassium",
    "potassium_label_true", "potassium_label_emr", "potassium_label_error",
    "true_hba1c", "hba1c_physiologic_noise", "hba1c_analytical_noise",
    "hba1c_calibration_bias", "measured_hba1c", "calibration_factor_hba1c",
    "diabetes_label_true", "diabetes_label_emr", "diabetes_label_error",
    "true_sbp", "sbp_physiologic_noise", "sbp_device_noise",
    "white_coat_flag", "white_coat_effect", "measured_sbp",
    "hypertension_label_true", "hypertension_label_emr", "hypertension_label_error",
    "label_flip_potassium_across_draws", "label_flip_hba1c_across_draws",
    "label_flip_sbp_across_draws", "total_label_errors",
    "fhir_loinc_potassium", "fhir_loinc_hba1c", "fhir_loinc_sbp",
    "fhir_ucum_potassium", "fhir_ucum_hba1c", "fhir_ucum_sbp",
    "fhir_snomed_diabetes", "fhir_snomed_hypertension",
    "fhir_observation_status",
    "threshold_potassium_used", "threshold_hba1c_used", "threshold_sbp_used",
]


def write_simulation_outputs(all_rows: List[Dict], out_dir: Path) -> None:
    if not all_rows:
        print("WARNING: No rows to write for simulation_outputs.csv")
        return
    out_path = out_dir / "simulation_outputs.csv"
    try:
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=SCHEMA_COLS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"  Written: {out_path} ({len(all_rows)} rows)")
    except Exception as e:
        print(f"WARNING: Could not write simulation_outputs.csv: {e}")


def write_scenario_summary(all_rows: List[Dict], out_dir: Path) -> None:
    from collections import defaultdict
    scenario_rows: Dict[str, List[Dict]] = defaultdict(list)
    for row in all_rows:
        scenario_rows[row["scenario"]].append(row)

    numeric_cols = [
        "true_potassium", "measured_potassium", "hemolysis_bias",
        "true_hba1c", "measured_hba1c", "true_sbp", "measured_sbp",
        "potassium_label_error", "diabetes_label_error", "hypertension_label_error",
        "total_label_errors", "label_flip_hba1c_across_draws", "label_flip_sbp_across_draws",
        "white_coat_effect", "hba1c_calibration_bias", "calibration_factor_hba1c",
    ]

    summary_rows = []
    for scenario, rows in scenario_rows.items():
        srow = {"scenario": scenario, "n_rows": len(rows)}
        for col in numeric_cols:
            vals = []
            for r in rows:
                v = r.get(col, None)
                if v is not None:
                    try:
                        vals.append(float(v))
                    except (ValueError, TypeError):
                        pass
            if vals:
                arr = np.array(vals)
                srow[f"{col}_mean"] = round(float(np.mean(arr)), 6)
                srow[f"{col}_std"] = round(float(np.std(arr)), 6)
                srow[f"{col}_min"] = round(float(np.min(arr)), 6)
                srow[f"{col}_max"] = round(float(np.max(arr)), 6)
            else:
                srow[f"{col}_mean"] = 0.0
                srow[f"{col}_std"] = 0.0
                srow[f"{col}_min"] = 0.0
                srow[f"{col}_max"] = 0.0
        # KPIs
        err_vals = [float(r.get("potassium_label_error", 0)) for r in rows]
        srow["potassium_label_error_rate"] = round(float(np.mean(err_vals)), 6)
        h_err = [float(r.get("diabetes_label_error", 0)) for r in rows]
        srow["diabetes_label_error_rate"] = round(float(np.mean(h_err)), 6)
        s_err = [float(r.get("hypertension_label_error", 0)) for r in rows]
        srow["hypertension_label_error_rate"] = round(float(np.mean(s_err)), 6)
        hem_rates = [float(r.get("hemolysis_flag", 0)) for r in rows]
        srow["hemolysis_rate"] = round(float(np.mean(hem_rates)), 6)
        summary_rows.append(srow)

    if not summary_rows:
        print("WARNING: No summary rows to write.")
        return

    all_keys = list(summary_rows[0].keys())
    out_path = out_dir / "scenario_summary.csv"
    try:
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"  Written: {out_path} ({len(summary_rows)} rows)")
    except Exception as e:
        print(f"WARNING: Could not write scenario_summary.csv: {e}")


def write_parameters_used(out_dir: Path) -> None:
    rows = []
    for name, info in MODEL_PROFILE["parameters"].items():
        rows.append({
            "name": name,
            "value": info["value"],
            "unit": info["unit"],
            "source": info["source"],
            "description": info["description"],
        })
    if not rows:
        print("WARNING: No parameter rows to write.")
        return
    out_path = out_dir / "parameters_used.csv"
    try:
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["name", "value", "unit", "source", "description"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Written: {out_path} ({len(rows)} rows)")
    except Exception as e:
        print(f"WARNING: Could not write parameters_used.csv: {e}")


# ─────────────────────────────────────────────
# FIGURES
# ─────────────────────────────────────────────

def fig1_hemolysis_potassium(s1_rows: List[Dict], out_dir: Path) -> None:
    try:
        true_k_normal, meas_k_normal = [], []
        true_k_hem, meas_k_hem = [], []
        for r in s1_rows:
            tk = r["true_potassium"]
            mk = r["measured_potassium"]
            if r["hemolysis_flag"] == 1:
                true_k_hem.append(tk)
                meas_k_hem.append(mk)
            else:
                true_k_normal.append(tk)
                meas_k_normal.append(mk)

        x_norm = np.array(true_k_normal)
        y_norm = np.array(meas_k_normal)
        x_hem = np.array(true_k_hem)
        y_hem = np.array(meas_k_hem)

        assert len(x_norm) == len(y_norm), "Mismatch in normal arrays"
        assert len(x_hem) == len(y_hem), "Mismatch in hemolyzed arrays"

        fig, ax = plt.subplots(figsize=(8, 6))
        if len(x_norm) > 0:
            ax.scatter(x_norm, y_norm, alpha=0.3, s=10, color="steelblue", label=f"Non-hemolyzed (n={len(x_norm)})")
        if len(x_hem) > 0:
            ax.scatter(x_hem, y_hem, alpha=0.5, s=15, color="crimson", label=f"Hemolyzed (n={len(x_hem)})")

        xlim = (2.5, 6.5)
        ylim = (2.0, 9.5)
        ax.plot([xlim[0], xlim[1]], [xlim[0], xlim[1]], "k--", lw=1, label="Identity line")
        ax.axhline(6.0, color="darkorange", lw=2, linestyle="--", label="Critical threshold 6.0 mmol/L")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("True Potassium (mmol/L)")
        ax.set_ylabel("Measured Potassium (mmol/L)")
        ax.set_title("Scenario 1: Pre-Analytical Hemolysis Bias on Potassium — True vs. Measured")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "fig_1_hemolysis_potassium.png", dpi=150)
        plt.close(fig)
        print("  Saved: fig_1_hemolysis_potassium.png")
    except Exception as e:
        print(f"WARNING: fig1 failed: {e}")


def fig2_hba1c_serial_label_flips(s2_rows: List[Dict], out_dir: Path) -> None:
    try:
        from collections import defaultdict
        patient_draws: Dict[str, List] = defaultdict(list)
        for r in s2_rows:
            patient_draws[r["patient_id"]].append((r["draw_index"], r["measured_hba1c"], r["diabetes_label_emr"]))

        h_thresh = 6.5
        # Select borderline patients
        borderline_pids = []
        for pid, draws in patient_draws.items():
            vals = [d[1] for d in draws]
            if any(abs(v - h_thresh) < 0.5 for v in vals):
                borderline_pids.append(pid)

        np.random.seed(42)
        n_plot = min(60, len(borderline_pids))
        if n_plot == 0:
            print("WARNING: No borderline patients for fig2")
            return
        selected = np.random.choice(borderline_pids, n_plot, replace=False)

        cmap = plt.cm.YlOrRd
        fig, ax = plt.subplots(figsize=(10, 6))
        for pid in selected:
            draws = sorted(patient_draws[pid], key=lambda x: x[0])
            x_arr = np.array([d[0] for d in draws])
            y_arr = np.array([d[1] for d in draws])
            lbl_arr = np.array([d[2] for d in draws])
            assert len(x_arr) == len(y_arr)
            frac_pos = float(np.mean(lbl_arr))
            color = cmap(frac_pos)
            ax.plot(x_arr, y_arr, color=color, alpha=0.5, lw=0.8)

        ax.axhline(h_thresh, color="red", lw=2, linestyle="--", label=f"Diabetes threshold {h_thresh}%")
        ax.set_xlabel("Draw Index")
        ax.set_ylabel("Measured HbA1c (%)")
        ax.set_title("Scenario 2: Serial HbA1c Label Instability Near 6.5% Diabetes Threshold")
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(0, 1))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="Fraction of draws labeled diabetic")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "fig_2_hba1c_serial_label_flips.png", dpi=150)
        plt.close(fig)
        print("  Saved: fig_2_hba1c_serial_label_flips.png")
    except Exception as e:
        print(f"WARNING: fig2 failed: {e}")


def fig3_sbp_white_coat(s3_rows: List[Dict], out_dir: Path) -> None:
    try:
        meas_sbp_wc, meas_sbp_no = [], []
        for r in s3_rows:
            msb = r["measured_sbp"]
            if r["white_coat_flag"] == 1:
                meas_sbp_wc.append(msb)
            else:
                meas_sbp_no.append(msb)

        arr_wc = np.array(meas_sbp_wc)
        arr_no = np.array(meas_sbp_no)

        fig, ax = plt.subplots(figsize=(9, 6))
        bins = np.linspace(60, 240, 60)
        if len(arr_no) > 0:
            ax.hist(arr_no, bins=bins, alpha=0.5, color="steelblue", label=f"Non-white-coat (n={len(arr_no)})", density=True)
        if len(arr_wc) > 0:
            ax.hist(arr_wc, bins=bins, alpha=0.5, color="darkorange", label=f"White-coat (n={len(arr_wc)})", density=True)
        ax.axvline(130.0, color="red", lw=2, linestyle="--", label="Hypertension threshold 130 mmHg")
        ax.set_xlabel("Measured Systolic BP (mmHg)")
        ax.set_ylabel("Density")
        ax.set_title("Scenario 3: Systolic BP Distribution — White-Coat vs. Non-White-Coat Patients")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "fig_3_sbp_white_coat.png", dpi=150)
        plt.close(fig)
        print("  Saved: fig_3_sbp_white_coat.png")
    except Exception as e:
        print(f"WARNING: fig3 failed: {e}")


def fig4_label_flip_rate_vs_true_value(s2_rows: List[Dict], s3_rows: List[Dict], out_dir: Path) -> None:
    """Label flip rate vs true biomarker value for HbA1c and SBP (and potassium from S1 derived)."""
    try:
        # For HbA1c: group patients by true_hba1c, compute label error rate
        def compute_flip_rate(rows, true_col, emr_col, true_label_col, n_bins=30, val_range=None):
            bins = np.linspace(val_range[0], val_range[1], n_bins + 1)
            centers = 0.5 * (bins[:-1] + bins[1:])
            flip_rates = np.zeros(n_bins)
            counts = np.zeros(n_bins)
            for r in rows:
                tv = r.get(true_col, None)
                le = r.get(emr_col, None)
                lt = r.get(true_label_col, None)
                if tv is None or le is None or lt is None:
                    continue
                idx = np.searchsorted(bins[1:], tv)
                idx = min(idx, n_bins - 1)
                counts[idx] += 1
                if int(le) != int(lt):
                    flip_rates[idx] += 1
            mask = counts > 0
            flip_rates[mask] /= counts[mask]
            return centers, flip_rates, mask

        h_centers, h_flips, h_mask = compute_flip_rate(
            s2_rows, "true_hba1c", "diabetes_label_emr", "diabetes_label_true",
            n_bins=30, val_range=(4.5, 10.0))
        sbp_centers, sbp_flips, sbp_mask = compute_flip_rate(
            s3_rows, "true_sbp", "hypertension_label_emr", "hypertension_label_true",
            n_bins=30, val_range=(80, 200))

        # For potassium: compute from s2_rows
        k_centers, k_flips, k_mask = compute_flip_rate(
            s2_rows, "true_potassium", "potassium_label_emr", "potassium_label_true",
            n_bins=30, val_range=(3.0, 6.5))

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        if np.any(h_mask):
            x_h = h_centers[h_mask]
            y_h = h_flips[h_mask]
            assert len(x_h) == len(y_h)
            axes[0].plot(x_h, y_h, color="darkorange", lw=2, marker="o", ms=4)
        axes[0].axvline(6.5, color="red", linestyle="--", label="DM threshold 6.5%")
        axes[0].set_xlabel("True HbA1c (%)")
        axes[0].set_ylabel("Label Flip Rate")
        axes[0].set_title("HbA1c: Label Flip Rate vs True Value")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0, 1)

        if np.any(sbp_mask):
            x_sbp = sbp_centers[sbp_mask]
            y_sbp = sbp_flips[sbp_mask]
            assert len(x_sbp) == len(y_sbp)
            axes[1].plot(x_sbp, y_sbp, color="steelblue", lw=2, marker="o", ms=4)
        axes[1].axvline(130.0, color="red", linestyle="--", label="HTN threshold 130 mmHg")
        axes[1].set_xlabel("True SBP (mmHg)")
        axes[1].set_ylabel("Label Flip Rate")
        axes[1].set_title("SBP: Label Flip Rate vs True Value")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 1)

        if np.any(k_mask):
            x_k = k_centers[k_mask]
            y_k = k_flips[k_mask]
            assert len(x_k) == len(y_k)
            axes[2].plot(x_k, y_k, color="green", lw=2, marker="o", ms=4)
        axes[2].axvline(6.0, color="red", linestyle="--", label="K+ threshold 6.0 mmol/L")
        axes[2].set_xlabel("True Potassium (mmol/L)")
        axes[2].set_ylabel("Label Flip Rate")
        axes[2].set_title("Potassium: Label Flip Rate vs True Value")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim(0, 1)

        fig.suptitle("Label Instability Rate vs. True Biomarker Value", fontsize=13)
        fig.tight_layout()
        fig.savefig(out_dir / "fig_4_label_flip_rate_vs_true_value.png", dpi=150)
        plt.close(fig)
        print("  Saved: fig_4_label_flip_rate_vs_true_value.png")
    except Exception as e:
        print(f"WARNING: fig4 failed: {e}")


def fig5_cross_site_prevalence(s4_rows: List[Dict], out_dir: Path) -> None:
    try:
        from collections import defaultdict
        site_data: Dict[int, Dict] = defaultdict(lambda: {"n": 0, "pos": 0, "calib": []})
        for r in s4_rows:
            sid = int(r["site_id"])
            site_data[sid]["n"] += 1
            site_data[sid]["pos"] += int(r["diabetes_label_emr"])
            site_data[sid]["calib"].append(float(r["calibration_factor_hba1c"]))

        n_sites = len(site_data)
        if n_sites == 0:
            print("WARNING: No site data for fig5")
            return

        site_ids = sorted(site_data.keys())
        prevalences = []
        biases = []
        for sid in site_ids:
            n = site_data[sid]["n"]
            pos = site_data[sid]["pos"]
            prev = pos / n if n > 0 else 0.0
            avg_calib = float(np.mean(site_data[sid]["calib"])) if site_data[sid]["calib"] else 1.0
            prevalences.append(prev)
            biases.append(avg_calib - 1.0)

        # Make a simple bar chart with colored bars showing prevalence per site
        # Also show a small heatmap-style matrix
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Bar plot
        colors = plt.cm.RdYlGn_r(np.array(prevalences))
        bars = axes[0].bar([f"Site {s}" for s in site_ids], prevalences, color=colors)
        axes[0].axhline(np.mean(prevalences), color="black", linestyle="--", lw=1.5, label=f"Mean prevalence={np.mean(prevalences):.3f}")
        axes[0].set_ylabel("Apparent Diabetes Prevalence")
        axes[0].set_title("Cross-Site Diabetes Prevalence by Site")
        axes[0].legend()
        axes[0].set_ylim(0, 1)
        for bar, prev in zip(bars, prevalences):
            axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                         f"{prev:.3f}", ha="center", fontsize=10)

        # Heatmap: sites x calibration bias buckets
        # Create a 4×1 heatmap matrix
        mat = np.array(prevalences).reshape(n_sites, 1)
        im = axes[1].imshow(mat, aspect="auto", cmap="RdYlGn_r", vmin=0.0, vmax=1.0)
        axes[1].set_yticks(range(n_sites))
        axes[1].set_yticklabels([f"Site {s}\nbias={biases[i]:.3f}" for i, s in enumerate(site_ids)])
        axes[1].set_xticks([0])
        axes[1].set_xticklabels(["Diabetes\nPrevalence"])
        axes[1].set_title("Heatmap: Prevalence by Site & Calibration Bias")
        for row_i in range(n_sites):
            axes[1].text(0, row_i, f"{mat[row_i, 0]:.3f}", ha="center", va="center",
                         color="white" if mat[row_i, 0] > 0.5 else "black", fontsize=12, fontweight="bold")
        plt.colorbar(im, ax=axes[1], label="Apparent Prevalence")

        fig.suptitle("Scenario 4: Cross-Site Calibration Bias — Apparent Diabetes Prevalence by Site", fontsize=12)
        fig.tight_layout()
        fig.savefig(out_dir / "fig_5_cross_site_prevalence.png", dpi=150)
        plt.close(fig)
        print("  Saved: fig_5_cross_site_prevalence.png")
    except Exception as e:
        print(f"WARNING: fig5 failed: {e}")


def fig6_threshold_sensitivity_sweep(s5_rows: List[Dict], out_dir: Path) -> None:
    try:
        from collections import defaultdict

        # Aggregate by draw_index (= threshold step index)
        step_k_errors: Dict[int, List] = defaultdict(list)
        step_h_errors: Dict[int, List] = defaultdict(list)
        step_s_errors: Dict[int, List] = defaultdict(list)
        step_k_thresh: Dict[int, float] = {}
        step_h_thresh: Dict[int, float] = {}
        step_s_thresh: Dict[int, float] = {}

        for r in s5_rows:
            d = int(r["draw_index"])
            step_k_errors[d].append(int(r["potassium_label_error"]))
            step_h_errors[d].append(int(r["diabetes_label_error"]))
            step_s_errors[d].append(int(r["hypertension_label_error"]))
            step_k_thresh[d] = float(r["threshold_potassium_used"])
            step_h_thresh[d] = float(r["threshold_hba1c_used"])
            step_s_thresh[d] = float(r["threshold_sbp_used"])

        steps = sorted(step_k_errors.keys())
        if len(steps) == 0:
            print("WARNING: No steps for fig6")
            return

        x_k = np.array([step_k_thresh[s] for s in steps])
        y_k = np.array([np.mean(step_k_errors[s]) for s in steps])
        x_h = np.array([step_h_thresh[s] for s in steps])
        y_h = np.array([np.mean(step_h_errors[s]) for s in steps])
        x_s = np.array([step_s_thresh[s] for s in steps])
        y_s = np.array([np.mean(step_s_errors[s]) for s in steps])

        assert len(x_k) == len(y_k)
        assert len(x_h) == len(y_h)
        assert len(x_s) == len(y_s)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].plot(x_k, y_k, "go-", lw=2, ms=5, label="Potassium")
        axes[0].axvline(6.0, color="red", linestyle="--", label="Guideline threshold 6.0")
        axes[0].set_xlabel("Potassium Threshold (mmol/L)")
        axes[0].set_ylabel("Label Noise Rate")
        axes[0].set_title("Potassium: Label Noise Rate vs Threshold")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0, 1)

        axes[1].plot(x_h, y_h, "bo-", lw=2, ms=5, label="HbA1c")
        axes[1].axvline(6.5, color="red", linestyle="--", label="Guideline threshold 6.5%")
        axes[1].set_xlabel("HbA1c Threshold (%)")
        axes[1].set_ylabel("Label Noise Rate")
        axes[1].set_title("HbA1c: Label Noise Rate vs Threshold")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 1)

        axes[2].plot(x_s, y_s, "rs-", lw=2, ms=5, label="SBP")
        axes[2].axvline(130.0, color="darkred", linestyle="--", label="Guideline threshold 130 mmHg")
        axes[2].set_xlabel("SBP Threshold (mmHg)")
        axes[2].set_ylabel("Label Noise Rate")
        axes[2].set_title("SBP: Label Noise Rate vs Threshold")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim(0, 1)

        fig.suptitle("Scenario 5: Label Noise Rate vs. Diagnostic Threshold", fontsize=13)
        fig.tight_layout()
        fig.savefig(out_dir / "fig_6_threshold_sensitivity_sweep.png", dpi=150)
        plt.close(fig)
        print("  Saved: fig_6_threshold_sensitivity_sweep.png")
    except Exception as e:
        print(f"WARNING: fig6 failed: {e}")


def fig7_pipeline_uncertainty_propagation(out_dir: Path) -> None:
    """Variance decomposition heatmap: 3 biomarkers x 3 uncertainty sources."""
    try:
        np.random.seed(999)
        n = 5000

        # Potassium: compute label error under each isolated noise source
        true_k = clamp_arr(np.random.normal(4.2, 0.4, n), 3.0, 5.8)
        k_thresh = 6.0
        true_k_label = (true_k >= k_thresh).astype(int)

        # Pre-analytical only (hemolysis)
        hem = (np.random.uniform(0, 1, n) < 0.08).astype(float)
        hem_bias = np.where(hem == 1, clamp_arr(np.random.normal(1.8, 0.5, n), 0, 3.5), 0.0)
        meas_k_preanalytical = clamp_arr(true_k + hem_bias, 2.5, 9.0)
        err_k_pre = float(np.mean(np.abs((meas_k_preanalytical >= k_thresh).astype(int) - true_k_label)))

        # Device/calibration only (analytical noise + drift)
        anal_k = np.random.normal(0, 0.15, n)
        drift = clamp(1.0 + 0.001 * 10, 0.9, 1.1)
        meas_k_device = clamp_arr((true_k + anal_k) * drift, 2.5, 9.0)
        err_k_dev = float(np.mean(np.abs((meas_k_device >= k_thresh).astype(int) - true_k_label)))

        # Context-dependent (none meaningful for potassium - small value)
        err_k_ctx = float(np.mean(np.abs((clamp_arr(true_k + np.random.normal(0, 0.05, n), 2.5, 9.0) >= k_thresh).astype(int) - true_k_label)))

        # HbA1c
        true_h = clamp_arr(np.random.normal(6.35, 0.25, n), 4.5, 10.0)
        h_thresh = 6.5
        true_h_label = (true_h >= h_thresh).astype(int)

        # Pre-analytical (biological variation - small)
        err_h_pre = float(np.mean(np.abs(
            (clamp_arr(true_h + np.random.normal(0, 0.05, n), 4.0, 13.0) >= h_thresh).astype(int) - true_h_label)))

        # Device/calibration (analytical CV + calibration bias)
        h_anal = np.random.normal(0, true_h * 0.02, n)
        site_bias = np.random.normal(0, 0.08, n)
        meas_h_dev = clamp_arr((true_h + h_anal) * (1 + site_bias), 4.0, 13.0)
        err_h_dev = float(np.mean(np.abs((meas_h_dev >= h_thresh).astype(int) - true_h_label)))

        # Context-dependent (none meaningful for HbA1c)
        err_h_ctx = float(np.mean(np.abs(
            (clamp_arr(true_h + np.random.normal(0, 0.07, n) + 0.1, 4.0, 13.0) >= h_thresh).astype(int) - true_h_label)))

        # SBP
        true_sbp = clamp_arr(np.random.normal(128.0, 8.0, n), 80.0, 200.0)
        sbp_thresh = 130.0
        true_sbp_label = (true_sbp >= sbp_thresh).astype(int)

        # Pre-analytical (small, e.g. physiologic)
        err_sbp_pre = float(np.mean(np.abs(
            (clamp_arr(true_sbp + np.random.normal(0, 2.0, n), 60.0, 240.0) >= sbp_thresh).astype(int) - true_sbp_label)))

        # Device/calibration (device noise)
        dev_noise_sbp = np.random.normal(0, 5.0, n)
        err_sbp_dev = float(np.mean(np.abs(
            (clamp_arr(true_sbp + dev_noise_sbp, 60.0, 240.0) >= sbp_thresh).astype(int) - true_sbp_label)))

        # Context-dependent (white-coat effect)
        wc_flag = (np.random.uniform(0, 1, n) < 0.2).astype(float)
        wc_eff = np.where(wc_flag == 1, clamp_arr(np.random.normal(14.0, 6.0, n), 0, 30), 0.0)
        err_sbp_ctx = float(np.mean(np.abs(
            (clamp_arr(true_sbp + wc_eff, 60.0, 240.0) >= sbp_thresh).astype(int) - true_sbp_label)))

        # Build contribution matrix
        raw_matrix = np.array([
            [err_k_pre, err_k_dev, err_k_ctx],
            [err_h_pre, err_h_dev, err_h_ctx],
            [err_sbp_pre, err_sbp_dev, err_sbp_ctx],
        ])
        # Normalize row-wise to get proportions
        row_sums = raw_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        contrib_matrix = raw_matrix / row_sums

        biomarkers = ["Potassium", "HbA1c", "SBP"]
        sources = ["Pre-analytical\nerror", "Device/\ncalibration noise", "Context-dependent\nbias"]

        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(contrib_matrix, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(3))
        ax.set_xticklabels(sources, fontsize=10)
        ax.set_yticks(range(3))
        ax.set_yticklabels(biomarkers, fontsize=11)
        ax.set_title("Quantitative Uncertainty Contribution Matrix:\nSource vs. Biomarker vs. Label Error Rate")
        plt.colorbar(im, ax=ax, label="Proportional Contribution to Label Error")
        for bi in range(3):
            for si in range(3):
                ax.text(si, bi, f"{contrib_matrix[bi, si]:.2f}", ha="center", va="center",
                        color="black" if contrib_matrix[bi, si] < 0.6 else "white", fontsize=12, fontweight="bold")
        fig.tight_layout()
        fig.savefig(out_dir / "fig_7_pipeline_uncertainty_propagation.png", dpi=150)
        plt.close(fig)
        print("  Saved: fig_7_pipeline_uncertainty_propagation.png")
    except Exception as e:
        print(f"WARNING: fig7 failed: {e}")


def fig8_fhir_observation_label_concordance(all_rows: List[Dict], out_dir: Path) -> None:
    try:
        # Aggregate per patient: count positive observations, final ai prob, true status
        from collections import defaultdict
        patient_data: Dict[str, Dict] = defaultdict(lambda: {"n_pos": [], "ai_prob": [], "true_pos": 0})
        for r in all_rows:
            pid = r["patient_id"]
            patient_data[pid]["n_pos"].append(int(r.get("n_positive_observations", 0)))
            patient_data[pid]["ai_prob"].append(float(r.get("final_ai_label_probability", 0.0)))
            patient_data[pid]["true_pos"] = int(r.get("true_positive_status", 0))

        x_arr = np.zeros(len(patient_data))
        y_arr = np.zeros(len(patient_data))
        c_arr = np.zeros(len(patient_data))

        for idx, (pid, data) in enumerate(patient_data.items()):
            x_arr[idx] = float(np.mean(data["n_pos"])) if data["n_pos"] else 0.0
            y_arr[idx] = float(np.mean(data["ai_prob"])) if data["ai_prob"] else 0.0
            c_arr[idx] = float(data["true_pos"])

        assert len(x_arr) == len(y_arr) == len(c_arr)

        if len(x_arr) == 0:
            print("WARNING: No data for fig8")
            return

        # Subsample for legibility
        np.random.seed(42)
        n_plot = min(3000, len(x_arr))
        idx_sample = np.random.choice(len(x_arr), n_plot, replace=False)
        x_plot = x_arr[idx_sample]
        y_plot = y_arr[idx_sample]
        c_plot = c_arr[idx_sample]

        assert len(x_plot) == len(y_plot) == len(c_plot)

        fig, ax = plt.subplots(figsize=(9, 6))
        scatter = ax.scatter(x_plot, y_plot, c=c_plot, cmap="coolwarm_r",
                             alpha=0.4, s=15, edgecolors="none", vmin=0, vmax=1)
        plt.colorbar(scatter, ax=ax, label="True Physiological Status (1=positive)")
        ax.set_xlabel("Mean N Positive FHIR Observations Across Draws")
        ax.set_ylabel("AI Label Probability (Mean Across Draws)")
        ax.set_title("FHIR Observation Concordance vs. AI Label Confidence (All Scenarios)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "fig_8_fhir_observation_label_concordance.png", dpi=150)
        plt.close(fig)
        print("  Saved: fig_8_fhir_observation_label_concordance.png")
    except Exception as e:
        print(f"WARNING: fig8 failed: {e}")


# ─────────────────────────────────────────────
# SUMMARY JSON
# ─────────────────────────────────────────────

def write_summary_json(all_rows: List[Dict], out_dir: Path) -> None:
    from collections import defaultdict
    scenario_rows: Dict[str, List[Dict]] = defaultdict(list)
    for row in all_rows:
        scenario_rows[row["scenario"]].append(row)

    summary = {"scenarios": {}, "total_rows": len(all_rows)}
    for scenario, rows in scenario_rows.items():
        k_err = [float(r.get("potassium_label_error", 0)) for r in rows]
        h_err = [float(r.get("diabetes_label_error", 0)) for r in rows]
        s_err = [float(r.get("hypertension_label_error", 0)) for r in rows]
        summary["scenarios"][scenario] = {
            "n_rows": len(rows),
            "potassium_label_error_rate": round(float(np.mean(k_err)), 6),
            "diabetes_label_error_rate": round(float(np.mean(h_err)), 6),
            "hypertension_label_error_rate": round(float(np.mean(s_err)), 6),
            "mean_total_label_errors": round(float(np.mean([float(r.get("total_label_errors", 0)) for r in rows])), 6),
            "hemolysis_rate": round(float(np.mean([float(r.get("hemolysis_flag", 0)) for r in rows])), 6),
        }
    out_path = out_dir / "summary.json"
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"  Written: {out_path}")
    except Exception as e:
        print(f"WARNING: Could not write summary.json: {e}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Measurement Uncertainty in EMR-Derived Clinical AI Pipelines Simulator"
    )
    parser.add_argument("--output", type=str, default="./sim_outputs",
                        help="Output directory for all generated files (default: ./sim_outputs)")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir.resolve()}")

    # Default base parameters
    base_params = {k: v["value"] for k, v in MODEL_PROFILE["parameters"].items()}

    scenario_runners = [
        ("Scenario_1_False_Critical_Potassium", run_scenario_1),
        ("Scenario_2_Borderline_HbA1c_Diabetes_Label", run_scenario_2),
        ("Scenario_3_Borderline_SBP_Hypertension_Label", run_scenario_3),
        ("Scenario_4_Calibration_Drift_Cross_Site", run_scenario_4),
        ("Scenario_5_Sensitivity_Analysis_Threshold_Perturbation", run_scenario_5),
    ]

    all_rows: List[Dict] = []
    scenario_data: Dict[str, List[Dict]] = {}

    for sc_idx, (sc_label, runner) in enumerate(scenario_runners):
        print(f"\nRunning {sc_label} (seed={sc_idx})...")
        # Get overrides
        overrides = {}
        for sc in MODEL_PROFILE["scenarios"]:
            if sc["label"] == sc_label:
                overrides = sc["param_overrides"]
                break

        params = {**base_params, **overrides}
        try:
            rows = runner(params, seed=sc_idx)
            print(f"  Generated {len(rows)} rows")
            all_rows.extend(rows)
            scenario_data[sc_label] = rows
        except Exception as e:
            print(f"WARNING: Scenario {sc_label} failed: {e}")
            scenario_data[sc_label] = []

    print(f"\nTotal rows across all scenarios: {len(all_rows)}")

    # Write CSVs
    print("\nWriting CSV files...")
    write_simulation_outputs(all_rows, out_dir)
    write_scenario_summary(all_rows, out_dir)
    write_parameters_used(out_dir)

    # Write summary JSON
    print("\nWriting summary JSON...")
    write_summary_json(all_rows, out_dir)

    # Generate figures
    print("\nGenerating figures...")
    s1 = scenario_data.get("Scenario_1_False_Critical_Potassium", [])
    s2 = scenario_data.get("Scenario_2_Borderline_HbA1c_Diabetes_Label", [])
    s3 = scenario_data.get("Scenario_3_Borderline_SBP_Hypertension_Label", [])
    s4 = scenario_data.get("Scenario_4_Calibration_Drift_Cross_Site", [])
    s5 = scenario_data.get("Scenario_5_Sensitivity_Analysis_Threshold_Perturbation", [])

    fig1_hemolysis_potassium(s1, out_dir)
    fig2_hba1c_serial_label_flips(s2, out_dir)
    fig3_sbp_white_coat(s3, out_dir)
    fig4_label_flip_rate_vs_true_value(s2, s3, out_dir)
    fig5_cross_site_prevalence(s4, out_dir)
    fig6_threshold_sensitivity_sweep(s5, out_dir)
    fig7_pipeline_uncertainty_propagation(out_dir)
    fig8_fhir_observation_label_concordance(all_rows, out_dir)

    print(f"\n✓ Simulation complete. All outputs in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()