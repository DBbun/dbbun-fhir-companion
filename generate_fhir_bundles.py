"""
generate_fhir_bundles.py — DBbun Measurement-Aware FHIR Companion
HL7 AI Challenge 2026

Reads sim_outputs/simulation_outputs.csv (produced by Simulator.py) and
exports one FHIR R4 Bundle JSON file per scenario, each containing a
representative sample of synthetic patients with:
  Patient, Observation, Specimen, Device, Condition, Provenance

Usage:
    python generate_fhir_bundles.py
    python generate_fhir_bundles.py --csv sim_outputs/simulation_outputs.csv
                                     --out fhir_bundles --n 5

Author : Uri Kartoun, PhD — DBbun LLC (github.com/DBbun)
License: MIT
"""

import argparse
import json
import uuid
import datetime
from pathlib import Path

import pandas as pd

# ── FHIR terminology systems ──────────────────────────────────────────────────
LOINC   = "http://loinc.org"
SNOMED  = "http://snomed.info/sct"
UCUM    = "http://unitsofmeasure.org"
OBS_INT = "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation"
SPEC_CQ = "http://terminology.hl7.org/CodeSystem/v2-0493"
CLIN    = "http://terminology.hl7.org/CodeSystem/condition-clinical"
VERIF   = "http://terminology.hl7.org/CodeSystem/condition-ver-status"
PROV_T  = "http://terminology.hl7.org/CodeSystem/provenance-participant-type"
DBBUN   = "http://dbbun.com/fhir/StructureDefinition/measurement-reliability"

# ── Scenario metadata ─────────────────────────────────────────────────────────
SCENARIO_META = {
    "Scenario_1_False_Critical_Potassium": {
        "loinc_code":    "2823-3",
        "loinc_display": "Potassium [Moles/volume] in Serum or Plasma",
        "value_col":     "measured_potassium",
        "true_col":      "true_potassium",
        "unit":          "mmol/L",
        "ucum":          "mmol/L",
        "threshold":     6.0,
        "ref_low":       3.5,
        "ref_high":      5.0,
        "snomed_code":   "34227000",
        "snomed_display":"Hyperkalemia (disorder)",
        "label_col":     "potassium_label_error",
        "hemolysis_col": "hemolysis_flag",
        "description":   "Pre-analytical hemolysis bias inflates potassium above critical threshold.",
    },
    "Scenario_2_Borderline_HbA1c_Diabetes_Label": {
        "loinc_code":    "4548-4",
        "loinc_display": "Hemoglobin A1c/Hemoglobin.total in Blood",
        "value_col":     "measured_hba1c",
        "true_col":      "true_hba1c",
        "unit":          "%",
        "ucum":          "%",
        "threshold":     6.5,
        "ref_low":       None,
        "ref_high":      5.6,
        "snomed_code":   "73211009",
        "snomed_display":"Diabetes mellitus (disorder)",
        "label_col":     "diabetes_label_error",
        "hemolysis_col": "hemolysis_flag",
        "description":   "Inter-assay CV and calibration bias cause label flips across diabetes threshold.",
    },
    "Scenario_3_Borderline_SBP_Hypertension_Label": {
        "loinc_code":    "8480-6",
        "loinc_display": "Systolic blood pressure",
        "value_col":     "measured_sbp",
        "true_col":      "true_sbp",
        "unit":          "mmHg",
        "ucum":          "mm[Hg]",
        "threshold":     130.0,
        "ref_low":       None,
        "ref_high":      120.0,
        "snomed_code":   "59621000",
        "snomed_display":"Essential hypertension (disorder)",
        "label_col":     "hypertension_label_error",
        "hemolysis_col": "hemolysis_flag",
        "description":   "White-coat effect and device noise cause oscillation across hypertension threshold.",
    },
    "Scenario_4_Calibration_Drift_Cross_Site": {
        "loinc_code":    "4548-4",
        "loinc_display": "Hemoglobin A1c/Hemoglobin.total in Blood",
        "value_col":     "measured_hba1c",
        "true_col":      "true_hba1c",
        "unit":          "%",
        "ucum":          "%",
        "threshold":     6.5,
        "ref_low":       None,
        "ref_high":      5.6,
        "snomed_code":   "73211009",
        "snomed_display":"Diabetes mellitus (disorder)",
        "label_col":     "diabetes_label_error",
        "hemolysis_col": "hemolysis_flag",
        "description":   "Cross-site calibration drift produces discordant prevalence for identical patients.",
    },
    "Scenario_5_Sensitivity_Analysis_Threshold_Perturbation": {
        "loinc_code":    "4548-4",
        "loinc_display": "Hemoglobin A1c/Hemoglobin.total in Blood",
        "value_col":     "measured_hba1c",
        "true_col":      "true_hba1c",
        "unit":          "%",
        "ucum":          "%",
        "threshold":     6.5,
        "ref_low":       None,
        "ref_high":      5.6,
        "snomed_code":   "73211009",
        "snomed_display":"Diabetes mellitus (disorder)",
        "label_col":     "diabetes_label_error",
        "hemolysis_col": "hemolysis_flag",
        "description":   "Threshold sensitivity sweep quantifies label noise rate across plausible threshold ranges.",
    },
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def uid():
    return str(uuid.uuid4())

def ts():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def reliability_code(row, meta):
    """Derive measurement-reliability extension value from row."""
    val   = row[meta["value_col"]]
    true_ = row[meta["true_col"]]
    thr   = meta["threshold"]
    hem   = row.get(meta["hemolysis_col"], 0)

    if hem and val > thr and (val - true_) > 0.8:
        return "artifact-suspected"
    if abs(val - thr) <= 0.3 * thr * 0.1 + 0.5:
        return "borderline-repeat-required"
    return "reliable"

def interp(val, threshold):
    if val >= threshold * 1.1:
        return "HH", "Critical high"
    if val >= threshold:
        return "H",  "High"
    return "N", "Normal"

def make_patient(pid):
    return {
        "resourceType": "Patient",
        "id": pid,
        "identifier": [{"system": "http://dbbun.com/synthetic", "value": f"SYN-{pid}"}],
        "extension": [{"url": "http://dbbun.com/fhir/synthetic-patient", "valueBoolean": True}]
    }

def make_specimen(sid, pid, hemolyzed):
    cond_code    = "HEM" if hemolyzed else "OK"
    cond_display = "Hemolyzed" if hemolyzed else "Satisfactory"
    return {
        "resourceType": "Specimen",
        "id": sid,
        "subject": {"reference": f"Patient/{pid}"},
        "collection": {"collectedDateTime": ts()},
        "condition": [{"coding": [{"system": SPEC_CQ, "code": cond_code,
                                    "display": cond_display}]}]
    }

def make_device(did, scenario_name):
    devices = {
        "Scenario_1": ("Abbott Architect c16000",  "Abbott Diagnostics",    "c16000-SYN"),
        "Scenario_2": ("Bio-Rad Variant II Turbo",  "Bio-Rad Laboratories",  "VariantII-SYN"),
        "Scenario_3": ("Omron HEM-907XL",           "Omron Healthcare",      "HEM907XL-SYN"),
        "Scenario_4": ("Multi-Site Analyzer",        "DBbun Synthetic",       "MULTI-SYN"),
        "Scenario_5": ("Threshold-Sweep Analyzer",   "DBbun Synthetic",       "SWEEP-SYN"),
    }
    key = next((k for k in devices if scenario_name.startswith(k)), "Scenario_1")
    name, mfr, model = devices[key]
    return {
        "resourceType": "Device",
        "id": did,
        "deviceName": [{"name": name, "type": "user-friendly-name"}],
        "manufacturer": mfr,
        "modelNumber": model,
        "note": [{"text": "Synthetic device — DBbun HL7 AI Challenge 2026."}]
    }

def make_observation(oid, pid, sid, did, meta, row):
    val = round(float(row[meta["value_col"]]), 3)
    int_code, int_display = interp(val, meta["threshold"])
    rel = reliability_code(row, meta)

    ref_range = {}
    if meta["ref_low"] is not None:
        ref_range["low"]  = {"value": meta["ref_low"],  "unit": meta["unit"],
                              "system": UCUM, "code": meta["ucum"]}
    if meta["ref_high"] is not None:
        ref_range["high"] = {"value": meta["ref_high"], "unit": meta["unit"],
                              "system": UCUM, "code": meta["ucum"]}

    true_val = round(float(row[meta["true_col"]]), 3)
    label_err = int(row.get(meta["label_col"], 0))

    return {
        "resourceType": "Observation",
        "id": oid,
        "status": "final",
        "code": {"coding": [{"system": LOINC, "code": meta["loinc_code"],
                              "display": meta["loinc_display"]}]},
        "subject":          {"reference": f"Patient/{pid}"},
        "effectiveDateTime": ts(),
        "valueQuantity": {
            "value": val, "unit": meta["unit"],
            "system": UCUM, "code": meta["ucum"]
        },
        "interpretation": [{"coding": [{"system": OBS_INT,
                                         "code": int_code, "display": int_display}]}],
        "referenceRange": [ref_range] if ref_range else [],
        "specimen": {"reference": f"Specimen/{sid}"},
        "device":   {"reference": f"Device/{did}"},
        "note": [{"text": (f"True value: {true_val} {meta['unit']}. "
                           f"Measured: {val} {meta['unit']}. "
                           f"Label error: {bool(label_err)}. "
                           f"Reliability: {rel}.")}],
        "extension": [{"url": DBBUN, "valueCode": rel}]
    }

def make_condition(cid, pid, meta, row):
    val      = float(row[meta["value_col"]])
    positive = val >= meta["threshold"]
    label_err= int(row.get(meta["label_col"], 0))

    if label_err:
        ver_code    = "entered-in-error"
        ver_display = "Label assigned in error — measurement artifact or borderline noise"
        clin_code   = "inactive"
    elif positive:
        ver_code    = "confirmed"
        ver_display = "Confirmed"
        clin_code   = "active"
    else:
        ver_code    = "refuted"
        ver_display = "Refuted"
        clin_code   = "inactive"

    return {
        "resourceType": "Condition",
        "id": cid,
        "subject": {"reference": f"Patient/{pid}"},
        "clinicalStatus":     {"coding": [{"system": CLIN,  "code": clin_code}]},
        "verificationStatus": {"coding": [{"system": VERIF, "code": ver_code,
                                            "display": ver_display}]},
        "code": {"coding": [{"system": SNOMED, "code": meta["snomed_code"],
                              "display": meta["snomed_display"]}]}
    }

def make_provenance(prid, obs_ref, cond_ref, scenario_name, meta, row):
    val      = round(float(row[meta["value_col"]]), 3)
    true_val = round(float(row[meta["true_col"]]), 3)
    return {
        "resourceType": "Provenance",
        "id": prid,
        "target": [{"reference": obs_ref}, {"reference": cond_ref}],
        "recorded": ts(),
        "agent": [{
            "type": {"coding": [{"system": PROV_T, "code": "author"}]},
            "who":  {"display": "DBbun LLC — generate_fhir_bundles.py (Synthetic)"}
        }],
        "entity": [{"role": "source",
                    "what": {"display": f"{scenario_name} — {meta['description']}"}}],
        "extension": [{"url": "http://dbbun.com/fhir/scenario-config",
                       "valueString": (f"scenario={scenario_name}; "
                                       f"true={true_val}; measured={val}; "
                                       f"threshold={meta['threshold']}; "
                                       f"label_error={int(row.get(meta['label_col'], 0))}")}]
    }

def make_bundle(entries, scenario_name, description):
    return {
        "resourceType": "Bundle",
        "id": uid(),
        "meta": {"tag": [{
            "system":  "http://dbbun.com/fhir/bundle-type",
            "code":    "synthetic-evaluation",
            "display": "DBbun Synthetic Evaluation Bundle — NOT real patient data"
        }]},
        "type": "collection",
        "timestamp": ts(),
        "entry": [{"fullUrl": f"urn:uuid:{e['id']}", "resource": e} for e in entries],
        "extension": [
            {"url": "http://dbbun.com/fhir/scenario-name",    "valueString": scenario_name},
            {"url": "http://dbbun.com/fhir/scenario-description", "valueString": description},
            {"url": "http://dbbun.com/fhir/bundle-note",
             "valueString": "Generated by generate_fhir_bundles.py from simulation_outputs.csv. "
                            "DBbun LLC — HL7 AI Challenge 2026. All data synthetic."}
        ]
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Generate FHIR R4 bundles from simulation_outputs.csv")
    parser.add_argument("--csv", default="sim_outputs/simulation_outputs.csv",
                        help="Path to simulation_outputs.csv")
    parser.add_argument("--out", default="fhir_bundles",
                        help="Output directory for FHIR Bundle JSON files")
    parser.add_argument("--n",   type=int, default=5,
                        help="Number of representative patients per scenario bundle")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run Simulator.py first.")
        return

    print(f"Reading {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"  {len(df):,} rows, columns: {list(df.columns[:8])} ...")

    # Identify scenario column
    scenario_col = next((c for c in df.columns if "scenario" in c.lower()), None)
    if scenario_col is None:
        # Simulator may not have a scenario column — infer from row count blocks
        print("  No scenario column found; inferring from row blocks.")
        n_scenarios = len(SCENARIO_META)
        rows_each   = len(df) // n_scenarios
        df[scenario_col := "scenario"] = [
            list(SCENARIO_META.keys())[i // rows_each]
            if i // rows_each < n_scenarios else list(SCENARIO_META.keys())[-1]
            for i in range(len(df))
        ]

    scenarios_found = df[scenario_col].unique()
    print(f"  Scenarios found: {list(scenarios_found)}")

    for sc_name, meta in SCENARIO_META.items():
        # Match scenario name flexibly
        match = next((s for s in scenarios_found if sc_name in s or s in sc_name), None)
        if match is None:
            print(f"  Skipping {sc_name} (not found in CSV)")
            continue

        sc_df = df[df[scenario_col] == match]

        # Select n representative rows: mix of label-error and non-error
        errors   = sc_df[sc_df[meta["label_col"]] == 1].head(args.n // 2 + 1)
        non_err  = sc_df[sc_df[meta["label_col"]] == 0].head(args.n - len(errors))
        sample   = pd.concat([errors, non_err]).head(args.n)

        entries = []
        for i, (_, row) in enumerate(sample.iterrows()):
            pid  = f"{sc_name[:4]}-{i:03d}"
            sid  = uid()[:8]
            did  = uid()[:8]
            oid  = uid()[:8]
            cid  = uid()[:8]
            prid = uid()[:8]

            hemolyzed = bool(row.get(meta["hemolysis_col"], 0))

            patient   = make_patient(pid)
            specimen  = make_specimen(sid, pid, hemolyzed)
            device    = make_device(did, sc_name)
            obs       = make_observation(oid, pid, sid, did, meta, row)
            cond      = make_condition(cid, pid, meta, row)
            prov      = make_provenance(prid,
                                        f"Observation/{oid}", f"Condition/{cid}",
                                        sc_name, meta, row)
            entries.extend([patient, specimen, device, obs, cond, prov])

        bundle = make_bundle(entries, sc_name, meta["description"])
        fname  = out_dir / f"{sc_name.lower()}.json"
        with open(fname, "w") as f:
            json.dump(bundle, f, indent=2)

        n_entries  = len(bundle["entry"])
        n_patients = len(sample)
        err_rate   = round(sample[meta["label_col"]].mean() * 100, 1)
        print(f"  {fname.name}: {n_patients} patients, "
              f"{n_entries} resources, label error rate {err_rate}%")

    print(f"\nDone. FHIR bundles written to ./{out_dir}/")
    print("Each bundle contains Patient, Observation, Specimen, Device, Condition, Provenance.")


if __name__ == "__main__":
    main()
