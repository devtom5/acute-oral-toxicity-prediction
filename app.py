import os
import joblib
import requests
import numpy as np
import pandas as pd
import streamlit as st

from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D
# -----------------------------
# Optional Mordred support
# -----------------------------
MORDRED_AVAILABLE = True
try:
    import numpy as _np
    if not hasattr(_np, "product"):
        _np.product = _np.prod
    from mordred import Calculator, descriptors
    MORDRED_CALC = Calculator(descriptors, ignore_3D=True)
except Exception:
    MORDRED_AVAILABLE = False
    MORDRED_CALC = None

# -----------------------------
# Streamlit page config
# -----------------------------
st.set_page_config(
    page_title="Acute Oral Toxicity Prediction Platform",
    page_icon="🧪",
    layout="wide"
)

# -----------------------------
# Global visit counter
# -----------------------------
def get_global_count():
    try:
        url = "https://api.countapi.xyz/hit/gnu-acute-toxicity-app/visits"
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            data = r.json()
            return data.get("value", "N/A")
        return "N/A"
    except Exception:
        return "N/A"

# -----------------------------
# Scientific styling
# -----------------------------
st.markdown(
    """
    <style>
    .main {
        background-color: #f8fafc;
    }
    .banner {
        background: linear-gradient(90deg, #0f172a 0%, #1d4ed8 100%);
        color: white;
        padding: 1.25rem 1.4rem;
        border-radius: 20px;
        margin-bottom: 1rem;
        box-shadow: 0 8px 24px rgba(15,23,42,0.18);
    }
    .banner h1 {
        margin: 0;
        font-size: 1.9rem;
        line-height: 1.2;
    }
    .banner p {
        margin: 0.35rem 0 0 0;
        font-size: 0.98rem;
        opacity: 0.96;
    }
    .card {
        background: white;
        border: 1px solid #dbe4ee;
        border-radius: 16px;
        padding: 0.95rem 1rem;
        box-shadow: 0 6px 18px rgba(15,23,42,0.05);
        margin-bottom: 0.8rem;
    }
    .counter-box {
        background: white;
        border: 1px solid #dbe4ee;
        border-radius: 14px;
        padding: 12px 16px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.06);
        margin-top: 12px;
        margin-bottom: 8px;
        font-size: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Header banner
# -----------------------------
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("""
    <div style="background:linear-gradient(90deg,#0f172a,#1e3a8a);
    padding:25px;
    border-radius:18px;
    color:white;
    box-shadow:0 8px 24px rgba(0,0,0,0.2);">

    <h1 style="margin-bottom:10px;">Acute Oral Toxicity Prediction</h1>

    <b>Laboratory of Pharmacology and Toxicology</b><br>
    College of Veterinary Medicine<br>
    Gyeongsang National University<br>
    Jinju 52828, Republic of Korea

    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="background:white;
    padding:18px;
    border-radius:14px;
    border:1px solid #dbe4ee;
    box-shadow:0 4px 15px rgba(0,0,0,0.08);">
    <b>Contact information</b><br><br>
    Kim, Euikyung (김의경)<br>
    Professor<br><br>
    Phone: 055-772-2355<br>
    Email: ekim@gnu.ac.kr<br>
    Office: 501-308<br>
    </div>
    """, unsafe_allow_html=True)

st.caption("Draw a molecule or paste a SMILES string to predict acute oral toxicity compounds using the trained stacked model.")

# -----------------------------
# Fixed model paths
# -----------------------------
MODEL_PATH = "Stack_Top5.joblib"
FEATURES_PATH = "final_selected_features.joblib"
IMPUTER_PATH = "median_imputer.joblib"

missing_files = []
for f in [MODEL_PATH, FEATURES_PATH, IMPUTER_PATH]:
    if not os.path.exists(f):
        missing_files.append(f)

if missing_files:
    st.error("Missing required file(s): " + ", ".join(missing_files))
    st.stop()

# -----------------------------
# Helper functions
# -----------------------------
def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

@st.cache_resource(show_spinner=False)
def load_artifacts(model_path, features_path, imputer_path):
    model = joblib.load(model_path)
    selected_features = joblib.load(features_path)
    imputer = joblib.load(imputer_path)
    return model, selected_features, imputer

def split_feature_groups(selected_features):
    groups = {
        "Morgan": [],
        "PubChem": [],
        "AtomPairs": [],
        "MACCS": [],
        "Mordred": []
    }

    for f in selected_features:
        sf = str(f)
        if sf.startswith("Morgan_") or sf.startswith("morgan_"):
            groups["Morgan"].append(f)
        elif sf.startswith("PubChem_") or sf.startswith("PubchemFP") or sf.startswith("pubchem_"):
            groups["PubChem"].append(f)
        elif sf.startswith("AtomPairs_") or sf.startswith("AP_"):
            groups["AtomPairs"].append(f)
        elif sf.startswith("MACCS_") or sf.startswith("maccs_"):
            groups["MACCS"].append(f)
        else:
            groups["Mordred"].append(f)
    return groups

def infer_nbits(feature_list, default_nbits):
    if not feature_list:
        return default_nbits
    idxs = []
    for f in feature_list:
        try:
            idxs.append(int(str(f).split("_")[-1]))
        except Exception:
            pass
    return max(max(idxs) + 1, default_nbits) if idxs else default_nbits

def smiles_to_mol(smiles):
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    return Chem.MolFromSmiles(smiles.strip())

def mol_to_png_bytes(mol, size=(500, 300)):
    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()

def compute_morgan(mol, nbits):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=nbits)
    return {f"Morgan_{i}": int(fp.GetBit(i)) for i in range(nbits)}

def compute_maccs(mol):
    fp = MACCSkeys.GenMACCSKeys(mol)
    return {f"MACCS_{i}": int(fp.GetBit(i)) for i in range(1, 167)}

def compute_atompairs(mol, nbits):
    fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=nbits)
    return {f"AtomPairs_{i}": int(fp.GetBit(i)) for i in range(nbits)}

def compute_mordred_selected(mol, needed_mordred):
    out = {f: np.nan for f in needed_mordred}
    if not MORDRED_AVAILABLE or mol is None:
        return out
    try:
        vals = MORDRED_CALC(mol).asdict()
        for full_name in needed_mordred:
            raw_name = full_name.replace("Mordred_", "")
            out[full_name] = safe_float(vals.get(raw_name, np.nan))
    except Exception:
        pass
    return out

def compute_pubchem_placeholder(needed_pubchem):
    return {f: 0.0 for f in needed_pubchem}

def build_feature_row(smiles, selected_features):
    mol = smiles_to_mol(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string.")

    groups = split_feature_groups(selected_features)
    morgan_nbits = infer_nbits(groups["Morgan"], 2048)
    ap_nbits = infer_nbits(groups["AtomPairs"], 2048)

    row = {}
    if groups["Morgan"]:
        row.update(compute_morgan(mol, morgan_nbits))
    if groups["MACCS"]:
        row.update(compute_maccs(mol))
    if groups["AtomPairs"]:
        row.update(compute_atompairs(mol, ap_nbits))
    if groups["Mordred"]:
        row.update(compute_mordred_selected(mol, groups["Mordred"]))
    if groups["PubChem"]:
        row.update(compute_pubchem_placeholder(groups["PubChem"]))

    df = pd.DataFrame([row])

    for f in selected_features:
        if f not in df.columns:
            if str(f).startswith(("Morgan_", "morgan_", "PubChem_", "PubchemFP", "pubchem_", "AtomPairs_", "AP_", "MACCS_", "maccs_")):
                df[f] = 0.0
            else:
                df[f] = np.nan

    df = df[selected_features]
    df = df.apply(pd.to_numeric, errors="coerce")
    return df

def predict_one(smiles, model, selected_features, imputer):
    feat_df = build_feature_row(smiles, selected_features)
    X = imputer.transform(feat_df)
    pred = int(model.predict(X)[0])
    prob = float(model.predict_proba(X)[0, 1]) if hasattr(model, "predict_proba") else float(pred)
    return pred, prob

# -----------------------------
# Load artifacts
# -----------------------------
try:
    model, selected_features, imputer = load_artifacts(MODEL_PATH, FEATURES_PATH, IMPUTER_PATH)
    if not MORDRED_AVAILABLE:
        st.warning("Mordred not available. Mordred features will be missing.")
except Exception as e:
    st.error(f"Could not load files:\n\n{e}")
    st.stop()

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2 = st.tabs(["Single compound", "Batch prediction"])

with tab1:
    st.subheader("Single-compound toxicity prediction")
    smiles = st.text_area("Enter SMILES", value="CC(=O)Oc1ccccc1C(=O)O", height=100)

    st.markdown("### Molecule drawing canvas")
    st.caption("Draw a molecule below and use the generated SMILES for prediction.")

    try:
        from streamlit_ketcher import st_ketcher
        drawn_smiles = st_ketcher(height=450)
        if drawn_smiles:
            st.success(f"Editor SMILES: {drawn_smiles}")

        if st.button("Use drawn molecule"):
            if drawn_smiles:
                st.session_state["drawn_smiles"] = drawn_smiles
            else:
                st.warning("No molecule drawn yet.")

        if "drawn_smiles" in st.session_state:
            smiles = st.session_state["drawn_smiles"]

    except Exception:
        st.info("To enable drawing, install: pip install streamlit-ketcher")

    if st.button("Predict toxicity", type="primary"):
        try:
            pred, prob = predict_one(smiles, model, selected_features, imputer)
            mol = smiles_to_mol(smiles)

            c1, c2 = st.columns([1, 1])

            with c1:
                st.markdown("### Prediction summary")
                st.metric("Predicted class", "Toxic" if pred == 1 else "Non-toxic")
                st.metric("Toxicity probability", f"{prob:.4f}")

                gauge_html = f"""
                <div class='card'>
                    <div style='font-weight:600; margin-bottom:0.55rem;'>Prediction probability</div>
                    <div style='width:100%; height:18px; background:#e5e7eb; border-radius:999px; overflow:hidden;'>
                        <div style='width:{prob*100:.1f}%; height:18px; background:{'#dc2626' if prob >= 0.8 else '#f59e0b' if prob >= 0.4 else '#16a34a'};'></div>
                    </div>
                    <div style='display:flex; justify-content:space-between; font-size:0.85rem; margin-top:0.35rem;'>
                        <span>0.0</span><span>{prob:.4f}</span><span>1.0</span>
                    </div>
                </div>
                """
                st.markdown(gauge_html, unsafe_allow_html=True)

                if prob >= 0.8:
                    st.error("High predicted toxicity")
                elif prob >= 0.5:
                    st.warning("Moderate predicted toxicity")
                else:
                    st.success("Non-toxic")

            with c2:
                st.markdown("### Chemical structure")
                if mol is not None:
                    png = mol_to_png_bytes(mol)
                    st.image(png, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

with tab2:
    st.subheader("Batch prediction from CSV")
    st.markdown(
        "<div class='card'>Upload a CSV containing a <code>SMILES</code> column for multi-compound toxicity screening.</div>",
        unsafe_allow_html=True
    )
    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:
        batch_df = pd.read_csv(uploaded)
        st.write("Preview")
        st.dataframe(batch_df.head(), use_container_width=True)

        if "SMILES" not in batch_df.columns:
            st.error("CSV must contain a `SMILES` column.")
        else:
            if st.button("Run batch prediction"):
                results = []
                for _, row in batch_df.iterrows():
                    smi = str(row["SMILES"])
                    try:
                        pred, prob = predict_one(smi, model, selected_features, imputer)
                        results.append({
                            **row.to_dict(),
                            "Predicted_Class": "Toxic" if pred == 1 else "Non-toxic",
                            "Predicted_Label": pred,
                            "Toxicity_Probability": prob,
                        })
                    except Exception as e:
                        results.append({
                            **row.to_dict(),
                            "Predicted_Class": "Error",
                            "Predicted_Label": np.nan,
                            "Toxicity_Probability": np.nan,
                            "Error": str(e),
                        })

                res_df = pd.DataFrame(results)
                st.success("Batch prediction complete")
                st.dataframe(res_df, use_container_width=True)

                csv = res_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download predictions CSV",
                    csv,
                    file_name="toxicity_predictions.csv",
                    mime="text/csv",
                )
st.markdown("---")

visit_count = get_global_count()

st.markdown(
    f"""
    <div style="
        background:white;
        border:1px solid #dbe4ee;
        border-radius:16px;
        padding:18px;
        box-shadow:0 4px 15px rgba(0,0,0,0.06);
        margin-top:18px;
        margin-bottom:18px;
    ">
        <h4 style="margin-top:0; margin-bottom:10px;">Global visitors</h4>
        <div style="font-size:1.05rem; margin-bottom:8px;">🌍 Total app visits: <b>{visit_count}</b></div>
        <div style="font-size:0.9rem; color:#475569;">
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
**Laboratory of Pharmacology and Toxicology**  
**College of Veterinary Medicine**  
**Gyeongsang National University, Jinju 52828, Republic of Korea**

**Contact information**  
Kim Euikyung (김의경)  
Professor  
Phone: 055-772-2355  
Email: ekim@gnu.ac.kr  
Office: 501-308
"""
)

