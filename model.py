# model.py
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import traceback
import h2o
from h2o.frame import H2OFrame

# --- utilidades propias
from utils import read_json_data, get_filter_configuration, aplicar_filtros, build_features_dataframe

# Inicializa H2O (requiere Java en la imagen)
h2o.init(nthreads=-1, min_mem_size="1G")  # ajusta memoria si hace falta
MODEL_PATH = os.path.join(os.path.dirname(__file__), "mojo", "XGBoost.zip")
model = h2o.import_mojo(MODEL_PATH)

app = Flask(__name__, template_folder="templates")
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route("/", methods=["GET"])
def home():
    # landing simple para comprobar que levantó
    return render_template("index.html")

@app.route("/health", methods=["GET"])
def health():
    return jsonify(status="ok"), 200

@app.route("/", methods=["POST"])
def predict():
    try:
        payload = request.get_json(force=True, silent=False)
        # 1) parseo & meta
        df_signals, meta = read_json_data(payload)      # <- devuélveme pandas.DataFrame y dict meta
        config = get_filter_configuration(payload)      # <- tu config de filtros

        # 2) filtros (pueden ser no-op si no configuras)
        df_filt = aplicar_filtros(df_signals, config, fs=meta.get("sampling", 3333))

        # 3) features para el modelo
        X = build_features_dataframe(df_filt, meta)     # <- pandas.DataFrame con columnas que espera el mojo

        # 4) predicción con H2O
        hf = H2OFrame(X)
        pred_h2o = model.predict(hf).as_data_frame()    # pandas DataFrame
        pred = pred_h2o.iloc[:, 0].tolist()             # primera columna

        # 5) prepara respuesta (evita NaN no serializables)
        import numpy as np
        def clean_list(a):
            out = []
            for x in a:
                if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
                    out.append(None)
                else:
                    out.append(float(x) if isinstance(x, (int, float, np.number)) else x)
            return out

        response = {
            "status": "ok",
            "meta": {
                "sampling": meta.get("sampling"),
                "count": len(pred)
            },
            "results": {
                "depth": clean_list(df_filt["Depth"].tolist() if "Depth" in df_filt else []),
                "prediction": clean_list(pred)
            }
        }
        return jsonify(response), 200

    except Exception as e:
        print("ERROR:", e)
        traceback.print_exc()
        return jsonify(status="error", error=str(e)), 500

if __name__ == "__main__":
    # para debug local: python model.py
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port, debug=True)
