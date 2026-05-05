from flask import Flask, request, jsonify
from functools import wraps
import os
import traceback
from utils import process_json_data

app = Flask(__name__)


def require_internal_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        expected = os.environ.get("INTERNAL_API_KEY", "")
        if not expected:
            return jsonify({"status": "error", "message": "API no configurada."}), 500
        provided = request.headers.get("X-Internal-Key", "")
        if not provided or provided != expected:
            return jsonify({"status": "error", "message": "No autorizado."}), 401
        return f(*args, **kwargs)
    return decorated


@app.route('/', methods=['POST'])
@require_internal_key
def procesar():
    """
    Endpoint para procesar archivos JSON de datos sísmicos y devolver datos para gráficos.
    """
    try:        
        json_data = request.get_json(force=True)
        result_data = process_json_data(json_data)
        return jsonify({
                "status": "success",
                "data": result_data
            })
        
    except Exception as e:
        error_message = f"Error al procesar el archivo: {str(e)}"
        return jsonify({
            "status": "error", 
            "message": error_message,
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5320, debug=False)
