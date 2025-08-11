from flask import Flask, request, jsonify
import os
import traceback
from utils_resistividad import process_json_data

# ========= FLASK SETUP =============
app = Flask(__name__)

@app.route('/', methods=['POST'])
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
    app.run(host="0.0.0.0", port=5420, debug=False)
