"""
Flask server for URSA.
Receives a user query, runs it through available tools, returns structured JSON.
"""
import os
import signal
import traceback
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from pyproj import Transformer
from ursa.agent.orchestration import DS, run_agent

load_dotenv()

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "frontend")

app = Flask(__name__)
CORS(app)


@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/query", methods=["POST"])
def query():
    # we need to send a JSON body with a 'message' field to this route
    body = request.get_json()

    if not body or "message" not in body:
        return jsonify({"error": "Request body must include a 'message' field"}), 400

    user_message = body["message"]
    history = body.get("history", [])

    try:
        result = run_agent(user_message, history)

        response = {
            "textResponse": result["text"],
            "dims": result["dims"],
            "charts": result["charts"],
            "toolLog": result["toolLog"],
            "selectionInfo": result["selectionInfo"],
        }

        return jsonify(response), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/dataset/info", methods=["GET"])
def dataset_info():

    _utm_to_latlon = Transformer.from_crs("EPSG:26917", "EPSG:4326",
                                          always_xy=True)

    x_min = float(DS["x"].min())
    x_max = float(DS["x"].max())
    y_min = float(DS["y"].min())
    y_max = float(DS["y"].max())
    lon_sw, lat_sw = _utm_to_latlon.transform(x_min, y_min)
    lon_ne, lat_ne = _utm_to_latlon.transform(x_max, y_max)
    return jsonify({
        "variables": list(DS.data_vars),
        "time_range": {
            "start": str(DS["time"].values[0])[:10],
            "end":   str(DS["time"].values[-1])[:10]
        },
        "spatial_bounds": {
            "x_min": x_min, "x_max": x_max,
            "y_min": y_min, "y_max": y_max
        },
        "lat_lon_bounds": {
            "sw": [round(lat_sw, 5), round(lon_sw, 5)],
            "ne": [round(lat_ne, 5), round(lon_ne, 5)]
        }
    }), 200


def _on_sigterm(signum, frame):
    print("\n[SIGTERM received] Stack trace:", flush=True)
    traceback.print_stack(frame)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _on_sigterm)
    app.run(debug=False, port=5001)
