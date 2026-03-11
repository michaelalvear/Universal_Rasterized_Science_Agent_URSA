"""
Flask server for URSA.
Receives a user query, runs it through available tools, returns structured JSON.
"""

import os
import sys
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

# tools.py imports from schemas.py using a relative name, so src/ must be on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from orchestration import run_agent

app = Flask(__name__)
CORS(app)


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
            "chartType": result["chart_type"],
            "labels": result["labels"],
            "data": result["data"],
        }

        return jsonify(response), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/dataset/info", methods=["GET"])
def dataset_info():
    from orchestration import DS
    return jsonify({
        "variables": list(DS.data_vars),
        "time_range": {
            "start": str(DS["time"].values[0])[:10],
            "end":   str(DS["time"].values[-1])[:10]
        },
        "spatial_bounds": {
            "x_min": float(DS["x"].min()),
            "x_max": float(DS["x"].max()),
            "y_min": float(DS["y"].min()),
            "y_max": float(DS["y"].max())
        }
    }), 200


if __name__ == "__main__":
    app.run(debug=True)
