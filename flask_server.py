from flask import Flask, request, jsonify
from kmodes.kprototypes import KPrototypes
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np
from pymongo import MongoClient
import os

app = Flask(__name__)

@app.route('/cluster', methods=['POST'])
def cluster_data():
    try:
        input_data = request.get_json()
        if not input_data or 'data' not in input_data:
            return jsonify({"error": "Data tidak ditemukan"}), 400
        df = pd.DataFrame(input_data['data'])

        if len(df) < 3:
            return jsonify({"error": "Minimal 3 data diperlukan."}), 400

        categorical = ['nama_am', 'customer', 'pekerjaan', 'stage', 'portofolio']
        numerical = ['sales_amount']

        matrix_encoded = pd.get_dummies(df[categorical])
        matrix_all = pd.concat([matrix_encoded, df[numerical]], axis=1)

        max_k = min(6, len(df) - 1)
        best_score = -1
        best_k = 2
        best_labels = None

        for k in range(2, max_k + 1):
            try:
                kproto = KPrototypes(n_clusters=k, init='Cao', random_state=42)
                clusters = kproto.fit_predict(df[categorical + numerical].to_numpy(), categorical=[0,1,2,3,4])
                if len(set(clusters)) < 2 or len(set(clusters)) >= len(df):
                    continue

                score = silhouette_score(matrix_all, clusters)
                if score > best_score:
                    best_score = score
                    best_k = k
                    best_labels = clusters
            except Exception as e:
                print(f"Error pada k={k}: {str(e)}")

        if best_labels is None:
            return jsonify({"error": "Gagal menentukan cluster."}), 500

        df['cluster'] = best_labels
        cluster_descriptions = {
            0: "Prospek Tinggi",
            1: "Prospek Sedang",
            2: "Prospek Rendah",
            3: "Potensi Lemah",
            4: "Tidak Tertarik",
            5: "Prospek Potensial"
        }
        df['deskripsi_cluster'] = df['cluster'].map(cluster_descriptions)

        result = df.to_dict(orient='records')

        # Simpan ke MongoDB
        try:
            mongo_uri = os.environ.get("MONGO_URI")  # Ambil dari Environment Variable
            client = MongoClient(mongo_uri)
            db = client["sistem-web-skripsi"]  # ← disesuaikan dengan nama database kamu
            cluster_collection = db["hasil_clusters"]
            meta_collection = db["cluster_metadata"]

            cluster_collection.delete_many({})
            cluster_collection.insert_many(result)

            meta_collection.delete_many({})
            meta_collection.insert_one({
                "silhouette_score": best_score,
                "k_terbaik": best_k
            })
        except Exception as e:
            return jsonify({"error": f"Gagal simpan ke MongoDB: {e}"}), 500

        for r in result:
            r.pop('_id', None)

        return jsonify({
            "data": result,
            "silhouette_score": best_score,
            "k_terbaik": best_k
        })

    except Exception as e:
        return jsonify({"error": f"Terjadi kesalahan: {e}"}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)