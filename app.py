"""
ShoeMetrics - Application de collecte et analyse descriptive des chaussures
Backend Flask avec analyse statistique intégrée
"""

from flask import Flask, jsonify, request, render_template, send_file
import sqlite3
import json
import os
import random
from datetime import datetime, timedelta
import statistics
import math

app = Flask(__name__)
DB_PATH = os.path.join(os.path.dirname(__file__), 'data', 'shoes.db')

# ─────────────────────────────────────────────
#  BASE DE DONNÉES
# ─────────────────────────────────────────────

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = get_db()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS chaussures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            marque TEXT NOT NULL,
            modele TEXT NOT NULL,
            categorie TEXT NOT NULL,
            prix REAL NOT NULL,
            pointure REAL NOT NULL,
            couleur TEXT NOT NULL,
            matiere TEXT NOT NULL,
            score_creativite INTEGER NOT NULL CHECK(score_creativite BETWEEN 1 AND 10),
            score_robustesse INTEGER NOT NULL CHECK(score_robustesse BETWEEN 1 AND 10),
            score_efficacite INTEGER NOT NULL CHECK(score_efficacite BETWEEN 1 AND 10),
            poids_grammes REAL,
            duree_vie_mois INTEGER,
            note_globale REAL,
            date_ajout TEXT NOT NULL,
            commentaire TEXT
        )
    ''')
    conn.commit()

    # Vérifier si des données existent déjà
    count = c.execute('SELECT COUNT(*) FROM chaussures').fetchone()[0]
    if count == 0:
        seed_data(conn)
    conn.close()

def seed_data(conn):
    """Données initiales réalistes pour démonstration"""
    marques = ['Nike', 'Adidas', 'New Balance', 'Salomon', 'Timberland',
               'Converse', 'Puma', 'Reebok', 'Vans', 'Merrell']
    categories = ['Running', 'Randonnée', 'Lifestyle', 'Sport', 'Travail', 'Casual']
    couleurs = ['Noir', 'Blanc', 'Gris', 'Bleu', 'Rouge', 'Vert', 'Beige', 'Marron']
    matieres = ['Cuir', 'Synthétique', 'Mesh', 'Toile', 'Gore-Tex', 'Nubuck']
    modeles_base = ['Pro X', 'Trail Boost', 'Urban Flex', 'Classic', 'Elite',
                    'Enduro', 'Storm', 'Flow', 'Peak', 'Core']

    random.seed(42)
    rows = []
    base_date = datetime(2024, 1, 1)

    for i in range(80):
        marque = random.choice(marques)
        cat = random.choice(categories)
        # Corrélations réalistes : randonnée = + robustesse
        if cat == 'Randonnée':
            rob = random.randint(6, 10)
            eff = random.randint(5, 9)
            crea = random.randint(3, 7)
            poids = random.uniform(350, 600)
            duree = random.randint(24, 60)
        elif cat == 'Running':
            rob = random.randint(4, 8)
            eff = random.randint(7, 10)
            crea = random.randint(4, 8)
            poids = random.uniform(200, 350)
            duree = random.randint(12, 30)
        elif cat == 'Lifestyle':
            rob = random.randint(3, 7)
            eff = random.randint(3, 7)
            crea = random.randint(6, 10)
            poids = random.uniform(250, 500)
            duree = random.randint(18, 42)
        else:
            rob = random.randint(4, 9)
            eff = random.randint(4, 9)
            crea = random.randint(3, 9)
            poids = random.uniform(250, 550)
            duree = random.randint(12, 48)

        note = round((crea + rob + eff) / 3 + random.uniform(-0.5, 0.5), 1)
        note = max(1.0, min(10.0, note))
        date_ajout = (base_date + timedelta(days=random.randint(0, 450))).strftime('%Y-%m-%d')

        rows.append((
            marque,
            f"{marque} {random.choice(modeles_base)} {random.randint(1,9)}",
            cat,
            round(random.uniform(40, 380), 2),
            random.choice([36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]),
            random.choice(couleurs),
            random.choice(matieres),
            crea, rob, eff,
            round(poids, 1),
            duree,
            note,
            date_ajout,
            ''
        ))

    conn.executemany('''
        INSERT INTO chaussures
        (marque, modele, categorie, prix, pointure, couleur, matiere,
         score_creativite, score_robustesse, score_efficacite,
         poids_grammes, duree_vie_mois, note_globale, date_ajout, commentaire)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    ''', rows)
    conn.commit()

# ─────────────────────────────────────────────
#  UTILITAIRES STATISTIQUES (sans numpy/pandas)
# ─────────────────────────────────────────────

def describe(values):
    """Calcul des statistiques descriptives"""
    if not values:
        return {}
    n = len(values)
    sorted_v = sorted(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / n
    std = math.sqrt(variance)
    median = sorted_v[n // 2] if n % 2 != 0 else (sorted_v[n // 2 - 1] + sorted_v[n // 2]) / 2
    q1 = sorted_v[n // 4]
    q3 = sorted_v[3 * n // 4]
    return {
        'n': n,
        'mean': round(mean, 2),
        'std': round(std, 2),
        'min': round(min(values), 2),
        'q1': round(q1, 2),
        'median': round(median, 2),
        'q3': round(q3, 2),
        'max': round(max(values), 2),
        'range': round(max(values) - min(values), 2),
        'cv': round((std / mean * 100), 2) if mean != 0 else 0
    }

def pearson_corr(x, y):
    """Coefficient de corrélation de Pearson"""
    n = len(x)
    if n < 2:
        return 0
    mx, my = sum(x)/n, sum(y)/n
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    den = math.sqrt(sum((xi - mx)**2 for xi in x) * sum((yi - my)**2 for yi in y))
    return round(num / den, 3) if den != 0 else 0

def frequency(values):
    freq = {}
    for v in values:
        freq[v] = freq.get(v, 0) + 1
    return dict(sorted(freq.items(), key=lambda x: -x[1]))

# ─────────────────────────────────────────────
#  ROUTES API
# ─────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')

# --- CRUD Chaussures ---

@app.route('/api/chaussures', methods=['GET'])
def get_chaussures():
    conn = get_db()
    rows = conn.execute('SELECT * FROM chaussures ORDER BY date_ajout DESC').fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

@app.route('/api/chaussures', methods=['POST'])
def add_chaussure():
    d = request.get_json()
    required = ['marque', 'modele', 'categorie', 'prix', 'pointure', 'couleur',
                'matiere', 'score_creativite', 'score_robustesse', 'score_efficacite']
    for f in required:
        if f not in d:
            return jsonify({'error': f'Champ manquant: {f}'}), 400

    sc = int(d['score_creativite'])
    sr = int(d['score_robustesse'])
    se = int(d['score_efficacite'])
    note = round((sc + sr + se) / 3, 1)

    conn = get_db()
    cur = conn.execute('''
        INSERT INTO chaussures
        (marque, modele, categorie, prix, pointure, couleur, matiere,
         score_creativite, score_robustesse, score_efficacite,
         poids_grammes, duree_vie_mois, note_globale, date_ajout, commentaire)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    ''', (
        d['marque'], d['modele'], d['categorie'],
        float(d['prix']), float(d['pointure']),
        d['couleur'], d['matiere'],
        sc, sr, se,
        d.get('poids_grammes'), d.get('duree_vie_mois'),
        note,
        datetime.now().strftime('%Y-%m-%d'),
        d.get('commentaire', '')
    ))
    new_id = cur.lastrowid
    conn.commit()
    row = conn.execute('SELECT * FROM chaussures WHERE id=?', (new_id,)).fetchone()
    conn.close()
    return jsonify(dict(row)), 201

@app.route('/api/chaussures/<int:shoe_id>', methods=['DELETE'])
def delete_chaussure(shoe_id):
    conn = get_db()
    conn.execute('DELETE FROM chaussures WHERE id=?', (shoe_id,))
    conn.commit()
    conn.close()
    return jsonify({'message': 'Supprimé'}), 200

# --- ANALYSES ---

@app.route('/api/analyse/global')
def analyse_global():
    conn = get_db()
    rows = conn.execute('SELECT * FROM chaussures').fetchall()
    conn.close()
    data = [dict(r) for r in rows]
    if not data:
        return jsonify({'error': 'Aucune donnée'})

    return jsonify({
        'total': len(data),
        'prix': describe([r['prix'] for r in data]),
        'score_creativite': describe([r['score_creativite'] for r in data]),
        'score_robustesse': describe([r['score_robustesse'] for r in data]),
        'score_efficacite': describe([r['score_efficacite'] for r in data]),
        'note_globale': describe([r['note_globale'] for r in data if r['note_globale']]),
        'poids': describe([r['poids_grammes'] for r in data if r['poids_grammes']]),
        'duree_vie': describe([r['duree_vie_mois'] for r in data if r['duree_vie_mois']]),
    })

@app.route('/api/analyse/categories')
def analyse_categories():
    conn = get_db()
    rows = conn.execute('SELECT * FROM chaussures').fetchall()
    conn.close()
    data = [dict(r) for r in rows]

    cats = {}
    for r in data:
        cat = r['categorie']
        if cat not in cats:
            cats[cat] = {'prix': [], 'crea': [], 'rob': [], 'eff': [], 'note': [], 'count': 0}
        cats[cat]['prix'].append(r['prix'])
        cats[cat]['crea'].append(r['score_creativite'])
        cats[cat]['rob'].append(r['score_robustesse'])
        cats[cat]['eff'].append(r['score_efficacite'])
        if r['note_globale']:
            cats[cat]['note'].append(r['note_globale'])
        cats[cat]['count'] += 1

    result = {}
    for cat, v in cats.items():
        result[cat] = {
            'count': v['count'],
            'prix_moyen': round(sum(v['prix']) / len(v['prix']), 2),
            'crea_moyen': round(sum(v['crea']) / len(v['crea']), 2),
            'rob_moyen': round(sum(v['rob']) / len(v['rob']), 2),
            'eff_moyen': round(sum(v['eff']) / len(v['eff']), 2),
            'note_moyen': round(sum(v['note']) / len(v['note']), 2) if v['note'] else 0,
        }
    return jsonify(result)

@app.route('/api/analyse/marques')
def analyse_marques():
    conn = get_db()
    rows = conn.execute('''
        SELECT marque,
               COUNT(*) as count,
               AVG(prix) as prix_moyen,
               AVG(score_creativite) as crea_moyen,
               AVG(score_robustesse) as rob_moyen,
               AVG(score_efficacite) as eff_moyen,
               AVG(note_globale) as note_moyen
        FROM chaussures
        GROUP BY marque
        ORDER BY count DESC
    ''').fetchall()
    conn.close()
    return jsonify([{
        'marque': r['marque'],
        'count': r['count'],
        'prix_moyen': round(r['prix_moyen'], 2),
        'crea_moyen': round(r['crea_moyen'], 2),
        'rob_moyen': round(r['rob_moyen'], 2),
        'eff_moyen': round(r['eff_moyen'], 2),
        'note_moyen': round(r['note_moyen'] or 0, 2),
    } for r in rows])

@app.route('/api/analyse/correlations')
def analyse_correlations():
    conn = get_db()
    rows = conn.execute('SELECT prix, score_creativite, score_robustesse, score_efficacite, poids_grammes, duree_vie_mois, note_globale FROM chaussures').fetchall()
    conn.close()
    data = [dict(r) for r in rows]

    fields = {
        'prix': [r['prix'] for r in data],
        'créativité': [r['score_creativite'] for r in data],
        'robustesse': [r['score_robustesse'] for r in data],
        'efficacité': [r['score_efficacite'] for r in data],
        'poids': [r['poids_grammes'] for r in data if r['poids_grammes']],
        'durée_vie': [r['duree_vie_mois'] for r in data if r['duree_vie_mois']],
    }

    correlations = {}
    keys = list(fields.keys())
    for i, k1 in enumerate(keys):
        for k2 in keys[i+1:]:
            v1, v2 = fields[k1], fields[k2]
            n = min(len(v1), len(v2))
            correlations[f'{k1}×{k2}'] = pearson_corr(v1[:n], v2[:n])

    return jsonify(correlations)

@app.route('/api/analyse/distribution')
def analyse_distribution():
    conn = get_db()
    rows = conn.execute('SELECT score_creativite, score_robustesse, score_efficacite, prix, couleur, matiere FROM chaussures').fetchall()
    conn.close()
    data = [dict(r) for r in rows]

    # Distribution des scores
    def dist_scores(field):
        vals = [r[field] for r in data]
        freq = {str(i): vals.count(i) for i in range(1, 11)}
        return freq

    # Tranches de prix
    prix = sorted([r['prix'] for r in data])
    tranches = {'0-80': 0, '80-150': 0, '150-250': 0, '250+': 0}
    for p in prix:
        if p < 80:
            tranches['0-80'] += 1
        elif p < 150:
            tranches['80-150'] += 1
        elif p < 250:
            tranches['150-250'] += 1
        else:
            tranches['250+'] += 1

    return jsonify({
        'creativite': dist_scores('score_creativite'),
        'robustesse': dist_scores('score_robustesse'),
        'efficacite': dist_scores('score_efficacite'),
        'tranches_prix': tranches,
        'couleurs': frequency([r['couleur'] for r in data]),
        'matieres': frequency([r['matiere'] for r in data]),
    })

@app.route('/api/analyse/top')
def analyse_top():
    conn = get_db()
    # Top créativité
    top_crea = conn.execute('''SELECT marque, modele, score_creativite, score_robustesse, score_efficacite, prix
                               FROM chaussures ORDER BY score_creativite DESC LIMIT 5''').fetchall()
    top_rob = conn.execute('''SELECT marque, modele, score_creativite, score_robustesse, score_efficacite, prix
                              FROM chaussures ORDER BY score_robustesse DESC LIMIT 5''').fetchall()
    top_eff = conn.execute('''SELECT marque, modele, score_creativite, score_robustesse, score_efficacite, prix
                              FROM chaussures ORDER BY score_efficacite DESC LIMIT 5''').fetchall()
    top_global = conn.execute('''SELECT marque, modele, score_creativite, score_robustesse, score_efficacite, note_globale, prix
                                 FROM chaussures ORDER BY note_globale DESC LIMIT 5''').fetchall()
    conn.close()
    return jsonify({
        'top_creativite': [dict(r) for r in top_crea],
        'top_robustesse': [dict(r) for r in top_rob],
        'top_efficacite': [dict(r) for r in top_eff],
        'top_global': [dict(r) for r in top_global],
    })

@app.route('/api/analyse/tendances')
def analyse_tendances():
    conn = get_db()
    rows = conn.execute('''
        SELECT strftime('%Y-%m', date_ajout) as mois,
               COUNT(*) as count,
               AVG(score_creativite) as crea,
               AVG(score_robustesse) as rob,
               AVG(score_efficacite) as eff,
               AVG(prix) as prix
        FROM chaussures
        GROUP BY mois
        ORDER BY mois
    ''').fetchall()
    conn.close()
    return jsonify([{
        'mois': r['mois'],
        'count': r['count'],
        'crea': round(r['crea'], 2),
        'rob': round(r['rob'], 2),
        'eff': round(r['eff'], 2),
        'prix': round(r['prix'], 2),
    } for r in rows])

if __name__ == '__main__':
    init_db()
    print("🥾 ShoeMetrics démarré sur http://127.0.0.1:5000")
    app.run(debug=True, port=5000)