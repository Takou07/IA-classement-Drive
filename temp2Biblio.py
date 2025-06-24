import gradio as gr
import fitz  # PyMuPDF
import re
import csv
import os
from sentence_transformers import SentenceTransformer, util
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

# Dictionnaire thèmes (nom utilisateur -> code)
THEMES = {
    "Développement personnel": "DevPerso",
    "Politique": "Politique",
    "Finance": "Finance",
    "Réseaux informatiques": "Reseaux",
    "Nanosciences": "Nano",
    "Mathématiques": "Maths",
    "Algorithmes et programmation": "Algo",
    "Machine Learning": "ML",
    "Deep Learning": "DL",
    "Large Language Models": "LLM",
    "Data Engineering": "DE",
    "Reinforcement Learning": "RL",
    "Statistiques / Mathématiques": "Maths"
}

theme_descriptions = {
    "DevPerso": "Personal development focuses on improving motivation, discipline, mindset, and productivity.",
    "Politique": "Politics deals with governance, history, colonialism, diplomacy, and society.",
    "Finance": "Finance involves money, investing, banking, economics, and financial systems.",
    "Reseaux": "Networks include communication systems, internet, social media, and influence structures.",
    "Nano": "Nanosciences cover quantum physics, atoms, molecules, and material science at the nanoscale.",
    "Maths": "Mathematics includes algebra, statistics, probability, calculus, and logical reasoning.",
    "Algo": "Algorithms involve programming, problem-solving, code efficiency, and computational logic.",
    "ML": "Machine learning teaches computers to learn from data using models and patterns.",
    "DL": "Deep learning is a subset of ML focusing on neural networks with many layers and large data.",
    "LLM": "Large Language Models like GPT or BERT are AI models that understand and generate human language.",
    "DE": "Data Engineering involves building data pipelines, using tools like Spark, ETL processes, and databases.",
    "RL": "Reinforcement Learning is a type of ML where agents learn through rewards and interactions."
}

model = SentenceTransformer("all-mpnet-base-v2")

def connecter_drive():
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()
    drive = GoogleDrive(gauth)
    return drive

drive = connecter_drive()

def nettoyer_texte(texte):
    texte = re.sub(r'\s+', ' ', texte)
    texte = re.sub(r'[^\x00-\x7F]+', '', texte)
    return texte.strip()

def lire_pdf(path):
    try:
        doc = fitz.open(path)
        texte = ""
        for page in doc:
            texte += page.get_text()
        return nettoyer_texte(texte)
    except Exception:
        return ""

def trouver_ou_creer_dossier(drive, nom):
    # On cherche par le nom complet (ex: "Développement personnel")
    file_list = drive.ListFile({
        'q': f"title='{nom}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    }).GetList()
    if file_list:
        return file_list[0]['id']
    else:
        folder = drive.CreateFile({'title': nom, 'mimeType': 'application/vnd.google-apps.folder'})
        folder.Upload()
        return folder['id']

def uploader_vers_drive(fichier_path, nom, dossier_id):
    f = drive.CreateFile({'title': nom, 'parents': [{'id': dossier_id}]})
    f.SetContentFile(fichier_path)
    f.Upload()

def classer_et_feedback(file, feedback):
    texte = lire_pdf(file.name)
    if not texte.strip():
        return "❌ Fichier vide ou illisible"

    emb_doc = model.encode(texte, convert_to_tensor=True)
    scores = {}
    for theme_nom, code in THEMES.items():
        description = theme_descriptions.get(code, "")
        score = util.cos_sim(emb_doc, model.encode(description, convert_to_tensor=True)).item()
        scores[theme_nom] = score

    top3 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
    best_theme = top3[0][0]
    resultat = f"📊 **Top 3 suggestions :**\n"
    for nom, sc in top3:
        resultat += f"- **{nom}** : `{sc:.4f}`\n"

    # Correction via menu déroulant (nom complet)
    correction = feedback if feedback and feedback.strip() else best_theme

    with open("feedback.csv", mode="a", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([os.path.basename(file.name), best_theme, correction])

    dossier_id = trouver_ou_creer_dossier(drive, correction)
    uploader_vers_drive(file.name, os.path.basename(file.name), dossier_id)

    resultat += f"\n✅ **Classement final : `{correction}`**\n🚀 Fichier envoyé dans Google Drive."
    return resultat

def compter_livres_par_dossier(drive):
    resultats = {}
    for nom in THEMES.keys():  # On cherche par nom complet ici
        file_list = drive.ListFile({
            'q': f"title='{nom}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
        }).GetList()
        if not file_list:
            resultats[nom] = 0
            continue
        folder_id = file_list[0]['id']

        fichiers = drive.ListFile({
            'q': f"'{folder_id}' in parents and mimeType='application/pdf' and trashed=false"
        }).GetList()
        resultats[nom] = len(fichiers)
    return resultats

def afficher_nombre_livres():
    comptes = compter_livres_par_dossier(drive)
    texte = "| 📁 Dossier | 📄 Nombre de PDF |\n|---|---|\n"
    for dossier, nb in comptes.items():
        texte += f"| {dossier} | {nb} |\n"
    return texte

# Interface Gradio
with gr.Blocks() as demo:
    with gr.Tab("📂 Upload & Classement"):
        fichier_pdf = gr.File(label="📤 Upload ton fichier PDF")
        correction_theme = gr.Dropdown(list(THEMES.keys()), label="📝 Corriger le thème (optionnel)", interactive=True)
        sortie = gr.Markdown()
        btn_classer = gr.Button("Classifier et envoyer")

        btn_classer.click(fn=classer_et_feedback,
                          inputs=[fichier_pdf, correction_theme],
                          outputs=sortie)

    with gr.Tab("📊 Stats Drive"):
        btn_stats = gr.Button("Afficher nombre de livres par dossier")
        sortie_stats = gr.Markdown()
        btn_stats.click(fn=afficher_nombre_livres, inputs=[], outputs=sortie_stats)

demo.launch()
