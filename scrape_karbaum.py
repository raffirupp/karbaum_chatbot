"""
scrape_karbaum.py
-----------------
Scraped alle Blogartikel von Dr. Markus Karbaum (https://karrierecoaching.eu/)
und speichert sie als JSON-Datei (articles.json).
"""

import feedparser
import requests
from bs4 import BeautifulSoup
import json
from time import sleep

# ---------------------------------------------
# 1️⃣ Kategorien-Feeds definieren
# ---------------------------------------------
feeds = {
    "bewerbungscoaching": "https://karrierecoaching.eu/category/bewerbungscoaching/feed/",
    "erfolgreich-fuehren": "https://karrierecoaching.eu/category/erfolgreich-fuehren/feed/",
    "fuer-gruenderinnen-und-gruender": "https://karrierecoaching.eu/category/fuer-gruenderinnen-und-gruender/feed/",
    "karriere": "https://karrierecoaching.eu/category/karriere/feed/",
    "personalarbeit": "https://karrierecoaching.eu/category/personalarbeit/feed/",
    "resilienz-und-selbstkompetenz": "https://karrierecoaching.eu/category/resilienz-und-selbstkompetenz/feed/",
    "selbstvermarktung": "https://karrierecoaching.eu/category/selbstvermarktung/feed/",
}

# ---------------------------------------------
# 2️⃣ Feed-Artikel sammeln (Titel + URLs)
# ---------------------------------------------
def sammle_artikel_links():
    all_articles = []

    for cat, url in feeds.items():
        feed = feedparser.parse(url)
        print(f"🔹 Kategorie: {cat} ({len(feed.entries)} Artikel)")

        for entry in feed.entries:
            all_articles.append({
                "category": cat,
                "title": entry.title,
                "url": entry.link
            })

    print(f"\n✅ Insgesamt {len(all_articles)} Artikel-Links gesammelt.")
    return all_articles

# ---------------------------------------------
# 3️⃣ Volltexte abrufen
# ---------------------------------------------
def lade_volltexte(artikel_liste):
    results = []

    for i, art in enumerate(artikel_liste, start=1):
        try:
            r = requests.get(art["url"], timeout=10)
            soup = BeautifulSoup(r.text, "html.parser")

            content_div = soup.find("div", class_="entry-content")
            if not content_div:
                print(f"⚠️ Kein Text gefunden für: {art['title']}")
                continue

            text = content_div.get_text(separator="\n").strip()

            results.append({
                "category": art["category"],
                "title": art["title"],
                "url": art["url"],
                "content": text
            })

            print(f"{i}/{len(artikel_liste)} ✅ {art['title']}")
            sleep(1)

        except Exception as e:
            print(f"{i}/{len(artikel_liste)} ⚠️ Fehler bei {art['url']}: {e}")

    return results

# ---------------------------------------------
# 4️⃣ Hauptprogramm
# ---------------------------------------------
def main():
    artikel_links = sammle_artikel_links()
    artikel_volltexte = lade_volltexte(artikel_links)

    with open("articles.json", "w", encoding="utf-8") as f:
        json.dump(artikel_volltexte, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Fertig! {len(artikel_volltexte)} Artikel gespeichert in articles.json")

# ---------------------------------------------
# 5️⃣ Script direkt ausführbar machen
# ---------------------------------------------
if __name__ == "__main__":
    main()
