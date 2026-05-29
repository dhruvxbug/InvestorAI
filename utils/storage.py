import os
import uuid
from datetime import datetime
from pathlib import Path

import markdown
from weasyprint import HTML, CSS

import chromadb
from chromadb.config import Settings

from models.report_schema import StockReport


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> list[str]:
    """Splits text into chunks of `chunk_size` characters with `overlap` characters of overlap."""
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= text_len:
            break
        start = end - overlap
    return chunks


def vector_db_upsert(report: StockReport, markdown_text: str):
    """Upserts the report's markdown text into ChromaDB."""
    db_path = Path("./investor_db")
    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_or_create_collection(name="historical_analysis")

    chunks = chunk_text(markdown_text, chunk_size=1000, overlap=100)
    for i, chunk in enumerate(chunks):
        chunk_id = f"{report.ticker}_{report.analysis_date}_{i}"
        
        metadata = {
            "ticker": report.ticker,
            "date": report.analysis_date,
            "report_type": "stock_analysis"
        }
        
        try:
            collection.add(
                documents=[chunk],
                metadatas=[metadata],
                ids=[chunk_id]
            )
        except Exception as e:
            # Upsert fallback
            try:
                collection.upsert(
                    documents=[chunk],
                    metadatas=[metadata],
                    ids=[chunk_id]
                )
            except Exception as e2:
                print(f"[VectorDB] Failed to upsert chunk {chunk_id}: {e2}")


def search_past_analysis(query: str, ticker: str = None, n_results: int = 3) -> list[str]:
    """Searches historical long-term memory for past analysis."""
    db_path = Path("./investor_db")
    if not db_path.exists():
        return []
        
    client = chromadb.PersistentClient(path=str(db_path))
    try:
        collection = client.get_collection(name="historical_analysis")
    except Exception:
        return []

    where_clause = {}
    if ticker:
        where_clause["ticker"] = ticker
        
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where_clause if where_clause else None
    )
    
    if results and "documents" in results and results["documents"]:
        return results["documents"][0]
    return []


def save_report_and_export(
    report: StockReport,
    markdown_text: str,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Persist JSON and export a PDF report, replacing the old markdown files. Also saves to DB."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d")
    stem = f"{report.ticker}_{timestamp}"
    json_path = output_dir / f"{stem}.json"
    pdf_path = output_dir / f"{stem}.pdf"

    # Save JSON
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    
    # Save to memory
    try:
        vector_db_upsert(report, markdown_text)
    except Exception as e:
        print(f"Warning: Failed to save to vector database: {e}")
    
    # Generate PDF
    try:
        html_content = markdown.markdown(
            markdown_text, 
            extensions=['tables', 'fenced_code']
        )
        
        css = CSS(string='''
            @page { margin: 2cm; }
            body { font-family: system-ui, -apple-system, sans-serif; font-size: 11pt; line-height: 1.5; color: #333; }
            h1 { font-size: 20pt; border-bottom: 2px solid #333; padding-bottom: 5px; color: #111; }
            h2 { font-size: 16pt; margin-top: 20px; color: #222; }
            h3 { font-size: 13pt; margin-top: 15px; }
            table { width: 100%; border-collapse: collapse; margin-block: 15px; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f8f9fa; font-weight: 600; }
            code { background-color: #f4f4f4; padding: 2px 5px; font-family: monospace; border-radius: 3px; }
            hr { border: 0; border-top: 1px solid #eee; margin: 20px 0; }
        ''')
        
        HTML(string=html_content).write_pdf(pdf_path, stylesheets=[css])
    except Exception as e:
        print(f"Failed to generate PDF: {e}. Falling back to MD file.")
        md_path = output_dir / f"{stem}.md"
        md_path.write_text(markdown_text, encoding="utf-8")
        return json_path, md_path
        
    return json_path, pdf_path
