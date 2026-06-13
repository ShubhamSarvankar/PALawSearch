"""
BM25 Indexer - Baseline retrieval method (optimized with bulk indexing)
"""

import os
import json
import zipfile

from tqdm import tqdm
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk, BulkIndexError

from config.settings import settings

from datetime import datetime
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def normalize_decision_date(raw):
    """
    Normalize decision_date string into ISO 'YYYY-MM-DD' or return None.

    Handles:
      - 'YYYY-MM-DD'
      - 'YYYY-MM'        -> YYYY-MM-01
      - 'YYYY'           -> YYYY-01-01
    Any unparsable value -> None (field omitted).
    """
    if not raw:
        return None

    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return None
    else:
        # non-string (e.g., None) -> ignore
        return None

    # Try a few likely formats
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            dt = datetime.strptime(raw, fmt)
            return dt.date().isoformat()  # 'YYYY-MM-DD'
        except ValueError:
            continue

    # Unrecognized format: skip the field
    return None

def create_bm25_index():
    """
    Create BM25 index with custom analyzer and
    indexing-friendly settings (no replicas, no auto-refresh).
    """
    es = Elasticsearch(
        settings.es_host,
        basic_auth=("elastic", settings.es_password),
        verify_certs=False,
    )

    mapping = {
        "settings": {
            "analysis": {
                "analyzer": {
                    "legal_text_analyzer": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": [
                            "lowercase",
                            "english_stop",
                            "english_stemmer"
                        ]
                    }
                },
                "filter": {
                    "english_stop": {
                        "type": "stop",
                        "stopwords": "_english_"
                    },
                    "english_stemmer": {
                        "type": "stemmer",
                        "language": "english"
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "id": {"type": "keyword"},

                # Boosted fields – use legal_text_analyzer
                "name": {
                    "type": "text",
                    "analyzer": "legal_text_analyzer"
                },
                "parties": {
                    "type": "text",
                    "analyzer": "legal_text_analyzer"
                },
                "judges": {
                    "type": "text",
                    "analyzer": "legal_text_analyzer"
                },
                "head_matter": {
                    "type": "text",
                    "analyzer": "legal_text_analyzer"
                },

                # Metadata fields
                "decision_date": {"type": "date"},
                "court_name": {
                    "type": "keyword",
                    "fields": {
                        "text": {
                            "type": "text",
                            "analyzer": "legal_text_analyzer"
                        }
                    }
                },

                "jurisdiction_name": {"type": "keyword"},
                "word_count": {"type": "integer"},

                # Main body
                "full_text": {
                    "type": "text",
                    "analyzer": "legal_text_analyzer"
                }
            }
        }
    }


    if es.indices.exists(index=settings.es_index_bm25):
        print(f"Index {settings.es_index_bm25} already exists. Deleting...")
        es.indices.delete(index=settings.es_index_bm25)

    es.indices.create(index=settings.es_index_bm25, body=mapping)
    print(f"Created index: {settings.es_index_bm25}")

    return es


def collect_json_files(data_dir="data"):
    """
    Collect all (zip_path, json_filename) pairs to index.
    """
    json_files_info = []

    for folder_name in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        for zip_name in os.listdir(folder_path):
            if not zip_name.endswith(".zip"):
                continue

            zip_path = os.path.join(folder_path, zip_name)
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                for file_info in zip_ref.namelist():
                    if file_info.endswith(".json") and "json/" in file_info:
                        json_files_info.append((zip_path, file_info))

    return json_files_info


def build_doc(case_data):
    """
    Build the ES document from a single case JSON dict.
    """
    casebody = case_data.get("casebody", {})
    opinions = casebody.get("opinions", [])
    opinions_text = "\n".join(
        [op.get("text", "") for op in opinions if "text" in op]
    )
    head_matter = casebody.get("head_matter", "")

    if head_matter:
        full_text = head_matter + "\n" + opinions_text
    else:
        full_text = opinions_text

    parties = casebody.get("parties", [])
    judges = casebody.get("judges", [])
    analysis = case_data.get("analysis", {})
    word_count = analysis.get("word_count", 0)
    court = case_data.get("court", {})
    jurisdiction = case_data.get("jurisdiction", {})

    decision_date = normalize_decision_date(case_data.get("decision_date"))

    doc = {
        "id": case_data.get("id"),
        "name": case_data.get("name"),
        "decision_date": decision_date,
        "court_name": court.get("name"),
        "jurisdiction_name": jurisdiction.get("name"),
        "parties": ", ".join(parties) if parties else "",
        "judges": ", ".join(judges) if judges else "",
        "word_count": word_count,
        "head_matter": head_matter or "",
        "full_text": full_text,
    }

    return doc


def index_documents(es, data_dir="data", batch_size=500):
    """
    Index all documents from data directory using Elasticsearch bulk API.
    """
    json_files_info = collect_json_files(data_dir=data_dir)
    total_docs = len(json_files_info)
    print(f"Found {total_docs} documents to index")

    actions = []
    for zip_path, json_filename in tqdm(
        json_files_info, desc="Indexing BM25 cases"
    ):
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            with zip_ref.open(json_filename) as f:
                case_data = json.load(f)

        doc = build_doc(case_data)

        action = {
            "_index": settings.es_index_bm25,
            "_source": doc,
        }
        actions.append(action)

        # When we reach batch_size, send a bulk request
        if len(actions) >= batch_size:
            try:
                bulk(es, actions, request_timeout=300)
            except BulkIndexError as e:
                print("Bulk indexing error; first error:", e.errors[0])
                raise
            actions = []  # reset batch

    # Flush any remaining docs
    if actions:
        try:
            bulk(es, actions, request_timeout=300)
        except BulkIndexError as e:
            print("Bulk indexing error in final batch; first error:", e.errors[0])
            raise

    print(f"Indexed {total_docs} documents to {settings.es_index_bm25}")

    # Optional: restore more normal index settings after bulk indexing
    try:
        # Give ES more time to apply settings on a big index
        es.options(request_timeout=60).indices.put_settings(
            index=settings.es_index_bm25,
            body={"index": {"refresh_interval": "1s"}}
        )
    except Exception as e:
        print(f"Warning: failed to update index refresh_interval: {e}")

def index_from_jsonl(es, jsonl_path="data/parsed/cases.jsonl", batch_size=500):
    """
    Index all documents from the pre-parsed cases.jsonl produced by ingest/parse.py.
    Fields in the JSONL are already flat strings; no ZIP extraction needed.
    """
    import json as _json

    actions = []
    total = 0

    with open(jsonl_path, encoding="utf-8") as f:
        for line in tqdm(f, desc="Indexing BM25 cases"):
            case = _json.loads(line)
            doc = {
                "id":               case.get("id"),
                "name":             case.get("name", ""),
                "decision_date":    normalize_decision_date(case.get("decision_date")),
                "court_name":       case.get("court_name", ""),
                "jurisdiction_name": case.get("jurisdiction", ""),
                "parties":          case.get("parties", ""),
                "judges":           case.get("judges", ""),
                "word_count":       case.get("word_count", 0),
                "head_matter":      case.get("head_matter", ""),
                "full_text":        case.get("full_text", ""),
            }
            actions.append({"_index": settings.es_index_bm25, "_source": doc})
            total += 1

            if len(actions) >= batch_size:
                try:
                    bulk(es, actions, request_timeout=300)
                except BulkIndexError as e:
                    print("Bulk indexing error; first error:", e.errors[0])
                    raise
                actions = []

    if actions:
        try:
            bulk(es, actions, request_timeout=300)
        except BulkIndexError as e:
            print("Bulk indexing error in final batch:", e.errors[0])
            raise

    print(f"Indexed {total} documents to {settings.es_index_bm25}")

    try:
        es.options(request_timeout=60).indices.put_settings(
            index=settings.es_index_bm25,
            body={"index": {"refresh_interval": "1s"}}
        )
    except Exception as e:
        print(f"Warning: failed to restore refresh_interval: {e}")


if __name__ == "__main__":
    print("Creating BM25 index...")
    es = create_bm25_index()

    print("Indexing documents from data/parsed/cases.jsonl ...")
    index_from_jsonl(es, jsonl_path="data/parsed/cases.jsonl", batch_size=500)

    print("Done!")
