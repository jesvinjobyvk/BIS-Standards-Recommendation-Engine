#!/usr/bin/env python3
"""
BIS Standards Recommendation Engine — Professional Web UI
FastAPI backend with comprehensive error handling and validation.
"""

import argparse
import json
import sys
import traceback
import uvicorn
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Add src/ to path so imports resolve regardless of working directory
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rag_engine import retrieve, BIS_STANDARDS  # type: ignore

# ── Pydantic Models ─────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=1000, description="Product description query")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results to return")

    @validator("query")
    def query_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()


class StandardResult(BaseModel):
    rank: int
    code: str
    id: str
    title: str
    category: str
    relevance_score: int
    raw_score: float
    description: str
    keywords: List[str]
    applications: List[str]
    test_standards: List[str]


class SearchResponse(BaseModel):
    query: str
    top_k: int
    results: List[StandardResult]
    latency_seconds: float
    total_standards_in_kb: int


class BatchResultItem(BaseModel):
    id: str
    retrieved_standards: List[str]
    latency_seconds: float
    results_detail: Optional[List[StandardResult]] = None


class BatchResponse(BaseModel):
    total_queries: int
    avg_latency_seconds: float
    results: List[BatchResultItem]


class ErrorResponse(BaseModel):
    detail: str
    error_type: str
    suggestion: Optional[str] = None


# ── FastAPI App ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="BIS Standards Recommendation Engine",
    description="AI-powered BIS standard discovery for Building Materials (BIS SP 21)",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (frontend)
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ── API Endpoints ───────────────────────────────────────────────────────────

@app.get("/")
async def root():
    """Serve the main UI page."""
    index_file = static_dir / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "BIS Standards API is running. Visit /static/index.html for the UI."}


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "standards_loaded": len(BIS_STANDARDS),
        "version": "1.0.0"
    }


@app.get("/api/standards")
async def list_standards():
    """List all available BIS standards in the knowledge base."""
    return {
        "total": len(BIS_STANDARDS),
        "categories": sorted(list(set(s["category"] for s in BIS_STANDARDS))),
        "standards": [
            {
                "code": s["code"],
                "title": s["title"],
                "category": s["category"],
                "keywords": s.get("keywords", [])[:5]
            }
            for s in BIS_STANDARDS
        ]
    }


@app.post("/api/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Search for relevant BIS standards based on a product description.
    """
    import time
    try:
        t_start = time.perf_counter()
        retrieved = retrieve(request.query, top_k=request.top_k)
        t_end = time.perf_counter()
        latency = round(t_end - t_start, 4)

        results = []
        for r in retrieved:
            results.append(StandardResult(
                rank=r["rank"],
                code=r["code"],
                id=r["id"],
                title=r["title"],
                category=r["category"],
                relevance_score=r["relevance_score"],
                raw_score=r["raw_score"],
                description=r["description"],
                keywords=r.get("keywords", []),
                applications=r.get("applications", []),
                test_standards=r.get("test_standards", [])
            ))

        return SearchResponse(
            query=request.query,
            top_k=request.top_k,
            results=results,
            latency_seconds=latency,
            total_standards_in_kb=len(BIS_STANDARDS)
        )

    except Exception as e:
        traceback_str = traceback.format_exc()
        print(f"[ERROR] Search failed: {e}\n{traceback_str}", file=sys.stderr)
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                detail=str(e),
                error_type=type(e).__name__,
                suggestion="Please try again with a different query or contact support."
            ).dict()
        )


@app.post("/api/batch", response_model=BatchResponse)
async def batch_search(file: UploadFile = File(...)):
    """
    Process a batch of queries from a JSON file.
    Expected format: [{"id": "q1", "query": "..."}, ...]
    """
    import time

    # Validate file type
    if not file.filename.endswith(".json"):
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(
                detail="Only JSON files are supported",
                error_type="InvalidFileType",
                suggestion="Please upload a .json file with the format: [{\"id\": \"q1\", \"query\": \"...\"}]"
            ).dict()
        )

    try:
        contents = await file.read()
        data = json.loads(contents.decode("utf-8"))

        if not isinstance(data, list):
            raise HTTPException(
                status_code=400,
                detail=ErrorResponse(
                    detail="JSON must be a list of query objects",
                    error_type="InvalidFormat",
                    suggestion="Use format: [{\"id\": \"q1\", \"query\": \"...\"}]"
                ).dict()
            )

        results = []
        total_latency = 0.0

        for i, item in enumerate(data):
            if not isinstance(item, dict):
                continue

            qid = item.get("id", f"q{i+1}")
            query_text = str(item.get("query", "")).strip()

            if not query_text:
                results.append(BatchResultItem(
                    id=qid,
                    retrieved_standards=[],
                    latency_seconds=0.0,
                    results_detail=[]
                ))
                continue

            t_start = time.perf_counter()
            retrieved = retrieve(query_text, top_k=5)
            t_end = time.perf_counter()
            latency = round(t_end - t_start, 4)
            total_latency += latency

            standard_codes = [r["code"] for r in retrieved]
            detail = [
                StandardResult(
                    rank=r["rank"],
                    code=r["code"],
                    id=r["id"],
                    title=r["title"],
                    category=r["category"],
                    relevance_score=r["relevance_score"],
                    raw_score=r["raw_score"],
                    description=r["description"],
                    keywords=r.get("keywords", []),
                    applications=r.get("applications", []),
                    test_standards=r.get("test_standards", [])
                )
                for r in retrieved
            ]

            results.append(BatchResultItem(
                id=qid,
                retrieved_standards=standard_codes,
                latency_seconds=latency,
                results_detail=detail
            ))

        avg_latency = round(total_latency / max(len(data), 1), 4)

        return BatchResponse(
            total_queries=len(data),
            avg_latency_seconds=avg_latency,
            results=results
        )

    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(
                detail=f"Invalid JSON file: {str(e)}",
                error_type="JSONDecodeError",
                suggestion="Please ensure the file is valid JSON and properly formatted."
            ).dict()
        )
    except Exception as e:
        traceback_str = traceback.format_exc()
        print(f"[ERROR] Batch processing failed: {e}\n{traceback_str}", file=sys.stderr)
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                detail=str(e),
                error_type=type(e).__name__,
                suggestion="Please check your input file format and try again."
            ).dict()
        )


# ── Global Exception Handler ────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    traceback_str = traceback.format_exc()
    print(f"[ERROR] Unhandled exception: {exc}\n{traceback_str}", file=sys.stderr)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            detail="An unexpected error occurred",
            error_type=type(exc).__name__,
            suggestion="Please refresh the page and try again."
        ).dict()
    )


# ── CLI Entry Point ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BIS Standards Web UI Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("  BIS Standards Recommendation Engine — Web UI")
    print(f"{'='*60}")
    print(f"  🌐 Open your browser: http://{args.host}:{args.port}")
    print(f"  📚 API Docs: http://{args.host}:{args.port}/api/docs")
    print(f"  📁 Static dir: {static_dir}")
    print(f"{'='*60}\n")

    uvicorn.run("web_app:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()

