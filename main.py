


# #############################################################
#  to delete the old data and metedata
# #############################################################

import os
import shutil

db_path = "rag_app.db"
chroma_dir = "chroma_db"

if os.path.exists(db_path):
    os.remove(db_path)
    print(f"✅ Removed existing database file: {db_path}")

if os.path.isdir(chroma_dir):
    shutil.rmtree(chroma_dir)
    print(f"✅ Removed existing ChromaDB dir: {chroma_dir}")





from fastapi.templating import Jinja2Templates         # << needed to pass into the template

templates = Jinja2Templates(directory="templates")




from fastapi import FastAPI,  HTTPException, Request
from pydantic_models import QueryInput, QueryResponse, DocumentInfo, DeleteFileRequest
from langchain_utils import get_rag_chain
from routes_upload import router as upload_router
from routes_metadata import router as metadata_router
from dotenv import load_dotenv
from datetime import datetime


from db_utils import (
    insert_application_logs, get_chat_history, get_all_documents,
    delete_document_record,
)
from chroma_utils import  delete_doc_from_chroma

import  uuid, logging
from fastapi.responses import HTMLResponse
load_dotenv()
chat_model = os.getenv("LLM_for_chat", "openai/gpt-oss-120b")

logging.basicConfig(filename="app.log", level=logging.INFO)
app = FastAPI()
app.include_router(upload_router)
app.include_router(metadata_router, prefix="/api", tags=["metadata"]) 
print("✅ use this url to access the app http://localhost:8000  or http://127.0.0.1:8000/ ✅ ")







# ────────────────────────── CHAT ───────────────────────────
@app.post("/chat", response_model=QueryResponse)
def chat(query: QueryInput):
    session_id = query.session_id or str(uuid.uuid4())
    logging.info(
        "session=%s question=%s model=%s",
        session_id, query.question, chat_model,
    )

    chat_history = get_chat_history(session_id)
    chain = get_rag_chain(chat_model)
    answer = chain.invoke(
        {"input": query.question, "chat_history": chat_history}
    )["answer"]

    insert_application_logs(session_id, query.question, answer, chat_model)
    return QueryResponse(
        answer=answer,
        session_id=session_id,
        model=chat_model,
    )




# ───────────────────────── LIST / DELETE DOCS ──────────────
@app.get("/list-docs", response_model=list[DocumentInfo])
def list_documents():
    return get_all_documents()

@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/delete-doc")
def delete_document(req: DeleteFileRequest):
    if delete_doc_from_chroma(req.file_id) and delete_document_record(req.file_id):
        return {"message": f"Deleted file_id {req.file_id}"}
    raise HTTPException(500, "Delete failed")

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})





