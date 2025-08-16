# routes_upload.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List, Dict, Any
import os, uuid, shutil, logging

from chroma_utils import (
    index_document_to_chroma, 
    UploadLimits,
    validate_document_batch,
    validate_single_document,
    format_file_size,
)
from db_utils import insert_document_record, delete_document_record


# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()
MAX_FILES_PER_REQUEST = 20
MAX_PAGES_PER_DOCUMENT = 1000
ALLOWED_EXTS = {".pdf", ".docx", ".html", ".txt"}

# Initialize upload limits configuration
upload_limits = UploadLimits(
    max_documents=MAX_FILES_PER_REQUEST,
    max_pages_per_doc=MAX_PAGES_PER_DOCUMENT,
    pdf_max_size=50 * 1024 * 1024,    # 50 MB
    docx_max_size=25 * 1024 * 1024,   # 25 MB
    txt_max_size=10 * 1024 * 1024,    # 10 MB
    html_max_size=15 * 1024 * 1024    # 15 MB
)

def save_uploaded_file(uploaded_file: UploadFile) -> str:
    """Save uploaded file to temporary location and return path"""
    ext = os.path.splitext(uploaded_file.filename or "")[1].lower()
    tmp_path = f"tmp_{uuid.uuid4()}{ext}"
    
    try:
        with open(tmp_path, "wb") as buffer:
            shutil.copyfileobj(uploaded_file.file, buffer)
        
        logger.info(f"✓ Saved uploaded file: {uploaded_file.filename} -> {tmp_path}")
        return tmp_path
        
    except Exception as e:
        # Clean up partial file if it exists
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except:
                pass
        raise Exception(f"Failed to save uploaded file {uploaded_file.filename}: {str(e)}")

def cleanup_temp_files(file_paths: List[str]) -> None:
    """Clean up temporary files safely"""
    for path in file_paths:
        try:
            if os.path.exists(path):
                os.remove(path)
                logger.debug(f"🗑️ Cleaned up temp file: {path}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to remove temp file {path}: {e}")

def validate_file_extensions(files: List[UploadFile]) -> List[Dict[str, Any]]:
    """Validate file extensions and return list of invalid files"""
    invalid_files = []
    
    for file in files:
        if not file.filename:
            invalid_files.append({
                "file": "unnamed_file",
                "error": "Missing filename"
            })
            continue
            
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ALLOWED_EXTS:
            invalid_files.append({
                "file": file.filename,
                "error": f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTS)}"
            })
    
    return invalid_files

@router.post("/upload-docs")
async def upload_docs(files: List[UploadFile] = File(...)):
    """
    Upload multiple documents with validation for:
    - Maximum 20 documents per request
    - Maximum 1000 pages per document
    - File size limits by type
    - Supported file formats: PDF, DOCX, HTML, TXT
    """
    
    # Early validation: Check file count
    if len(files) > MAX_FILES_PER_REQUEST:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Too many files",
                "message": f"Upload limit is {MAX_FILES_PER_REQUEST} files. Received {len(files)} files.",
                "limits": {
                    "max_documents": MAX_FILES_PER_REQUEST,
                    "max_pages_per_document": MAX_PAGES_PER_DOCUMENT
                }
            }
        )
    
    # Validate file extensions first (before saving files)
    extension_errors = validate_file_extensions(files)
    if extension_errors:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid file types",
                "invalid_files": extension_errors,
                "allowed_extensions": list(ALLOWED_EXTS)
            }
        )
    
    # Initialize tracking variables
    tmp_file_paths = []
    file_name_mapping = {}
    uploaded = []
    failed = []
    
    try:
        logger.info(f"📤 Processing {len(files)} uploaded files...")
        
        # Step 1: Save all uploaded files to temporary locations
        for uploaded_file in files:
            try:
                tmp_path = save_uploaded_file(uploaded_file)
                tmp_file_paths.append(tmp_path)
                file_name_mapping[tmp_path] = uploaded_file.filename
                
            except Exception as e:
                logger.error(f"❌ Failed to save {uploaded_file.filename}: {e}")
                failed.append({
                    "file": uploaded_file.filename or "unnamed_file",
                    "error": f"File save error: {str(e)}"
                })
        
        # Step 2: Batch validation for all saved files
        if tmp_file_paths:
            logger.info(f"🔍 Validating {len(tmp_file_paths)} files for size and page limits...")
            
            validation_result = validate_document_batch(tmp_file_paths, upload_limits)
            
            if not validation_result["valid"]:
                logger.error(f"❌ Batch validation failed: {validation_result['errors']}")
                
                # Add detailed validation errors
                validation_details = []
                for doc in validation_result["documents"]:
                    if not doc["valid"]:
                        filename = file_name_mapping.get(doc["file_path"], "unknown")
                        validation_details.append({
                            "file": filename,
                            "errors": doc["errors"],
                            "file_size": format_file_size(doc["file_size"]) if doc["file_size"] > 0 else "unknown",
                            "page_count": doc["page_count"]
                        })
                
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "Document validation failed",
                        "validation_errors": validation_result["errors"],
                        "invalid_documents": validation_details,
                        "limits": {
                            "max_documents": upload_limits.max_documents,
                            "max_pages_per_document": upload_limits.max_pages_per_doc,
                            "file_size_limits": {
                                "PDF": format_file_size(upload_limits.pdf_max_size),
                                "DOCX": format_file_size(upload_limits.docx_max_size),
                                "TXT": format_file_size(upload_limits.txt_max_size),
                                "HTML": format_file_size(upload_limits.html_max_size)
                            }
                        }
                    }
                )
            
            logger.info(f"✅ Batch validation passed:")
            logger.info(f"  📄 Total documents: {validation_result['summary']['total_documents']}")
            logger.info(f"  📊 Total pages: {validation_result['summary']['total_pages']}")
            logger.info(f"  💾 Total size: {validation_result['summary']['total_size_formatted']}")
        
        # Step 3: Process each validated file
        for tmp_path in tmp_file_paths:
            filename = file_name_mapping.get(tmp_path, "unknown")
            file_id = None
            
            try:
                logger.info(f"🔄 Processing: {filename}")
                
                # Insert database record
                file_id = insert_document_record(filename)
                logger.info(f"📝 Created database record with file_id: {file_id}")
                
                # Index document to Chroma
                success = index_document_to_chroma(tmp_path, file_id)
                
                if success:
                    # Get document info for response
                    doc_info = validate_single_document(tmp_path, upload_limits)
                    
                    uploaded.append({
                        "file": filename,
                        "file_id": file_id,
                        "status": "success",
                        "page_count": doc_info["page_count"],
                        "file_size": format_file_size(doc_info["file_size"])
                    })
                    logger.info(f"✅ Successfully processed: {filename} (file_id: {file_id})")
                    
                else:
                    # Clean up database record on indexing failure
                    if file_id is not None:
                        delete_document_record(file_id)
                        logger.warning(f"🗑️ Cleaned up database record for failed indexing: {file_id}")
                    
                    failed.append({
                        "file": filename,
                        "error": "Document indexing failed",
                        "file_id": file_id
                    })
                    logger.error(f"❌ Indexing failed: {filename}")
            
            except Exception as exc:
                logger.exception(f"💥 Error processing {filename}")
                
                # Clean up database record on any error
                if file_id is not None:
                    try:
                        delete_document_record(file_id)
                        logger.warning(f"🗑️ Cleaned up database record after error: {file_id}")
                    except:
                        logger.error(f"💥 Failed to clean up database record: {file_id}")
                
                failed.append({
                    "file": filename,
                    "error": str(exc),
                    "file_id": file_id
                })
    
    finally:
        # Always clean up temporary files
        if tmp_file_paths:
            logger.info(f"🗑️ Cleaning up {len(tmp_file_paths)} temporary files...")
            cleanup_temp_files(tmp_file_paths)
    
    # Prepare response
    response = {
        "uploaded": uploaded,
        "failed": failed,
        "summary": {
            "total_files": len(files),
            "successful": len(uploaded),
            "failed": len(failed),
            "success_rate": f"{(len(uploaded) / len(files)) * 100:.1f}%" if files else "0%"
        }
    }
    
    # Log final summary
    logger.info(f"📈 Upload summary:")
    logger.info(f"  📤 Total files: {response['summary']['total_files']}")
    logger.info(f"  ✅ Successful: {response['summary']['successful']}")
    logger.info(f"  ❌ Failed: {response['summary']['failed']}")
    logger.info(f"  📊 Success rate: {response['summary']['success_rate']}")
    
    # Handle all files failed case
    if not uploaded and failed:
        raise HTTPException(
            status_code=500, 
            detail={
                "error": "All files failed to process",
                "failed_files": failed,
                "summary": response["summary"]
            }
        )
    
    return response

@router.get("/upload-limits")
async def get_upload_limits():
    """Get current upload limits and configuration"""
    return {
        "limits": {
            "max_documents": upload_limits.max_documents,
            "max_pages_per_document": upload_limits.max_pages_per_doc,
            "file_size_limits": {
                "PDF": format_file_size(upload_limits.pdf_max_size),
                "DOCX": format_file_size(upload_limits.docx_max_size),
                "TXT": format_file_size(upload_limits.txt_max_size),
                "HTML": format_file_size(upload_limits.html_max_size)
            }
        },
        "allowed_extensions": list(ALLOWED_EXTS),
        "validation_rules": [
            f"Maximum {upload_limits.max_documents} documents per upload",
            f"Maximum {upload_limits.max_pages_per_doc} pages per document",
            "Supported formats: PDF, DOCX, TXT, HTML",
            "File size limits vary by type (see file_size_limits)"
        ]
    }

@router.post("/validate-files")
async def validate_files_endpoint(files: List[UploadFile] = File(...)):
    """Validate files without uploading them (dry run)"""
    
    # Early validation
    if len(files) > MAX_FILES_PER_REQUEST:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files: {len(files)} exceeds limit of {MAX_FILES_PER_REQUEST}"
        )
    
    # Extension validation
    extension_errors = validate_file_extensions(files)
    if extension_errors:
        return {
            "valid": False,
            "errors": extension_errors,
            "message": "File extension validation failed"
        }
    
    # Save files temporarily for validation
    tmp_file_paths = []
    file_name_mapping = {}
    
    try:
        for uploaded_file in files:
            tmp_path = save_uploaded_file(uploaded_file)
            tmp_file_paths.append(tmp_path)
            file_name_mapping[tmp_path] = uploaded_file.filename
        
        # Run batch validation
        validation_result = validate_document_batch(tmp_file_paths, upload_limits)
        
        # Enhance response with readable information
        for doc in validation_result["documents"]:
            filename = file_name_mapping.get(doc["file_path"], "unknown")
            doc["filename"] = filename
            doc["file_size_formatted"] = format_file_size(doc["file_size"]) if doc["file_size"] > 0 else "unknown"
        
        return {
            "valid": validation_result["valid"],
            "errors": validation_result["errors"],
            "documents": validation_result["documents"],
            "summary": validation_result["summary"],
            "message": "Validation complete" if validation_result["valid"] else "Validation failed"
        }
    
    finally:
        cleanup_temp_files(tmp_file_paths)
