# routes_metadata.py
from fastapi import APIRouter, HTTPException
import logging

from chroma_utils import (
    list_all_documents,
    get_document_info,
    get_vectorstore_stats
)

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/documents/metadata", tags=["documents"], summary="Get all documents metadata")
async def get_all_documents_metadata():
    """
    Get metadata for all processed documents in the vector store.
    
    Returns:
        - List of all documents with their metadata
        - Total count and summary statistics
    """
    try:
        logger.info("📋 Getting all documents metadata")
        
        # Get all documents
        documents = list_all_documents()
        
        # Get vector store stats
        stats = get_vectorstore_stats()
        
        return {
            "status": "success",
            "total_documents": len(documents),
            "documents": documents,
            "vectorstore_stats": stats
        }
        
    except Exception as e:
        logger.error(f"❌ Error retrieving all documents metadata: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve documents metadata: {str(e)}"
        )

@router.get("/documents/{file_id}/metadata", tags=["documents"], summary="Get single document metadata")
async def get_single_document_metadata(file_id: int):
    """
    Get detailed metadata for a specific document by file_id.
    
    Args:
        file_id: The unique identifier of the document
        
    Returns:
        - Document existence status
        - Chunk count and metadata details
        - Source file information
    """
    try:
        logger.info(f"📄 Getting metadata for file_id {file_id}")
        
        # Get document info
        doc_info = get_document_info(file_id)
        
        if not doc_info.get("exists", False):
            raise HTTPException(
                status_code=404,
                detail=f"Document with file_id {file_id} not found"
            )
        
        return {
            "status": "success",
            "file_id": file_id,
            "document_info": doc_info
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error retrieving metadata for file_id {file_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve document metadata: {str(e)}"
        )
