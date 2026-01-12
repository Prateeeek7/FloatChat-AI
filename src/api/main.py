"""FastAPI backend for FloatChat - optimized for 16GB RAM."""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from llm.rag_system import ARGORAGSystem
from database.models import DatabaseManager

# Initialize FastAPI app
app = FastAPI(
    title="FloatChat API",
    description="AI-powered ARGO ocean data analysis API",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for caching
rag_system = None
db_manager = None

@app.on_event("startup")
async def startup_event():
    """Initialize systems on startup."""
    global rag_system, db_manager
    try:
        rag_system = ARGORAGSystem()
        db_manager = DatabaseManager()
        print("FloatChat API initialized successfully")
    except Exception as e:
        print(f"Error initializing API: {str(e)}")

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    limit: Optional[int] = 100

class QueryResponse(BaseModel):
    query: str
    sql_query: str
    parameters: Dict[str, Any]
    results: List[Dict[str, Any]]
    response: str
    num_results: int

class DatabaseStats(BaseModel):
    floats: int
    profiles: int
    trajectories: int
    bgc_records: int
    memory_usage_mb: float

# API endpoints
@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "FloatChat API - ARGO Ocean Data Analysis"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "rag_system": rag_system is not None, "db_manager": db_manager is not None}

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a natural language query."""
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    try:
        result = rag_system.process_natural_language_query(request.query)
        
        # Limit results if requested
        if request.limit and len(result['results']) > request.limit:
            result['results'] = result['results'][:request.limit]
            result['num_results'] = len(result['results'])
        
        return QueryResponse(**result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/stats", response_model=DatabaseStats)
async def get_database_stats():
    """Get database statistics."""
    if not db_manager:
        raise HTTPException(status_code=500, detail="Database manager not initialized")
    
    try:
        session = db_manager.get_session()
        
        # Get counts
        float_count = session.query(db_manager.ARGOFloat).count()
        profile_count = session.query(db_manager.ARGOProfile).count()
        trajectory_count = session.query(db_manager.ARGOTrajectory).count()
        bgc_count = session.query(db_manager.ARGOBGCData).count()
        
        # Get memory usage
        memory_info = db_manager.get_memory_usage()
        
        session.close()
        
        return DatabaseStats(
            floats=float_count,
            profiles=profile_count,
            trajectories=trajectory_count,
            bgc_records=bgc_count,
            memory_usage_mb=memory_info['total_size_mb']
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

@app.get("/floats")
async def get_floats(region: Optional[str] = Query(None), limit: int = Query(100)):
    """Get ARGO float information."""
    if not db_manager:
        raise HTTPException(status_code=500, detail="Database manager not initialized")
    
    try:
        session = db_manager.get_session()
        
        query = session.query(db_manager.ARGOFloat)
        if region:
            query = query.filter(db_manager.ARGOFloat.region == region)
        
        floats = query.limit(limit).all()
        
        result = []
        for float_obj in floats:
            result.append({
                "platform_number": float_obj.platform_number,
                "region": float_obj.region,
                "status": float_obj.status,
                "total_profiles": float_obj.total_profiles,
                "first_deployment_date": float_obj.first_deployment_date.isoformat() if float_obj.first_deployment_date else None,
                "last_profile_date": float_obj.last_profile_date.isoformat() if float_obj.last_profile_date else None
            })
        
        session.close()
        return {"floats": result, "count": len(result)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting floats: {str(e)}")

@app.get("/profiles")
async def get_profiles(
    platform_number: Optional[str] = Query(None),
    min_lat: Optional[float] = Query(None),
    max_lat: Optional[float] = Query(None),
    min_lon: Optional[float] = Query(None),
    max_lon: Optional[float] = Query(None),
    limit: int = Query(100)
):
    """Get ARGO profile data with optional filters."""
    if not db_manager:
        raise HTTPException(status_code=500, detail="Database manager not initialized")
    
    try:
        session = db_manager.get_session()
        
        query = session.query(db_manager.ARGOProfile)
        
        if platform_number:
            query = query.filter(db_manager.ARGOProfile.platform_number == platform_number)
        
        if min_lat is not None:
            query = query.filter(db_manager.ARGOProfile.latitude >= min_lat)
        if max_lat is not None:
            query = query.filter(db_manager.ARGOProfile.latitude <= max_lat)
        if min_lon is not None:
            query = query.filter(db_manager.ARGOProfile.longitude >= min_lon)
        if max_lon is not None:
            query = query.filter(db_manager.ARGOProfile.longitude <= max_lon)
        
        profiles = query.limit(limit).all()
        
        result = []
        for profile in profiles:
            result.append({
                "platform_number": profile.platform_number,
                "cycle_number": profile.cycle_number,
                "latitude": profile.latitude,
                "longitude": profile.longitude,
                "profile_date": profile.profile_date.isoformat() if profile.profile_date else None,
                "max_depth": profile.max_depth,
                "data_quality": profile.data_quality
            })
        
        session.close()
        return {"profiles": result, "count": len(result)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting profiles: {str(e)}")

@app.get("/trajectories")
async def get_trajectories(
    platform_number: Optional[str] = Query(None),
    min_lat: Optional[float] = Query(None),
    max_lat: Optional[float] = Query(None),
    min_lon: Optional[float] = Query(None),
    max_lon: Optional[float] = Query(None),
    limit: int = Query(1000)
):
    """Get ARGO trajectory data with optional filters."""
    if not db_manager:
        raise HTTPException(status_code=500, detail="Database manager not initialized")
    
    try:
        session = db_manager.get_session()
        
        query = session.query(db_manager.ARGOTrajectory)
        
        if platform_number:
            query = query.filter(db_manager.ARGOTrajectory.platform_number == platform_number)
        
        if min_lat is not None:
            query = query.filter(db_manager.ARGOTrajectory.latitude >= min_lat)
        if max_lat is not None:
            query = query.filter(db_manager.ARGOTrajectory.latitude <= max_lat)
        if min_lon is not None:
            query = query.filter(db_manager.ARGOTrajectory.longitude >= min_lon)
        if max_lon is not None:
            query = query.filter(db_manager.ARGOTrajectory.longitude <= max_lon)
        
        trajectories = query.limit(limit).all()
        
        result = []
        for traj in trajectories:
            result.append({
                "platform_number": traj.platform_number,
                "latitude": traj.latitude,
                "longitude": traj.longitude,
                "position_date": traj.position_date.isoformat() if traj.position_date else None,
                "direction": traj.direction,
                "speed": traj.speed
            })
        
        session.close()
        return {"trajectories": result, "count": len(result)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting trajectories: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)





