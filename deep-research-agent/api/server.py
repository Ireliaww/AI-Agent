"""
FastAPI Server for Multi-Agent System

提供HTTP API接口，支持云端部署到Google Cloud Run
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import asyncio
import os

# 延迟导入agents以避免启动时的问题
app = FastAPI(
    title="Multi-Agent AI System",
    description="Intelligent multi-agent system for research and coding powered by Gemini",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该限制来源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量存储agents
coordinator = None
research_agent = None
coding_agent = None
gemini_client = None


class QueryRequest(BaseModel):
    """查询请求模型"""
    query: str = Field(..., description="用户查询内容", min_length=1)
    agent: str = Field(
        default="auto",
        description="指定agent类型: auto, research, coding, coordinator",
        pattern="^(auto|research|coding|coordinator)$"
    )
    timeout: int = Field(default=120, description="超时时间（秒）", ge=10, le=300)


class QueryResponse(BaseModel):
    """查询响应模型"""
    result: str = Field(..., description="处理结果（Markdown格式）")
    agent_used: str = Field(..., description="实际使用的agent")
    execution_time: Optional[float] = Field(None, description="执行时间（秒）")


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    agents_available: list[str]
    google_api_configured: bool


@app.on_event("startup")
async def startup_event():
    """应用启动时初始化agents"""
    global coordinator, research_agent, coding_agent, gemini_client
    
    try:
        # 检查环境变量
        if not os.getenv("GOOGLE_API_KEY"):
            print("WARNING: GOOGLE_API_KEY not set. API functionality will be limited.")
            return
        
        # 导入并初始化agents
        from src.client import GeminiClient
        from src.agents.researcher import ResearchAgent
        from src.agents.coder import CodingAgent
        from agents.coordinator import CoordinatorAgent
        
        gemini_client = GeminiClient()
        research_agent = ResearchAgent(gemini_client=gemini_client)
        coding_agent = CodingAgent(gemini_client=gemini_client)
        coordinator = CoordinatorAgent(
            research_agent=research_agent,
            coding_agent=coding_agent,
            gemini_client=gemini_client
        )
        
        print("✓ All agents initialized successfully")
        
    except Exception as e:
        print(f"ERROR initializing agents: {e}")
        # 不要抛出异常，让服务器继续运行以便调试


@app.get("/", response_model=dict)
async def root():
    """根路径 - API信息"""
    return {
        "message": "Multi-Agent AI System API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查端点"""
    agents_available = []
    
    if research_agent is not None:
        agents_available.append("research")
    if coding_agent is not None:
        agents_available.append("coding")
    if coordinator is not None:
        agents_available.append("coordinator")
    
    return HealthResponse(
        status="healthy",
        agents_available=agents_available,
        google_api_configured=os.getenv("GOOGLE_API_KEY") is not None
    )


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    处理用户查询
    
    支持的agent类型：
    - auto: 自动选择最合适的agent（推荐）
    - research: 仅使用研究agent
    - coding: 仅使用编程agent
    - coordinator: 使用协调器（支持多agent协作）
    """
    if coordinator is None:
        raise HTTPException(
            status_code=503,
            detail="Agents not initialized. Please check GOOGLE_API_KEY environment variable."
        )
    
    try:
        import time
        start_time = time.time()
        
        if request.agent == "research":
            if research_agent is None:
                raise HTTPException(503, "Research agent not available")
            result_obj = await research_agent.research(request.query)
            result = result_obj.report
            agent_used = "research"
            
        elif request.agent == "coding":
            if coding_agent is None:
                raise HTTPException(503, "Coding agent not available")
            result = await coding_agent.process(request.query)
            agent_used = "coding"
            
        elif request.agent == "coordinator" or request.agent == "auto":
            result = await coordinator.process_request(request.query)
            agent_used = "coordinator"
            
        else:
            raise HTTPException(
                400, 
                f"Invalid agent type: {request.agent}. Use: auto, research, coding, or coordinator"
            )
        
        execution_time = time.time() - start_time
        
        return QueryResponse(
            result=result,
            agent_used=agent_used,
            execution_time=round(execution_time, 2)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )


@app.get("/agents")
async def list_agents():
    """列出可用的agents及其状态"""
    return {
        "research_agent": {
            "available": research_agent is not None,
            "description": "Deep research and information gathering"
        },
        "coding_agent": {
            "available": coding_agent is not None,
            "description": "Code generation and problem solving"
        },
        "coordinator": {
            "available": coordinator is not None,
            "description": "Multi-agent orchestration and intelligent routing"
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    # 从环境变量获取端口（Cloud Run会设置PORT）
    port = int(os.getenv("PORT", 8080))
    
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=port,
        reload=False  # 生产环境不要reload
    )
