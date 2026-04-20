import json
import os
from typing import List, Dict, Any

from pydantic import BaseModel, Field
from google import genai
from google.genai import types

class OperationalRecommendation(BaseModel):
    action: str = Field(description="The specific operational action to execute.")
    target_hub: str = Field(description="The hub or facility affected by the action.")
    expected_impact: str = Field(description="The expected quantitative or qualitative impact of this action.")

class ExecutiveBriefResponse(BaseModel):
    executive_brief: str = Field(
        description="A structured exactly 3-paragraph executive brief in Markdown format analyzing network health, GNN clusters, and strategy."
    )
    operational_recommendations: List[OperationalRecommendation] = Field(
        description="List of strict operational recommendations pulled from the brief for validation."
    )

class InsightAgent:
    """
    Insight Agent responsible for reasoning over Graph topology and bottlenecks.
    Uses Gemini API to ingest graph data and output structured executive briefs.
    """
    
    def __init__(self, api_key: str = None, model_name: str = "gemini-2.5-flash"):
        """
        Initializes the Insight Agent using the latest Google GenAI SDK.
        
        Args:
            api_key: Gemini API key. Defaults to GEMINI_API_KEY env var if None.
            model_name: The Gemini model to use for generation (defaults to gemini-2.5-flash).
        """
        if api_key is None:
            api_key = os.environ.get("GEMINI_API_KEY")
            
        if not api_key:
            raise ValueError("Gemini API key is required. Set GEMINI_API_KEY env var or pass it explicitly.")
            
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        
        self.system_instruction = (
            "You are a Senior Logistics Strategist specializing in graph theory, network topology, "
            "and supply chain optimization. Your role is to interpret complex Graph Neural Network (GNN) embeddings, "
            "cluster assignments, and network centrality metrics (like betweenness and eigenvector centrality).\n"
            "You must synthesize raw JSON representations of network bottlenecks and node clusters into actionable, "
            "data-driven intelligence. Focus on identifying *why* a hub is a bottleneck based on its network position, "
            "and suggest operational changes (e.g., shifting specific volumes, routing adjustments) to relieve congestion."
        )

    def generate_brief(self, bottlenecks: List[Dict[str, Any]], hub_clusters: List[Dict[str, Any]]) -> ExecutiveBriefResponse:
        """
        Generates a structured executive brief from bottleneck and cluster data.
        
        Args:
            bottlenecks: JSON-serialized list of bottleneck nodes and metrics.
            hub_clusters: JSON-serialized list of hub clusters and GNN features.
            
        Returns:
            ExecutiveBriefResponse validated by pydantic containing the markdown brief and recommendations.
        """
        prompt = (
            "Analyze the following graph-based logistics network data to generate an Executive Brief.\n\n"
            f"=== Bottleneck Data ===\n{json.dumps(bottlenecks, indent=2)}\n\n"
            f"=== Cluster Data ===\n{json.dumps(hub_clusters, indent=2)}\n\n"
            "Task Requirements:\n"
            "1. Output a structured 3-paragraph executive brief in Markdown format in the 'executive_brief' field.\n"
            "   - Paragraph 1: Executive Summary of network health and critical bottlenecks.\n"
            "   - Paragraph 2: Detailed Graph Analysis (interpreting centrality and GNN cluster logic).\n"
            "   - Paragraph 3: Actionable Strategy (resource allocation and routing adjustments).\n"
            "2. Extract strict operational recommendations into the 'operational_recommendations' array."
        )
        
        # Using Gemini structured output feature to guarantee schema compliance
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ExecutiveBriefResponse,
                system_instruction=self.system_instruction,
            ),
        )
        
        # The new SDK parses it directly into the schema automatically via response.parsed
        if response.parsed:
            return response.parsed
        else:
            # Fallback to manual validation from response.text if parsed is None
            return ExecutiveBriefResponse.model_validate_json(response.text)
