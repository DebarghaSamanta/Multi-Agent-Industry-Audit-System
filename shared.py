from typing import TypedDict, Annotated, List
import operator

class ResearchState(TypedDict):
    topic: str
    # 'operator.add' allows agents to append findings without deleting old ones
    search_queries: Annotated[List[str], operator.add] 
    raw_data: Annotated[List[str], operator.add]
    plan: List[str]
    draft_report: str
    critique: str
    is_verified: bool
    iterations: int