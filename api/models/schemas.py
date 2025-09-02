from pydantic import BaseModel, validator
from typing import List, Optional

VALID_SECTORS = [
    "Cyber Security",
    "Data & Analytics", 
    "Infrastructure & Network",
    "Communication & Collaboration",
    "Marketing & Customer Experience",
    "Other"
]

VALID_COUNTRIES = [
    "United States",
    "Israel",
    "Germany", 
    "United Kingdom",
    "France",
    "Other"
]

VALID_CURRENCIES = ["USD", "EUR", "GBP", "CAD", "Other"]

class FeatureInput(BaseModel):
    features: List[float]
    model: str = 'lightgbm'

class ChatRequest(BaseModel):
    message: str
    name: str = None
    preferred_model: str = None
    history: list = None

class AdvancedMetrics(BaseModel):
    """Advanced metrics that users can override in advanced mode."""
    sales_marketing: float = None
    ebitda: float = None
    cash_burn: float = None
    rule_of_40: float = None
    arr_yoy_growth: float = None
    revenue_yoy_growth: float = None
    magic_number: float = None
    burn_multiple: float = None
    customers_eop: float = None
    expansion_upsell: float = None
    churn_reduction: float = None
    gross_margin: float = None
    headcount: float = None
    net_profit_margin: float = None

class HistoricalARR(BaseModel):
    q1_arr: float = None  # 4 quarters ago
    q2_arr: float = None  # 3 quarters ago  
    q3_arr: float = None  # 2 quarters ago
    q4_arr: float = None  # 1 quarter ago (most recent)

class Tier2Metrics(BaseModel):
    """Tier 2 advanced metrics for enhanced predictions."""
    gross_margin: float = None
    sales_marketing: float = None
    cash_burn: float = None
    customers: float = None
    churn_rate: float = None
    expansion_rate: float = None

class TierBasedRequest(BaseModel):
    """Tier-based prediction request following the new model structure."""
    # Tier 1 (Required)
    company_name: str = None
    q1_arr: float  # Q1 2023 ARR
    q2_arr: float  # Q2 2023 ARR
    q3_arr: float  # Q3 2023 ARR
    q4_arr: float  # Q4 2023 ARR
    headcount: int
    sector: str
    
    # Tier 2 (Optional - Advanced Analysis)
    tier2_metrics: Optional[Tier2Metrics] = None
    
    @validator('sector')
    def validate_sector(cls, v):
        if v not in VALID_SECTORS:
            raise ValueError(f'Invalid sector. Must be one of: {", ".join(VALID_SECTORS)}')
        return v
    
    @validator('q1_arr', 'q2_arr', 'q3_arr', 'q4_arr')
    def validate_arr_values(cls, v):
        if v <= 0:
            raise ValueError('ARR values must be greater than 0')
        return v
    
    @validator('headcount')
    def validate_headcount(cls, v):
        if v <= 0:
            raise ValueError('Headcount must be greater than 0')
        return v

class GuidedInputRequest(BaseModel):
    company_name: str
    current_arr: float
    net_new_arr: float
    historical_arr: HistoricalARR = None  # Optional historical ARR data
    advanced_mode: bool = False
    advanced_metrics: AdvancedMetrics = None

class EnhancedGuidedInputRequest(BaseModel):
    # Basic inputs (required)
    company_name: str = None
    current_arr: float
    net_new_arr: float
    
    # Enhanced mode (optional)
    enhanced_mode: bool = False
    
    # Enhanced inputs (optional, only used if enhanced_mode=True)
    sector: str = None
    country: str = None
    currency: str = None
    
    # Historical data (optional)
    historical_arr: HistoricalARR = None
    
    # Advanced metrics (optional)
    advanced_mode: bool = False
    advanced_metrics: AdvancedMetrics = None
    
    @validator('sector')
    def validate_sector(cls, v, values):
        if values.get('enhanced_mode') and v is not None:
            if v not in VALID_SECTORS:
                raise ValueError(f'Invalid sector. Must be one of: {", ".join(VALID_SECTORS)}')
        return v
    
    @validator('country')
    def validate_country(cls, v, values):
        if values.get('enhanced_mode') and v is not None:
            if v not in VALID_COUNTRIES:
                raise ValueError(f'Invalid country. Must be one of: {", ".join(VALID_COUNTRIES)}')
        return v
    
    @validator('currency')
    def validate_currency(cls, v, values):
        if values.get('enhanced_mode') and v is not None:
            if v not in VALID_CURRENCIES:
                raise ValueError(f'Invalid currency. Must be one of: {", ".join(VALID_CURRENCIES)}')
        return v
