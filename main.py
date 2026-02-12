import httpx
from bs4 import BeautifulSoup
from typing import List
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, HttpUrl
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()


# --- 1. Fixed Niches ---
ALLOWED_NICHES = [
    "Business", "Digital Marketing & Advertising", "Technology / Gadgets & Technology",
    "SaaS", "Ecommerce", "Finance", "Health", "Lifestyle", "Education", 
    "Real Estate", "Casino", "CBD", "Blockchain & Cryptocurrency", 
    "Gaming", "Pharmacy", "Banking", "Energy", "Manufacturing & Industry", "Science"
]

class CategoryResponse(BaseModel):
    """Returns a list of categories strictly from the allowed niches."""
    categories: List[str] = Field(
        description="A list of one or more categories that apply to the website."
    )

class UrlInput(BaseModel):
    url: HttpUrl

# --- 2. Async Content Extractor ---
async def extract_website_text_async(url: str) -> str:
    # We use a context manager for the client to handle connections efficiently
    async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            response = await client.get(url, headers=headers)
            
            if response.status_code == 403:
                raise HTTPException(status_code=403, detail="Access denied by website (Cloudflare/Bot protection).")
            
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            for element in soup(["script", "style", "nav", "footer", "header"]):
                element.decompose()
                
            text = soup.get_text(separator=" ", strip=True)
            if not text:
                raise ValueError("Website returned no readable text.")
                
            return text[:5000] # Increased limit slightly
            
        except httpx.RequestError as exc:
            raise HTTPException(status_code=502, detail=f"Network error: {str(exc)}")

# --- 3. FastAPI App ---
app = FastAPI(title="Website Categorizer")

@app.post("/api/v1/categorize", response_model=CategoryResponse)
async def categorize_website(input_data: UrlInput):
    # 1. Scraping (Asynchronous)
    text_content = await extract_website_text_async(str(input_data.url))
    
    # 2. Setup LLM (Using ainvoke for non-blocking calls)
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0) 
        structured_llm = llm.with_structured_output(CategoryResponse)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are a website classification expert. 
            Your task is to analyze the text and assign categories.
            
            STRICT RULE: You must ONLY choose from the following list of categories:
            {', '.join(ALLOWED_NICHES)}
            
            If multiple categories apply, include them all in the list. 
            If only one applies, return a list with that single item."""),
            ("human", "WebsiteContent: {content}")
        ])
        
        chain = prompt | structured_llm
        
        # .ainvoke is the async version of .invoke
        result = await chain.ainvoke({"content": text_content})
        
        # Final validation: Ensure the LLM didn't hallucinate a new category
        valid_categories = [c for c in result.categories if c in ALLOWED_NICHES]
        return CategoryResponse(categories=valid_categories or ["Business"]) # Default to Business if all fail

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="LLM failed to process the request."
        )