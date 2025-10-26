import os
import json
import requests
from datetime import datetime
from uuid import uuid4
from dotenv import load_dotenv

from uagents import Agent, Protocol, Context, Model
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    TextContent,
    chat_protocol_spec,
)

# Load environment variables
load_dotenv()

# Set your API keys
ASI_ONE_API_KEY = os.environ.get("ASI_ONE_API_KEY")
EXA_API_KEY = os.environ.get("EXA_API_KEY")
AGENTVERSE_API_KEY = os.environ.get("AGENTVERSE_API_KEY")

if not ASI_ONE_API_KEY:
    raise ValueError("Please set ASI_ONE_API_KEY environment variable")
if not EXA_API_KEY:
    raise ValueError("Please set EXA_API_KEY environment variable")
if not AGENTVERSE_API_KEY:
    raise ValueError("Please set AGENTVERSE_API_KEY environment variable")

# REST API Models
class BrandResearchRequest(Model):
    brand_name: str

class BrandResearchResponse(Model):
    success: bool
    brand_name: str
    research_result: str
    timestamp: str
    agent_address: str

# ASI:One API configuration
ASI_BASE_URL = "https://api.asi1.ai/v1"
ASI_HEADERS = {
    "Authorization": f"Bearer {ASI_ONE_API_KEY}",
    "Content-Type": "application/json"
}

class WebSearchAgent:
    def __init__(self):
        self.exa_api_key = EXA_API_KEY
        
    def exa_search(self, query: str, num_results: int = 10) -> dict:
        """Perform web search using Exa API"""
        try:
            print(f"🔍 Starting Exa search for query: '{query}'")
            
            # Construct the search request
            search_payload = {
                "query": query,
                "type": "auto",  # Intelligently combines neural and keyword search
                "numResults": num_results,
                "contents": {
                    "text": {
                        "maxCharacters": 2000,
                        "includeHtmlTags": False
                    },
                    "highlights": {
                        "numSentences": 3,
                        "highlightsPerUrl": 2,
                        "query": query
                    },
                    "summary": {
                        "query": f"Summarize key information about {query}"
                    }
                }
            }
            
            print("📤 Sending search request to Exa API...")
            response = requests.post(
                "https://api.exa.ai/search",
                json=search_payload,
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": self.exa_api_key
                }
            )
            
            print(f"📥 Exa API response status: {response.status_code}")
            
            if response.status_code == 200:
                search_data = response.json()
                print(f"✅ Search completed successfully!")
                print(f"📊 Found {len(search_data.get('results', []))} results")
                
                return {
                    "success": True,
                    "results": search_data.get("results", []),
                    "request_id": search_data.get("requestId"),
                    "search_type": search_data.get("resolvedSearchType"),
                    "context": search_data.get("context"),
                    "cost": search_data.get("costDollars")
                }
            else:
                print(f"❌ Exa API error: {response.status_code} - {response.text}")
                return {
                    "success": False,
                    "error": f"Exa API error: {response.status_code} - {response.text}"
                }
                
        except Exception as e:
            print(f"❌ Search failed with exception: {str(e)}")
            return {
                "success": False,
                "error": f"Search failed: {str(e)}"
            }

    def create_search_tool_schema(self):
        """Define the Exa search tool schema for ASI:One"""
        return {
            "type": "function",
            "function": {
                "name": "exa_search",
                "description": "Perform comprehensive web search using Exa AI to find the most recent and relevant information. Use this tool when the query requires current information, specific entity research, or comprehensive analysis. Returns search results with titles, URLs, text content, highlights, and summaries.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to find relevant web pages"
                        },
                        "num_results": {
                            "type": "integer",
                            "description": "Number of search results to return (1-100)",
                            "default": 10
                        }
                    },
                    "required": ["query"],
                    "additionalProperties": False
                },
                "strict": True
            }
        }

    def format_search_results(self, search_data: dict) -> str:
        """Format search results into a readable string"""
        if not search_data.get("success"):
            return f"Search failed: {search_data.get('error', 'Unknown error')}"
        
        results = search_data.get("results", [])
        if not results:
            return "No search results found."
        
        formatted = f"Found {len(results)} search results:\n\n"
        
        for i, result in enumerate(results, 1):
            formatted += f"Result {i}:\n"
            formatted += f"Title: {result.get('title', 'No title')}\n"
            formatted += f"URL: {result.get('url', 'No URL')}\n"
            
            if result.get('author'):
                formatted += f"Author: {result.get('author')}\n"
            
            if result.get('publishedDate'):
                formatted += f"Published: {result.get('publishedDate')}\n"
            
            if result.get('summary'):
                formatted += f"Summary: {result.get('summary')}\n"
            
            if result.get('highlights'):
                formatted += f"Highlights:\n"
                for highlight in result.get('highlights', []):
                    formatted += f"  - {highlight}\n"
            
            if result.get('text'):
                text_preview = result.get('text')[:300] + "..." if len(result.get('text', '')) > 300 else result.get('text')
                formatted += f"Text Preview: {text_preview}\n"
            
            formatted += "\n---\n\n"
        
        return formatted

    def process_search_query(self, user_query: str) -> str:
        """Process user query using ASI:One with Exa search tool"""
        try:
            search_tool = self.create_search_tool_schema()
            
            system_prompt = """You are a comprehensive research assistant with access to real-time web search through the exa_search tool.

Your task is to intelligently decide when to use the exa_search tool based on the query nature.

DECISION CRITERIA FOR TOOL USAGE:
1. CURRENT INFORMATION NEEDED: Does the query require recent, up-to-date information?
2. SPECIFIC ENTITIES: Is it about specific brands, companies, products, people, or events?
3. RESEARCH DEPTH: Does the user want comprehensive analysis, news, or market trends?
4. TIME-SENSITIVE: Are they asking about "latest", "recent", "current", or "what's happening"?
5. FACTUAL VERIFICATION: Do they need verified, sourced information?

USE THE TOOL WHEN:
- Querying about specific brands, companies, or products
- Asking for recent news, developments, or current events
- Requesting comprehensive research or analysis
- Seeking information about controversies, market trends, or reputation
- Asking "what's happening with", "latest news about", or "current status of"

DO NOT USE THE TOOL WHEN:
- Asking general knowledge questions that don't require current information
- Requesting explanations of concepts, definitions, or how-to guides
- Asking for personal advice or opinions

After using the tool, provide comprehensive analysis with proper citations and sources."""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ]

            payload = {
                "model": "asi1-extended",
                "messages": messages,
                "tools": [search_tool],
                "tool_choice": "auto",
                "temperature": 0.3
            }

            print(f"📤 Making ASI:One request with tool_choice: auto")
            
            response = requests.post(
                f"{ASI_BASE_URL}/chat/completions",
                headers=ASI_HEADERS,
                json=payload
            )

            if response.status_code != 200:
                return f"ASI:One API error: {response.status_code} - {response.text}"

            response_data = response.json()
            
            if "choices" not in response_data or not response_data["choices"]:
                return "No response received from ASI:One"

            choice = response_data["choices"][0]["message"]
            
            # Check if the model wants to call a tool
            if "tool_calls" in choice and choice["tool_calls"]:
                print("🔧 Tool calls detected!")
                
                messages.append({
                    "role": "assistant",
                    "content": choice.get("content", ""),
                    "tool_calls": choice["tool_calls"]
                })
                
                for tool_call in choice["tool_calls"]:
                    print(f"🔧 Processing tool call: {tool_call['function']['name']}")
                    
                    if tool_call["function"]["name"] == "exa_search":
                        args = json.loads(tool_call["function"]["arguments"])
                        print(f"🔍 Search query: {args.get('query')}")
                        print(f"🔍 Num results: {args.get('num_results', 10)}")
                        
                        # Execute Exa search
                        search_result = self.exa_search(
                            query=args["query"],
                            num_results=args.get("num_results", 10)
                        )
                        
                        print(f"📊 Search completed: {search_result.get('success', False)}")
                        
                        # Format results for the LLM
                        formatted_results = self.format_search_results(search_result)
                        
                        # Add tool result to messages
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": formatted_results
                        })

                # Send updated conversation back to ASI:One for final response
                print("📤 Sending final request to ASI:One with search results...")
                final_payload = {
                    "model": "asi1-extended",
                    "messages": messages,
                    "tools": [search_tool],
                    "temperature": 0.3
                }

                final_response = requests.post(
                    f"{ASI_BASE_URL}/chat/completions",
                    headers=ASI_HEADERS,
                    json=final_payload
                )

                print(f"📥 Final response status: {final_response.status_code}")
                
                if final_response.status_code == 200:
                    final_data = final_response.json()
                    
                    if "choices" in final_data and final_data["choices"]:
                        final_content = final_data["choices"][0]["message"]["content"]
                        print(f"✅ Final response length: {len(final_content)} characters")
                        return final_content
                    else:
                        return "No final response received from ASI:One"
                else:
                    return f"Final ASI:One API error: {final_response.status_code} - {final_response.text}"
            
            else:
                print("ℹ️ No tool calls made - direct response")
                return choice.get("content", "No response content received")

        except json.JSONDecodeError as e:
            return f"JSON parsing error: {str(e)}"
        except requests.RequestException as e:
            return f"Request error: {str(e)}"
        except Exception as e:
            return f"Unexpected error: {str(e)}"

# Initialize the web search agent
web_search_agent = WebSearchAgent()

# Create uAgent
agent = Agent(
    name="brandx_exa_search_agent",
    port=8080,
    seed="brandx exa search agent seed asi one",
    mailbox=True,
    endpoint=["http://localhost:8081/submit"]
)

# Initialize the chat protocol
chat_proto = Protocol(spec=chat_protocol_spec)

# Startup Handler
@agent.on_event("startup")
async def startup_handler(ctx: Context):
    ctx.logger.info(f"ASI:One Exa Search Agent started with address: {ctx.agent.address}")
    ctx.logger.info("Agent uses simple Exa search API for fast results!")
    ctx.logger.info("REST API endpoint available at: http://localhost:8081/research/brand")

# Message Handler
@chat_proto.on_message(ChatMessage)
async def handle_message(ctx: Context, sender: str, msg: ChatMessage):
    ctx.logger.info(f"Received message from {sender}")
    
    # Extract text content from the message
    user_query = ""
    for item in msg.content:
        if isinstance(item, TextContent):
            user_query = item.text
            break
    
    if not user_query:
        ctx.logger.warning("No text content found in message")
        return

    ctx.logger.info(f"Processing search query: {user_query}")
    
    try:
        # Process the query using ASI:One with Exa search
        response_text = web_search_agent.process_search_query(user_query)
        
        # Send response back to sender
        response_msg = ChatMessage(
            timestamp=datetime.utcnow(),
            msg_id=uuid4(),
            content=[TextContent(type="text", text=response_text)]
        )
        
        await ctx.send(sender, response_msg)
        ctx.logger.info(f"Sent search response to {sender}")
        
    except Exception as e:
        error_msg = f"Error processing search query: {str(e)}"
        ctx.logger.error(error_msg)
        
        error_response = ChatMessage(
            timestamp=datetime.utcnow(),
            msg_id=uuid4(),
            content=[TextContent(type="text", text=error_msg)]
        )
        
        await ctx.send(sender, error_response)

# Acknowledgement Handler
@chat_proto.on_message(ChatAcknowledgement)
async def handle_acknowledgement(ctx: Context, sender: str, msg: ChatAcknowledgement):
    ctx.logger.info(f"Received acknowledgement from {sender} for message: {msg.acknowledged_msg_id}")

# REST API Handler for Brand Research
@agent.on_rest_post("/research/brand", BrandResearchRequest, BrandResearchResponse)
async def handle_brand_research(ctx: Context, req: BrandResearchRequest) -> BrandResearchResponse:
    ctx.logger.info(f"Received brand research request for: {req.brand_name}")
    
    try:
        # Process the brand research query
        research_query = f"Research {req.brand_name} brand comprehensively including recent news, developments, products, controversies, and market position"
        response_text = web_search_agent.process_search_query(research_query)
        
        ctx.logger.info(f"Brand research completed for: {req.brand_name}")
        
        return BrandResearchResponse(
            success=True,
            brand_name=req.brand_name,
            research_result=response_text,
            timestamp=datetime.utcnow().isoformat(),
            agent_address=ctx.agent.address
        )
        
    except Exception as e:
        error_msg = f"Error processing brand research for {req.brand_name}: {str(e)}"
        ctx.logger.error(error_msg)
        
        return BrandResearchResponse(
            success=False,
            brand_name=req.brand_name,
            research_result=error_msg,
            timestamp=datetime.utcnow().isoformat(),
            agent_address=ctx.agent.address
        )

# Include the chat protocol
agent.include(chat_proto, publish_manifest=True)

if __name__ == '__main__':
    print("🚀 Starting ASI:One Exa Search Agent...")
    print(f"✅ Agent address: {agent.address}")
    print("📡 Using simple Exa search API for fast results")
    print("🧠 Powered by ASI:One AI reasoning and Exa Search")
    print("\n🌐 REST API Endpoint:")
    print("POST http://localhost:8081/research/brand")
    print("Body: {\"brand_name\": \"Tesla\"}")
    print("\n🧪 Test queries:")
    print("- 'Research Tesla brand comprehensively'")
    print("- 'Latest news about Apple'")
    print("- 'What's happening with OpenAI?'")
    print("\nPress CTRL+C to stop the agent")
    
    try:
        agent.run()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down ASI:One Exa Search Agent...")
        print("✅ Agent stopped.")
