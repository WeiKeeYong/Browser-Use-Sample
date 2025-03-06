import asyncio, os, sys, time
from datetime import datetime
from dotenv import load_dotenv
from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr
from browser_use import Agent

# Load environment variables
load_dotenv()

def read_api_key_from_file(file_path: str, start_with: str) -> str:
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if line.startswith(start_with):
                    return line.strip().split(':')[1].strip()
        raise ValueError(f"Key not found starting with '{start_with}' in file '{file_path}'")
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    except ValueError as e:
        print(str(e))
        sys.exit(1)

# Set up LangChain tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = read_api_key_from_file(r'D:\codes\keys\keys.txt', 'LangSmith:')

def get_llm(model_type: Literal['openai', 'google', 'deepseek']):
    keys_file = r'D:\codes\keys\keys.txt'
    if model_type == 'openai':
        return ChatOpenAI(
            model="gpt-4o", 
            api_key=read_api_key_from_file(keys_file, 'OPENAPI-ALL-Access:')
        )
    
    elif model_type == 'google':
        api_key = read_api_key_from_file(keys_file, 'GOOGLE_AI_KEY:')
        return ChatGoogleGenerativeAI(
            model='gemini-2.0-flash-exp', 
            api_key=SecretStr(api_key)
        )
    
    elif model_type == 'deepseek':
        api_key = read_api_key_from_file(keys_file, 'DeepSeekAPI01:')
        return ChatOpenAI(
            base_url='https://api.deepseek.com/v1', 
            model="deepseek-chat", 
            api_key=SecretStr(api_key)
        )
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

async def main(model_type: Literal['openai', 'google', 'deepseek'] = 'openai') -> str:
    start_time = time.time()
    start_datetime = datetime.now()
    print(f"🕒 Task Started at: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")

    task = "Go to ebay, find me the 'product amd max+ 395 notebook', and give me the price of the first item you see "
    
    llm = get_llm(model_type)
    
    agent = Agent(
        task=task,
        llm=llm,
        use_vision=False,
    )
    result = await agent.run()
    
    # Record end time
    end_time = time.time()
    end_datetime = datetime.now()

    # Calculate total execution time
    execution_time = end_time - start_time

    # Format the results as a multi-line string
    execution_details = (
        f"\nExecution Details:\n"
        f"Model Used: {model_type.upper()}\n"
        f"Start Time: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"End Time: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Total Execution Time: {execution_time:.2f} seconds"
    )
    
    print("\n📝 Result:")
    print(result)

    return execution_details

if __name__ == '__main__':
    async def run_all_models():
        # Run each model and collect results
        openai_details = await main('openai')
        google_details = await main('google')
        deepseek_details = await main('deepseek')
        
        # Print all execution details together at the end
        log_string = "\n\n=== EXECUTION SUMMARY ===\n"
        log_string += openai_details + "\n"
        log_string += google_details + "\n"
        log_string += deepseek_details + "\n"
        log_string += "=========================\n"
        print(log_string)

    # Run all models in sequence
    asyncio.run(run_all_models())