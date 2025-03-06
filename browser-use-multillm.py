import asyncio, os, sys, time
from datetime import datetime
from dotenv import load_dotenv
from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr
from browser_use import Agent
from browser_use.browser.browser import Browser
from browser_use.browser.browser import BrowserConfig
from browser_use.browser.context import BrowserContextConfig, BrowserContext

# Load environment variables
load_dotenv()
debug_on = True
lang_smith = True

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

if debug_on and lang_smith:
    # Set up LangChain tracing
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_API_KEY"] = read_api_key_from_file(r'D:\codes\keys\keys.txt', 'LangSmith:') #Please change to your key path

def get_llm(model_type: Literal['openai-4o', 'openai-4o-mini','google', 'deepseek']):
    keys_file = r'D:\codes\keys\keys.txt' #Please change to your key path
    if model_type == 'openai-4o':
        return ChatOpenAI(
            model="gpt-4o", 
            api_key=read_api_key_from_file(keys_file, 'OPENAPI-ALL-Access:')
        )
    
    elif model_type == 'openai-4o-mini':
        return ChatOpenAI(
            model="gpt-4o-mini", 
            api_key=read_api_key_from_file(keys_file, 'OPENAPI-ALL-Access:')
        )
    
    elif model_type == 'google':
        return ChatGoogleGenerativeAI(
            model='gemini-2.0-flash-exp', 
            api_key=SecretStr(read_api_key_from_file(keys_file, 'GOOGLE_AI_KEY:'))
        )
    
    elif model_type == 'deepseek':
        return ChatOpenAI(
            base_url='https://api.deepseek.com/v1', 
            model="deepseek-chat", 
            api_key=SecretStr(read_api_key_from_file(keys_file, 'DeepSeekAPI01:'))
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

#Set browser setting, seem like many ways to set it up. Another way is to use BrowserContextConfig directly
browser = Browser(
	config=BrowserConfig(
		new_context_config=BrowserContextConfig(
			viewport_expansion=0,
			browser_window_size={'width': 1920, 'height': 1080},
			locale='en-US',
			user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.102 Safari/537.36',
		)
	)
)

async def main(model_type: Literal['openai-4o', 'openai-4o-mini','google', 'deepseek'] = 'openai-4o') -> str:
    start_time = time.time()
    start_datetime = datetime.now()
    print(f"🕒 Task Started at: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    task = "Go to ebay, search for me 'amd max+ 395 notebook', and give me the price of the first item you see "
    
    llm = get_llm(model_type)
    agent = Agent(
        task=task,
        llm=llm,
        use_vision=False,
        browser=browser,
    )
    result = await agent.run(max_steps=10)
    await browser.close()

    # Record end time, cal exec time
    end_time = time.time()
    end_datetime = datetime.now()
    execution_time = end_time - start_time

    # Constructe the data
    execution_details = (
        f"\nExecution Details:\n"
        f"Model Used: {model_type.upper()}\n"
        f"Start Time: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"End Time: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Total Execution Time: {execution_time:.2f} seconds"
    )
  
    model_outputs_data = result.model_outputs()

    done_text = None

    for agent_output in model_outputs_data: # need to find better way to get the done text. 
        for action in agent_output.action:  # Iterate over each ActionModel in the action list
            if action.done:  # Check if there is a done action
                done_text = action.done.text  # Extract the text
                break  # Break after finding the first done action
    if done_text:
         execution_details += f"\nResult: {done_text}"

    if debug_on:
        print("\nResult:")
        print(result)
        print("\nType:")
        print(type(result))
        print("\nDir:")
        print(dir(result))

    return execution_details

if __name__ == '__main__':
    async def run_all_models():
        # Run each model and collect results
       openai_details = await main('openai-4o')
       openai_mini_details = await main('openai-4o-mini')
       google_details = await main('google')
       deepseek_details = await main('deepseek')
        
       if debug_on:
           log_string = "\n\n=== EXECUTION SUMMARY ===\n"
           log_string += openai_details + "\n"
           log_string += openai_mini_details + "\n"
           log_string += google_details + "\n"
           log_string += deepseek_details + "\n"
           log_string += "=========================\n"
           print(log_string)

    asyncio.run(run_all_models())
