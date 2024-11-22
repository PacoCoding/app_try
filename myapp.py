#%%
import os
import asyncio
from streamlit.web.bootstrap import load_config_options
from streamlit.web.server.server import Server

if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))

    # Set Streamlit options
    flag_options = {
        "server.port": 8501,  # Default port
        "global.developmentMode": False,
    }

    # Load Streamlit configuration options
    load_config_options(flag_options=flag_options)
    flag_options["_is_running_with_streamlit"] = True

    # Create and start the Streamlit server
    async def run_streamlit():
        server = Server("./application/hello.py", flag_options=flag_options)
        await server.start()
        await server.stopped

    # Use asyncio.run only if not already in an event loop
    try:
        asyncio.run(run_streamlit())
    except RuntimeError as e:
        if "asyncio.run() cannot be called from a running event loop" in str(e):
            loop = asyncio.get_event_loop()
            loop.run_until_complete(run_streamlit())
