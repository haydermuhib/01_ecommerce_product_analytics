import runpy

# Forward execution to streamlit_app.py for existing Streamlit Cloud deployments
runpy.run_path("streamlit_app.py", run_name="__main__")
