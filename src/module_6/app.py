import uvicorn

from src.module_6.routes import create_app

app = create_app()

# Execute with: poetry run uvicorn module_6.app:app --reload --app-dir src
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
