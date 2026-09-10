try:
    from converted_app import app
except Exception as e:
    import traceback
    from fastapi import FastAPI
    from fastapi.responses import PlainTextResponse
    app = FastAPI()
    err_msg = traceback.format_exc()

    @app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE"])
    async def debug_error(full_path: str = ""):
        return PlainTextResponse(f"FastAPI Startup Exception:\n{err_msg}", status_code=500)