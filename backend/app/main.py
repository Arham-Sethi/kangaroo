"""FastAPI application factory."""


def create_app():
    """Create and configure the FastAPI application."""
    from fastapi import FastAPI

    app = FastAPI(
        title="Kangaroo Shift",
        description="Context management platform for seamless LLM switching",
        version="0.1.0",
    )

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "ok", "version": "0.1.0"}

    return app
