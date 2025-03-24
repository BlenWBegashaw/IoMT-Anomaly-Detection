from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()

# ✅ Get the GitHub Codespace domain dynamically
CODESPACE_NAME = os.getenv("CODESPACE_NAME", "")
CODESPACE_DOMAIN = f"https://{CODESPACE_NAME}-8000.github.dev" if CODESPACE_NAME else ""

# ✅ Allow specific origins (including GitHub Codespaces)
allowed_origins = [
    "https://*.github.dev",
    "https://*.githubpreview.dev",
    CODESPACE_DOMAIN,  # Automatically allow this Codespace URL
]

# ✅ CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,  # Restrict CORS to only GitHub Codespaces
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

@app.get("/")
async def root():
    return {"message": "Welcome to the FastAPI server!"}


# ✅ Run FastAPI Server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
