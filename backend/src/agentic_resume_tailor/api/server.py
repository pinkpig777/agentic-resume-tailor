from __future__ import annotations

import uvicorn

from agentic_resume_tailor.api.app import create_app
from agentic_resume_tailor.settings import load_settings

app = create_app()


def main() -> None:
    settings = load_settings()
    uvicorn.run(app, host="0.0.0.0", port=settings.port)


if __name__ == "__main__":
    main()
