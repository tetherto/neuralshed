# Collection Builder

A FastAPI backend and a React frontend (Vite + TypeScript).

## Develop (two terminals)

To install the dependencies and run the backend:
```bash
pip install fastapi "uvicorn[standard]"
cd collection_builder
uvicorn backend.main:app --reload --port 8000
````

To install the dependencies and run the frontend:
```bash
cd collection_builder/frontend
npm install
npm run dev                                        # http://localhost:5173
```

Open http://localhost:5173. Both sides hot-reload on save.

## Production (single server)

```bash
cd collection_builder/frontend && npm run build    # emits frontend/dist
cd .. && uvicorn backend.main:app --port 8000      # serves the app + API at :8000
```

Open http://localhost:8000. The backend mounts `frontend/dist` at `/` and the API at `/api`, reading the SQLite database at `src/neuralshed/database/neuralshed_metadata.db`.
