# Interactive Brokers Web API - CLAUDE.md

## Build/Run Commands
- Build and start container: `docker-compose up -d`
- Shell into container: `docker exec -it ibkr bash`
- Run server: `sh ./start.sh`
- Run Flask app only: `flask --app webapp/app run --debug -p 5056 -h 0.0.0.0`

## Code Style Guidelines
- **Imports**: Group standard library imports first, then third-party imports
- **Formatting**: 4-space indentation, double blank lines between route functions
- **Types**: Use type hints where applicable (see `# type: ignore` example in scripts)
- **Variables**: Use `snake_case` for variables and functions
- **Constants**: Use `UPPERCASE` for constants
- **API Calls**: Use f-strings for URL construction
- **Error Handling**: Use try/except blocks with specific exceptions when possible
- **HTML Templates**: Use Jinja2 templating with consistent indentation
- **API Pattern**: Request → Parse JSON → Render template

## Project Structure
- Flask app in `webapp/app.py`
- Templates in `webapp/templates/`
- Docker setup in `Dockerfile` and `docker-compose.yml`
- Example scripts in `scripts/`