# Demo authentication

You can enable the simple demo auth without exporting an environment variable by placing a token in a local file.

The server looks for the token in this order:

1) Environment variable: `DEMO_TOKEN`
2) Path in environment: `DEMO_TOKEN_FILE` (points to a text file containing the token)
3) Local files in the repository (first match wins):
   - `.demo_token`
   - `demo_token.txt`
   - `config/demo_token.txt`

The file should contain only the token string (one line, no quotes). Example:

```text
petercodes
```

When a token is configured, the app:

- Redirects `/` to `/login` until you submit the token
- Sets an HttpOnly cookie on successful login so all API calls and the websocket work without headers
- Requires the token for `/api/*` and `/ws`

Notes:

- For public/hosted demos (Render, Railway, etc.), using an environment variable is more secure than committing a token file.
- If you do use a file locally, consider adding it to `.gitignore` so it’s not committed.
