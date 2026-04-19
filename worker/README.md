# unit-demo-config Worker

A tiny Cloudflare Worker that backs the `Save as Initial View` button on
the UniT demo page. Stores a single JSON blob (`DEMO_CONFIGS`) in
Workers KV; the live page fetches it on load, the admin panel writes to
it with an API key.

## Endpoints

| Method | Path      | Auth                   | Purpose                                  |
|--------|-----------|------------------------|------------------------------------------|
| GET    | `/config` | public                 | Returns the stored `DEMO_CONFIGS` (or `{}`). |
| POST   | `/config` | `X-Admin-Key` header   | Replaces the stored config with the JSON body. |

## One-time deploy

```bash
cd webpage/worker

# 1. Log in (opens a browser; Cloudflare SSO via the default profile).
npx wrangler login

# 2. Create the KV namespace and paste the returned id into wrangler.toml.
npx wrangler kv namespace create CONFIG

# 3. Set the admin secret (you pick this string — it's what the
#    `?setting=<...>` query param in the page must match).
npx wrangler secret put ADMIN_SECRET

# 4. Deploy.
npx wrangler deploy
```

The final `wrangler deploy` prints the Worker URL. Paste it into the
`CONFIG_WORKER_URL` constant in `webpage/script.js`.

## Why the admin key lives in the URL

The only URL that exposes the save UI is `?setting=<ADMIN_SECRET>`.
Anyone with that URL can write; everyone else reads anonymously. No
GitHub PAT, no per-user login — you just share the magic URL with
whoever should be able to save.
