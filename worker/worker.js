/**
 * UniT demo-config Worker
 * ----------------------------------------------------------------------
 *   GET  /config          → returns the stored per-scene config JSON
 *                            (or `{}` on a cold KV).  Public.
 *   POST /config          → overwrites the stored config.  Requires the
 *                            X-Admin-Key header to match env.ADMIN_SECRET.
 *
 * Deploy:
 *   - KV namespace binding:  CONFIG   (stores the single key "DEMO_CONFIGS")
 *   - Secret:                ADMIN_SECRET
 *
 * Why a Worker?  The page needs a backend that survives reloads so the
 * admin can bake in camera tweaks without redeploying the static site.
 * KV writes are globally readable within ~60s, which is as "instant" as
 * we need; Workers Free tier covers 100k req/day comfortably.
 */

const CORS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type, X-Admin-Key',
  'Access-Control-Max-Age': '86400',
};

const KEY = 'DEMO_CONFIGS';

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (request.method === 'OPTIONS') {
      return new Response(null, { headers: CORS });
    }
    if (url.pathname !== '/config') {
      return new Response('Not found', { status: 404, headers: CORS });
    }

    if (request.method === 'GET') {
      const stored = await env.CONFIG.get(KEY, 'json');
      return new Response(JSON.stringify(stored || {}), {
        headers: { 'Content-Type': 'application/json', ...CORS },
      });
    }

    if (request.method === 'POST') {
      const key = request.headers.get('X-Admin-Key') || '';
      if (!env.ADMIN_SECRET || key !== env.ADMIN_SECRET) {
        return new Response('Unauthorized', { status: 401, headers: CORS });
      }
      let payload;
      try {
        payload = await request.json();
      } catch {
        return new Response('Bad JSON', { status: 400, headers: CORS });
      }
      if (payload === null || typeof payload !== 'object' || Array.isArray(payload)) {
        return new Response('Payload must be a JSON object', { status: 400, headers: CORS });
      }
      await env.CONFIG.put(KEY, JSON.stringify(payload));
      return new Response(JSON.stringify({ ok: true }), {
        headers: { 'Content-Type': 'application/json', ...CORS },
      });
    }

    return new Response('Method not allowed', { status: 405, headers: CORS });
  },
};
