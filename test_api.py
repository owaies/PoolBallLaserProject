import urllib.request, json, time, sys

BASE = 'http://127.0.0.1:8000'
results = []

def test(name, url, method='GET', data=None):
    try:
        start = time.time()
        if method == 'GET':
            req = urllib.request.Request(url)
        else:
            payload = json.dumps(data).encode()
            req = urllib.request.Request(url, data=payload, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=10) as resp:
            latency = round((time.time() - start)*1000)
            body = json.loads(resp.read().decode())
            results.append({'test': name, 'status': 'PASS', 'code': resp.status, 'latency_ms': latency, 'sample': str(body)[:120]})
    except urllib.error.HTTPError as e:
        latency = round((time.time() - start)*1000)
        results.append({'test': name, 'status': f'HTTP {e.code}', 'code': e.code, 'latency_ms': latency, 'error': str(e)})
    except Exception as e:
        results.append({'test': name, 'status': 'FAIL', 'error': str(e)})

# 1. Health
test('GET /api/health', f'{BASE}/api/health')

# 2. Model info
test('GET /api/model', f'{BASE}/api/model')

# 3. Calibration
test('GET /api/calibration', f'{BASE}/api/calibration')

# 4. Statistics
test('GET /api/statistics', f'{BASE}/api/statistics')

# 5. Logs
test('GET /api/logs', f'{BASE}/api/logs?lines=5')

# 6. Coordinate mapping - valid
test('POST /api/mapping (valid)', f'{BASE}/api/mapping', 'POST', {'pixel_x': 400.0, 'pixel_y': 300.0})

# 7. Coordinate mapping - missing fields
test('POST /api/mapping (invalid)', f'{BASE}/api/mapping', 'POST', {})

# 8. Detect folder - traversal blocked
test('POST /api/detect/folder (traversal)', f'{BASE}/api/detect/folder', 'POST', {'folder_path': '../datasets'})

# Print results
print('='*80)
print('BACKEND API ENDPOINT TEST RESULTS')
print('='*80)
for r in results:
    print()
    print(f"[{r['status']}] {r['test']}")
    if 'latency_ms' in r:
        print(f"  Latency: {r['latency_ms']}ms  |  HTTP: {r.get('code','')}")
    if 'sample' in r:
        print(f"  Response: {r['sample']}")
    if 'error' in r and r['status'] == 'FAIL':
        print(f"  Error: {r['error']}")
print()
print('='*80)
passes = sum(1 for r in results if r['status'] == 'PASS')
print(f"SUMMARY: {passes}/{len(results)} tests passed")
print('='*80)
