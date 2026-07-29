import urllib.request, urllib.parse, json, time, os

BASE = 'http://127.0.0.1:8000'

def multipart_post(url, filepath):
    """Upload image as multipart/form-data to detect endpoint."""
    boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
    filename = os.path.basename(filepath)
    
    with open(filepath, 'rb') as f:
        img_data = f.read()
    
    # Build multipart body manually
    body = (
        f'--{boundary}\r\n'
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f'Content-Type: image/jpeg\r\n\r\n'
    ).encode() + img_data + f'\r\n--{boundary}--\r\n'.encode()
    
    req = urllib.request.Request(
        url,
        data=body,
        headers={'Content-Type': f'multipart/form-data; boundary={boundary}'}
    )
    
    start = time.time()
    with urllib.request.urlopen(req, timeout=30) as resp:
        latency = round((time.time() - start) * 1000)
        return json.loads(resp.read().decode()), latency

# ---- Test 1: Valid image detection ----
print("=" * 70)
print("TEST: POST /api/detect/image")
print("=" * 70)
img_path = 'input_images/synthetic_pool_balls_000001.jpg'
try:
    result, latency = multipart_post(f'{BASE}/api/detect/image', img_path)
    print(f"[PASS] Status: 200  |  Latency: {latency}ms")
    print(f"  Detections: {len(result['detections'])}")
    print(f"  Processing time: {result['processing_time']*1000:.1f}ms")
    print(f"  Annotated image URL: {result['annotated_image_url']}")
    print()
    print("  Top detections:")
    for d in result['detections'][:5]:
        print(f"    [{d['class_name']}] conf={d['confidence']:.2f}  center=({d['center_x']:.0f}, {d['center_y']:.0f})")
except Exception as e:
    print(f"[FAIL] {e}")

# ---- Test 2: Invalid file type ----
print()
print("=" * 70)
print("TEST: POST /api/detect/image (invalid type .txt)")
print("=" * 70)
boundary = '----Boundary'
body = (
    f'--{boundary}\r\n'
    f'Content-Disposition: form-data; name="file"; filename="test.txt"\r\n'
    f'Content-Type: text/plain\r\n\r\n'
    f'this is not an image'
    f'\r\n--{boundary}--\r\n'
).encode()
req = urllib.request.Request(f'{BASE}/api/detect/image', data=body, headers={'Content-Type': f'multipart/form-data; boundary={boundary}'})
try:
    with urllib.request.urlopen(req, timeout=10) as r:
        print(f"[UNEXPECTED PASS] Status: {r.status}")
except urllib.error.HTTPError as e:
    print(f"[PASS] Correctly rejected with HTTP {e.code} (422 Unprocessable)")
    
# ---- Test 3: Coordinate mapping at edge ----
print()
print("=" * 70)
print("TEST: POST /api/mapping (corner coordinates)")
print("=" * 70)
for px, py in [(0, 0), (800, 600), (100, 100), (700, 500)]:
    try:
        payload = json.dumps({'pixel_x': float(px), 'pixel_y': float(py)}).encode()
        req = urllib.request.Request(f'{BASE}/api/mapping', data=payload, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=5) as resp:
            r = json.loads(resp.read().decode())
            print(f"  Pixel ({px:4d},{py:4d}) -> World ({r['world_x']:.1f}mm, {r['world_y']:.1f}mm)")
    except Exception as e:
        print(f"  [FAIL] ({px},{py}) -> {e}")

print()
print("All tests complete!")
