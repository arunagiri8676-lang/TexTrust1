import streamlit as st
import numpy as np
import cv2
import glob
import os

SIZE, GRID = 512, 8
P = SIZE // GRID
MATCH_MIN = 15      # minimum matched points to say "recognised"
GENUINE_MIN = 70    # texture score needed for "genuine"

# ---------- texture features ----------
def lbp_hist(g):
    c = g[1:-1, 1:-1]
    code = np.zeros(c.shape, np.uint8)
    sh = [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]
    for i, (dy, dx) in enumerate(sh):
        n = g[1+dy:g.shape[0]-1+dy, 1+dx:g.shape[1]-1+dx]
        code |= ((n >= c).astype(np.uint8) << i)
    h, _ = np.histogram(code // 32, bins=8, range=(0, 8))
    return h / h.sum()

def patch_feats(p):
    pf = p.astype(np.float32)
    lap = cv2.Laplacian(pf, cv2.CV_32F).var()
    gx = cv2.Sobel(pf, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(pf, cv2.CV_32F, 0, 1)
    grad = np.sqrt(gx**2 + gy**2).mean()
    F = np.abs(np.fft.fftshift(np.fft.fft2(pf - pf.mean())))
    yy, xx = np.indices(F.shape)
    r = np.hypot(yy - F.shape[0]/2, xx - F.shape[1]/2)
    hf = F[r > 0.25 * F.shape[0]].sum() / (F.sum() + 1e-9)
    return np.concatenate([[np.log1p(lap), np.log1p(grad), hf], lbp_hist(p)])

def extract(img):
    img = cv2.resize(img, (SIZE, SIZE))
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    f = [patch_feats(g[y:y+P, x:x+P])
         for y in range(0, SIZE, P) for x in range(0, SIZE, P)]
    return img, np.array(f)

def variants(img):
    h, w = img.shape[:2]
    out = [img]
    rng = np.random.RandomState(0)
    for k in range(1, 6):
        dy, dx = int(h * 0.02 * k), int(w * 0.02 * k)
        v = cv2.convertScaleAbs(img[dy:h-dy, dx:w-dx], alpha=1 + 0.03*k, beta=2*k)
        v = np.clip(v.astype(np.float32) + rng.randn(*v.shape) * 1.5, 0, 255).astype(np.uint8)
        out.append(v)
    return out

def texture_stats(imgs):
    f = np.vstack([extract(a)[1] for im in imgs for a in variants(im)])
    mu = f.mean(0)
    sd = np.maximum(f.std(0), 0.1 * np.abs(mu) + 1e-3)
    return mu, sd

def texture_score(stats, test):
    mu, sd = stats
    img, f = extract(test)
    z = np.abs((f - mu) / sd).mean(1)
    return float(100 * np.exp(-(z.mean() / 2.5) ** 2)), z, img

# ---------- identify which medicine ----------
orb = cv2.ORB_create(1500)

def signature(img):
    h, w = img.shape[:2]
    s = 800 / max(h, w)
    g = cv2.cvtColor(cv2.resize(img, (int(w*s), int(h*s))), cv2.COLOR_BGR2GRAY)
    return orb.detectAndCompute(g, None)

def match_count(a, b):
    ka, da = a
    kb, db = b
    if da is None or db is None or len(ka) < 10 or len(kb) < 10:
        return 0
    m = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    good = [x[0] for x in m if len(x) == 2 and x[0].distance < 0.75 * x[1].distance]
    if len(good) < 10:
        return 0
    A = np.float32([ka[g.queryIdx].pt for g in good])
    B = np.float32([kb[g.trainIdx].pt for g in good])
    H, mask = cv2.findHomography(A, B, cv2.RANSAC, 5.0)
    return int(mask.sum()) if mask is not None else 0

# ---------- load medicines from the refs folder (only the host can add) ----------
@st.cache_resource
def load_db():
    db = {}
    for folder in sorted(glob.glob("refs/*/")):
        name = os.path.basename(os.path.normpath(folder)).replace("_", " ")
        imgs = [cv2.imread(p) for p in sorted(glob.glob(folder + "*"))]
        imgs = [i for i in imgs if i is not None]
        if imgs:
            db[name] = {"sigs": [signature(i) for i in imgs],
                        "stats": texture_stats(imgs)}
    return db

def load(file):
    return cv2.imdecode(np.frombuffer(file.getvalue(), np.uint8), cv2.IMREAD_COLOR)

# ---------- app ----------
st.set_page_config(page_title="TexTrust", page_icon="💊")
st.title("💊 TexTrust")
st.caption("Counterfeit medicine check using packaging micro-texture")

db = load_db()
if not db:
    st.error("No medicines found. Add photos in GitHub under refs/<Medicine_Name>/1.jpg")
    st.stop()
st.caption("Medicines in database: " + ", ".join(db.keys()))

if "hist" not in st.session_state:
    st.session_state.hist = []

shot = st.camera_input("Scan a medicine pack")
if shot is None:
    shot = st.file_uploader("...or upload a photo", type=["jpg", "jpeg", "png"])

if shot:
    test = load(shot)
    sig = signature(test)
    best, best_m = None, 0
    for name, d in db.items():
        m = max(match_count(sig, s) for s in d["sigs"])
        if m > best_m:
            best, best_m = name, m

    if best is None or best_m < MATCH_MIN:
        st.subheader("🚫 Not a recognised medicine pack")
        st.write("This does not match any medicine in the database.")
        verdict = "Not recognised"
    else:
        score, z, img = texture_score(db[best]["stats"], test)
        heat = cv2.resize(z.reshape(GRID, GRID).astype(np.float32), (SIZE, SIZE))
        heat = cv2.applyColorMap((np.clip(heat / 4, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
        over = cv2.addWeighted(img, 0.55, heat, 0.45, 0)

        st.subheader("💊 Medicine identified")
        st.header(best)
        if score >= GENUINE_MIN:
            st.success("✅ Looks genuine")
            verdict = "Genuine"
        else:
            st.error("❌ Fake / altered pack suspected")
            verdict = "Fake/altered"
        st.metric("Authenticity score", f"{score:.0f} / 100")
        c1, c2 = st.columns(2)
        c1.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Scanned")
        c2.image(cv2.cvtColor(over, cv2.COLOR_BGR2RGB), caption="Red = suspicious regions")
        if verdict == "Fake/altered":
            st.button("🚩 Report suspect pack")
    st.session_state.hist.append((best if verdict != "Not recognised" else "-", verdict))

if st.session_state.hist:
    st.header("Scan history")
    for n, v in reversed(st.session_state.hist[-10:]):
        st.write(f"{n}: {v}")