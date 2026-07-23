"""Recompute targeted ASR from saved <stem>.boxes.json — CPU only, no model, no rerun.
Three metrics per run:
  original : flipped = van@box in patched                 / eligible          (buggy baseline)
  strict   : exclude ambiguous (van@box in CLEAN);
             flipped = van@box in patched (non-ambiguous)  / (eligible-ambiguous)
  new-van  : flipped = van@box in patched AND NOT in clean / eligible
lost rate unchanged. Ambiguous = eligible car that already has a van@box in the CLEAN image.
"""
import glob, os, json
import patch_attack_targeted as T
IMG = 640
LBL = "datasets/VisDrone/labels/test"

RUNS = [
    ("b0_clean",        "patch_examples_b0_clean"),
    ("s2 (ctr20 marg)", "patch_examples_s2_adv_20"),
    ("s2b (ctr20)",     "patch_examples_s2b_adv_20_hinge"),
    ("s2d (off20)",     "patch_examples_s2d_offset_20"),
    ("s2e (off10)",     "patch_examples_s2e_offset_10"),
    ("s2h (ctr10)",     "patch_examples_s2h_center_10"),
    ("s2i (ctr05)",     "patch_examples_s2i_center_05"),
    ("03 (s2l)",        "patch_examples_s2l_center_03"),
    ("04 (s2j)",        "patch_examples_s2j_center_04"),
    ("05@160 (s2n)",    "patch_examples_s2n_center_05_e160"),
    ("07 (s2k)",        "patch_examples_s2k_center_07"),
    ("15 (s2m)",        "patch_examples_s2m_center_15"),
]

def recompute(run_dir):
    elig = amb = orig = newvan = lost = 0
    for jf in glob.glob(os.path.join(run_dir, "*.boxes.json")):
        stem = os.path.basename(jf)[:-len(".boxes.json")]
        d = json.load(open(jf)); clean, patched = d["clean"], d["patched"]
        gt, _ = T.load_car_boxes(os.path.join(LBL, stem + ".txt"))
        for (c, xc, yc, w, h) in gt:
            bpx = ((xc-w/2)*IMG, (yc-h/2)*IMG, (xc+w/2)*IMG, (yc+h/2)*IMG)
            if not T.detected_as(clean, T.SRC_IDX, bpx):
                continue
            elig += 1
            van_c = T.detected_as(clean,   T.TGT_IDX, bpx)
            van_p = T.detected_as(patched, T.TGT_IDX, bpx)
            car_p = T.detected_as(patched, T.SRC_IDX, bpx)
            if van_c: amb += 1
            if van_p: orig += 1
            if van_p and not van_c: newvan += 1
            if (not van_p) and (not car_p): lost += 1
    strict_den = elig - amb
    return dict(elig=elig, amb=amb,
                asr_orig=orig/elig if elig else 0.0,
                asr_strict=newvan/strict_den if strict_den else 0.0,
                asr_newvan=newvan/elig if elig else 0.0,
                lost=lost/elig if elig else 0.0)

print(f"{'run':16} {'elig':>5} {'amb':>4} {'orig':>6} {'strict':>7} {'new-van':>7} {'lost':>6}")
ambs=[]
for label, d in RUNS:
    if not os.path.isdir(d):
        print(f"{label:16} (dir missing: {d})"); continue
    r = recompute(d); ambs.append((label, r['amb'], r['elig']))
    print(f"{label:16} {r['elig']:>5} {r['amb']:>4} {r['asr_orig']:>6.3f} "
          f"{r['asr_strict']:>7.3f} {r['asr_newvan']:>7.3f} {r['lost']:>6.3f}")

print("\n### ambiguous count across runs (is it constant?) ###")
avals=[a for _,a,_ in ambs]; evals=[e for _,_,e in ambs]
print(f"  eligible : min {min(evals)} max {max(evals)}  ({'CONSTANT' if min(evals)==max(evals) else 'VARIES'})")
print(f"  ambiguous: min {min(avals)} max {max(avals)}  ({'CONSTANT' if min(avals)==max(avals) else 'VARIES'})")
for label,a,e in ambs: print(f"    {label:16} amb={a} elig={e}")
