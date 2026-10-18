"""Build an English local project page and report from recorded experiment JSON only."""
from pathlib import Path
import json
import shutil
import os
import html
import numpy as np

root=Path(__file__).resolve().parents[1]
out=root/'docs/assets/research';out.mkdir(parents=True,exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR',str(root/'outputs/.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

def read(path):return json.loads((root/path).read_text())
run=read('outputs/formal_nerf/run.json');config=run['config']
test=read('outputs/formal_test/metrics.json');sweep=read('outputs/sample_sweep/metrics.json')
image=read('outputs/image_controlled/metrics.json');stereo=read('outputs/stereo_controlled/metrics.json')
curve=read('outputs/formal_nerf/curve.json');benchmark=read('outputs/nerf_benchmark/run.json')
initial_benchmark=read('outputs/nerf_benchmark_initial/run.json')
summary=test['summary'][0]
results=root/'docs/results/research';results.mkdir(parents=True,exist_ok=True)
for name,value in [('run',run),('test',test),('sample_sweep',sweep),('image',image),('stereo',stereo),('curve',curve),('benchmark',benchmark),('initial_benchmark',initial_benchmark)]:
    (results/f'{name}.json').write_text(json.dumps(value,indent=2))

fig,axes=plt.subplots(1,2,figsize=(10,3.7))
axes[0].plot([r['step'] for r in curve],[r['train_psnr_db'] for r in curve],label='Training minibatch',alpha=.7)
val=[r for r in curve if 'val_psnr_db' in r]
axes[0].plot([r['step'] for r in val],[r['val_psnr_db'] for r in val],'o-',label='Mean validation (5 views)')
axes[0].axvline(run['best_step'],color='black',linestyle=':',label='Selected checkpoint')
axes[0].set(xlabel='Optimization step',ylabel='PSNR (dB)',title='Training and validation');axes[0].legend(fontsize=8)
axes[1].plot([r['step'] for r in curve],[r['train_mse'] for r in curve]);axes[1].set(yscale='log',xlabel='Step',ylabel='Minibatch MSE',title='Training loss')
fig.tight_layout();fig.savefig(out/'training.png',dpi=160);plt.close(fig)

fig,axes=plt.subplots(1,4,figsize=(12,3.6))
axes[0].imshow(Image.open(root/'data/nerf/aurora.jpg'));axes[0].set_title('Target');axes[0].axis('off')
image_rows=[]
for ax,freq in zip(axes[1:],[0,2,6]):
    rows=[r for r in image['rows'] if r['frequency']==freq]
    mean=float(np.mean([r['psnr_db'] for r in rows]));std=float(np.std([r['psnr_db'] for r in rows],ddof=1))
    ss=float(np.mean([r['ssim'] for r in rows]));image_rows.append((freq,mean,std,ss))
    ax.imshow(Image.open(root/f'outputs/image_controlled/L{freq}_seed5670/prediction.png'))
    ax.set_title(f'L={freq}\n{mean:.2f} ± {std:.2f} dB');ax.axis('off')
fig.suptitle('Fresh coordinate-MLP training; seed 5670 shown; metrics span 3 seeds',fontsize=13)
fig.tight_layout(rect=[0,0,1,.90]);fig.savefig(out/'frequency.png',dpi=160);plt.close(fig)

ordered=sorted(test['rows'],key=lambda r:r['psnr_db'])
worst,best=ordered[0],ordered[-1]
fig,axes=plt.subplots(2,3,figsize=(9,6))
for line,row in zip(axes,[worst,best]):
    folder=root/f"outputs/formal_test/s{row['samples']}_view{row['view']}"
    for ax,name,label in zip(line,['target','prediction','error'],['Target','Prediction','Absolute RGB error [0,1]']):
        ax.imshow(Image.open(folder/f'{name}.png'));ax.axis('off');ax.set_title(label+f" (view {row['view']})" if name=='target' else label)
    line[0].set_ylabel(f"View {row['view']}")
fig.suptitle(f"Independent test examples: lowest {worst['psnr_db']:.2f} / highest {best['psnr_db']:.2f} dB (all views reported)")
fig.tight_layout();fig.savefig(out/'test.png',dpi=160);plt.close(fig)

fig,ax=plt.subplots(figsize=(6,3.6));ss=sweep['summary']
ax.plot([r['samples'] for r in ss],[r['psnr_db'] for r in ss],'o-',color='#175e89')
ax.set(xlabel='Samples per ray at rendering only',ylabel='Mean validation PSNR (dB)')
ax2=ax.twinx();ax2.plot([r['samples'] for r in ss],[r['render_seconds'] for r in ss],'s--',color='#a04624');ax2.set_ylabel('Mean render seconds',color='#a04624')
fig.tight_layout();fig.savefig(out/'samples.png',dpi=160);plt.close(fig)

fig,axes=plt.subplots(1,3,figsize=(12,4))
for ax,name in zip(axes,['ssd','sad','zncc']):
    ax.imshow(Image.open(root/f'outputs/stereo_controlled/0_3_{name}/pointcloud.png'));ax.axis('off');ax.set_title(name.upper())
fig.tight_layout();fig.savefig(out/'stereo.png',dpi=160);plt.close(fig)

stereo_rows=[]
for kernel in ['ssd','sad','zncc']:
    rows=[r for r in stereo['rows'] if r['kernel']==kernel]
    stereo_rows.append((kernel.upper(),float(np.mean([r['matching_filter_median_seconds'] for r in rows])),
                        float(np.mean([r['lr_consistency_fraction'] for r in rows])),
                        float(np.mean([r['final_valid_depth_fraction'] for r in rows])),sum(r['point_count'] for r in rows)))

def md_table(headers,rows):return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in rows)
def html_table(headers,rows):return '<table><thead><tr>'+''.join('<th>'+html.escape(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in r)+'</tr>' for r in rows)+'</tbody></table>'
it=[[f'L={f}',f'{m:.2f} ± {s:.2f}',f'{ss:.3f}'] for f,m,s,ss in image_rows]
st=[[k,f'{t:.3f}',f'{lr:.1%}',f'{v:.1%}',str(n)] for k,t,lr,v,n in stereo_rows]
sw=[[r['samples'],f"{r['psnr_db']:.2f}",f"{r['ssim']:.3f}",f"{r['render_seconds']:.3f}"] for r in sweep['summary']]
perview=[[r['view'],f"{r['psnr_db']:.2f}",f"{r['ssim']:.3f}"] for r in test['rows']]
hw=run['hardware'];step_cost=benchmark['seconds_per_training_step_after_warmup']
report=f'''# Calibrated Stereo and Audited Neural Rendering

University of Pennsylvania CIS 5800 course project · Reproduction and experimental extension · 2026-10-17

**Local review only.** Course-code and dataset redistribution permissions remain unresolved; see [publication review](PUBLICATION_REVIEW.md). This is a static-scene reconstruction project with known camera calibration, not SLAM or a dynamic world model.

## Scope and contribution

The course supplied the task, data, camera calibration, network architecture and framework. The supplied student files contain implementations but authorship of every line is not established by file presence. This extension, built with coding-assistant support, audits conventions, repairs the image-down ray sign, preserves the baseline, adds validation-selected training, independent evaluation, matched experiments and reproducible reporting. These engineering and experimental contributions do not constitute an original reconstruction algorithm. Personal resume claims should reflect work the applicant can explain and defend.

## Audit and baseline

The original implementation is retained at commit `5c29239` and in `src/vision5800/baselines/`. Legacy CLI inference still uses original rendering. In 412 cross-view SIFT matches, the known-pose median Sampson distance was 0.145 px for positive image-down y and 1.074 px for the previous negative sign. +Z already points toward the scene; it was retained. Encoding was consistent. New experiments also use midpoint samples, finite [.667,2.0] integration, black residual background and softplus density. These choices differ from the baseline's endpoint samples, infinite terminal interval and ReLU density; comparisons are not single-variable causal claims. Unit tests verify camera reprojection, finite analytic opacity, gradients and baseline behavior.

## Compute and fixed protocol

Hardware: {hw.get('gpu','CPU')}, {hw.get('vram_bytes',0)/2**30:.1f} GiB VRAM; {hw.get('cpu','CPU')}; {hw.get('ram_bytes',0)/2**30:.1f} GiB RAM. Python {hw['python']}, PyTorch {hw['torch']}, CUDA {hw['cuda_runtime']}. Float32, no mixed precision; seed {config['seed']}. Exact environment, dataset/checkpoint SHA-256 and full configuration are recorded in results/research.

A pre-training 100-step benchmark confirmed the run was affordable (conservative estimate {initial_benchmark['seconds_per_training_step_after_warmup']*config['steps']/60:.1f} minutes). Its timer initially included post-loop metadata overhead; that timing issue was fixed and a post-run repeat with a 10-step warmup measured {step_cost:.4f} s/training step, or {step_cost*config['steps']/60:.1f} projected optimization minutes excluding validation/setup. Both records are retained. The completed run took {run['elapsed_seconds']/60:.1f} minutes including validation/checkpoint writes; peak allocated GPU memory was {run.get('peak_gpu_bytes',0)/2**30:.2f} GiB.

NeRF: {config['width']}-wide eight-layer position trunk, skip connection and direction-conditioned RGB head; position/direction frequencies {config['xf']}/{config['df']}; {config['samples']} samples/ray; {config['batch']} random rays/step, sampled across 90 training views; Adam starts at {config['lr']}, decays as 0.1^(step/10000). Images remain 100×100. Training is views 0–89; validation 90/92/94/96/98; independent evaluation 100–105; other views unused. Validation every {config['validate_every']} steps chooses maximum mean PSNR; selected step {run['best_step']}. The six test views did not select weights or rendering settings. Legacy experiments had previously inspected some images; disjointness applies to this fresh model, not an external untouched benchmark.

![Training curves](assets/research/training.png)

## Independent test results

Mean per-view PSNR **{summary['psnr_db']:.2f} dB**, mean SSIM **{summary['ssim']:.3f}**, using the fixed {config['samples']}-sample renderer. PSNR uses RGB MSE in [0,1], averaged as per-view dB. SSIM uses 11×11 Gaussian windows (sigma 1.5), population covariance, K1=.01/K2=.03, valid 5-pixel crop and RGB-channel averaging. Six views of one synthetic scene are a small evaluation; no real-world generalization is claimed.

{md_table(['Test view','PSNR (dB)','SSIM'],perview)}

![Test target, prediction and error](assets/research/test.png)

## Controlled experiment 1: position encoding

Fresh 2D fits use L=0/2/6, three fixed seeds, {image['settings']['steps']} steps, batch 1024, Adam 5e-4, width 128, identical target/resolution and minibatch stream per seed. The first-layer parameter count grows with encoding dimension. Metrics below are input-image reconstruction, not held-out generalization; ± is sample standard deviation across three seeds. Historical supplied checkpoints are excluded from this controlled comparison.

{md_table(['Frequency','Mean PSNR ± SD','Mean SSIM'],it)}

![Frequency experiment](assets/research/frequency.png)

## Controlled experiment 2: samples per ray

**Rendering-only**: the selected {config['samples']}-sample-trained model is frozen; only inference samples vary. All five validation views, resolution, bounds, weights and camera convention are fixed. No retraining occurs. Timings use GPU synchronization and one warmup; they are local measurements, not production throughput benchmarks. The final test renderer remains predeclared at {config['samples']} samples regardless of this table. This experiment does not support conclusions about changing samples during training.

{md_table(['Render samples','Val PSNR','Val SSIM','Mean seconds/view'],sw)}

In this measured setting, 64 samples improves mean validation PSNR over 32; 128 adds only {sweep['summary'][3]['psnr_db']-sweep['summary'][2]['psnr_db']:.2f} dB over 64 while increasing render time from {sweep['summary'][2]['render_seconds']:.3f} to {sweep['summary'][3]['render_seconds']:.3f} seconds/view. At these tiny image sizes, launch/chunk overhead can dominate the 16-versus-32 timing comparison. These are local observations, not universal optimal settings.

![Sampling-quality/cost experiment](assets/research/samples.png)

## Stereo comparison

SSD/SAD/ZNCC use the same three pairs (0/3, 2/4, 13/15), scale .25, patch 5, rectification padding 20, depth window (.5,.6), HSV threshold 45, closing kernel 11, statistical filtering k=10/std=2 and supplied calibration. Three repeats per pair; median matching+depth+filter time excludes common rectification and output IO. Table averages per-pair values; point counts sum the three separate results (not fused/unique geometry). All ratios use every pixel in each identical rectified canvas, including background/padding. Positive-depth fraction is 100% by construction because matching is constrained to positive disparity; final filtered coverage is the meaningful reported quantity. LR consistency is not correctness.

{md_table(['Kernel','Median sec/pair (mean)','LR pass','Final depth coverage','Total points'],st)}

![Stereo point-cloud previews](assets/research/stereo.png)

No local ground-truth depth is available, so **no depth accuracy is reported**. Runtime, coverage and consistency describe different tradeoffs and do not identify the most accurate kernel.

## Failure cases and limitations

Test view {worst['view']} is the lowest-scoring view ({worst['psnr_db']:.2f} dB / SSIM {worst['ssim']:.3f}); its image/error panel is shown rather than hiding it. Inspect thin structures, edges and specular surfaces for remaining artifacts. The model uses one scene, known poses and no hierarchical sampling. Finite training and coarse quadrature can miss details; validation selection does not guarantee external generalization. Stereo has holes, outliers and scene-specific filtering; lower coverage may reject wrong matches or discard valid ones. No geometric accuracy, autonomous robot deployment or novel algorithm is established. NeRF has one training seed; only the 2D study measures seed variability.

## Reproduce

From the repository root with its environment installed:

```powershell
.\\.venv\\Scripts\\python.exe scripts/audit_coordinates.py
.\\.venv\\Scripts\\python.exe -m pytest -q
.\\.venv\\Scripts\\python.exe -m vision5800.research benchmark --output outputs/nerf_benchmark
.\\.venv\\Scripts\\python.exe -m vision5800.research train --output outputs/formal_nerf
.\\.venv\\Scripts\\python.exe -m vision5800.research test --run outputs/formal_nerf --output outputs/formal_test
.\\.venv\\Scripts\\python.exe -m vision5800.research sample-sweep --run outputs/formal_nerf --output outputs/sample_sweep
.\\.venv\\Scripts\\python.exe -m vision5800.research image --output outputs/image_controlled
.\\.venv\\Scripts\\python.exe scripts/compare_stereo.py
.\\.venv\\Scripts\\python.exe scripts/build_research_report.py
```

The training command starts fresh, never loads supplied weights. Re-running it replaces that output run, so use a new folder to preserve a run. Dataset preparation remains local; see README and PUBLICATION_REVIEW.md. Raw metrics: [test](results/research/test.json), [training](results/research/run.json), [stereo](results/research/stereo.json), [encoding](results/research/image.json), [sampling](results/research/sample_sweep.json).
'''
(root/'docs/EXPERIMENT_REPORT.md').write_text(report,encoding='utf-8')

page=root/'docs/project';page.mkdir(exist_ok=True)
styles='''*{box-sizing:border-box}body{margin:0;background:#f7f5ef;color:#20282e;font:17px/1.65 system-ui,sans-serif}main{max-width:1080px;margin:auto;padding:52px 28px}h1,h2{font-family:Georgia,serif;line-height:1.12}h1{font-size:clamp(42px,6vw,72px);max-width:920px;margin:20px 0}h2{font-size:34px;margin-top:0}h3{margin-bottom:6px}p{max-width:860px}.eyebrow{font-size:13px;letter-spacing:.16em;text-transform:uppercase;color:#175e89}.sub{font-size:21px;color:#52606a}.notice{border-left:3px solid #b87931;padding:12px 18px;background:#fff9e9;font-size:14px}.metrics{display:flex;gap:38px;flex-wrap:wrap;margin:35px 0}.metrics strong{font:42px Georgia,serif;display:block;color:#175e89}.metrics span{font-size:14px}section{padding:38px 0;border-top:1px solid #d6d8d5}img{max-width:100%;height:auto;background:white;display:block}figure{margin:26px 0}figcaption{font-size:14px;color:#5a6570;margin-top:10px}table{border-collapse:collapse;width:100%;font-size:15px}td,th{text-align:left;border-bottom:1px solid #d9dedf;padding:12px 8px}th{color:#175e89}a{color:#175e89}nav{display:flex;gap:24px;flex-wrap:wrap;margin:25px 0}code,pre{font-size:13px;background:#e9edf0}pre{padding:20px;overflow:auto}.grid{display:grid;grid-template-columns:1fr 1fr;gap:30px}@media(max-width:650px){.grid{grid-template-columns:1fr}main{padding:30px 18px}table{font-size:12px}td,th{padding:8px 4px}}'''
body=f'''<div class="eyebrow">Yixiao Liu · University of Pennsylvania · Course project & experimental extension</div>
<h1>Calibrated Stereo &amp;<br>Audited Neural Rendering</h1>
<p class="sub">From known camera poses to colored geometry and synthesized views—with traceable baselines, controlled experiments, and independent evaluation.</p>
<div class="notice">Local review draft · Course-code and dataset publication permissions remain unresolved. No public code/data release is claimed.</div>
<nav><a href="#results">Results</a><a href="#experiments">Controlled experiments</a><a href="#limits">Limitations</a><a href="../EXPERIMENT_REPORT.md">Full report</a><a href="../PUBLICATION_REVIEW.md">Publication review</a></nav>
<div class="metrics"><div><strong>{summary['psnr_db']:.2f} dB</strong><span>Mean PSNR · six test views</span></div><div><strong>{summary['ssim']:.3f}</strong><span>Mean test SSIM</span></div><div><strong>100 × 100</strong><span>Single synthetic scene · known poses</span></div></div>
<figure><img src="../assets/research/test.png" alt="Independent test target, prediction and absolute error, including the lowest-scoring view"><figcaption>Lowest- and highest-scoring test views. Absolute error uses the same [0,1] RGB scale; every test view is included in the report.</figcaption></figure>
<section id="results"><h2>A reproducible experiment, with the assumptions exposed.</h2><div class="grid"><div><h3>Geometric audit</h3><p>Camera +Z was already forward. A cross-view epipolar check exposed the incorrect image-y sign; projection tests now enforce the corrected convention. Original code and checkpoints remain traceable.</p></div><div><h3>Selection before testing</h3><p>90 training views, five validation views, six test views. Validation PSNR selects checkpoint step {run['best_step']}; test images do not choose weights or inference samples. A {config['steps']}-step RTX 4070 Ti run took {run['elapsed_seconds']/60:.1f} minutes.</p></div></div><img src="../assets/research/training.png" alt="Training loss, minibatch PSNR and mean validation PSNR curves"></section>
<section id="experiments"><div class="eyebrow">Controlled experiment 01 · fresh training</div><h2>Position encoding and image detail</h2><p>Equal optimizer, steps and minibatch streams across L=0/2/6; three seeds. Scores measure reconstruction of the fitted image. Encoding increases the first-layer parameter count.</p><img src="../assets/research/frequency.png" alt="Target and three positional encoding fits">{html_table(['Frequency','PSNR ± SD (dB)','Mean SSIM'],it)}</section>
<section><div class="eyebrow">Controlled experiment 02 · rendering only</div><h2>More samples, fixed weights</h2><p>A single checkpoint trained with {config['samples']} samples is rendered with 16/32/64/128 samples on validation views. This isolates inference settings; it does not measure changes to training sample count.</p><img src="../assets/research/samples.png" alt="Validation quality and render-time tradeoff as inference samples change">{html_table(['Samples','Val PSNR','Val SSIM','Seconds/view'],sw)}</section>
<section><div class="eyebrow">Classical geometry · matched settings</div><h2>Stereo coverage, consistency, and cost</h2><p>Three image pairs, identical resolution, patch and filtering settings. Timing is the mean of per-pair medians from three repeats. Point counts sum separate pair results. No depth ground truth is available: these are not accuracy scores.</p>{html_table(['Kernel','Seconds/pair','LR pass','Final coverage','Points'],st)}<img src="../assets/research/stereo.png" alt="SSD, SAD and ZNCC point-cloud previews for the same image pair"></section>
<section id="limits"><h2>What the results do—and do not—show</h2><p>The lowest test score is {worst['psnr_db']:.2f} dB (view {worst['view']}). Remaining artifacts, thin structures and view-dependent effects are visible in the error panels. One scene, known poses and six test views do not establish real-world generalization. Stereo retains holes and outliers; depth coverage is not depth accuracy. NeRF uses one training seed.</p><h3>Contribution boundary</h3><p>The course supplied the problem, data, architecture and framework. This extension adds convention checks, a ray-sign correction, validation-selected training, independent metrics and controlled comparisons, with coding-assistant support. It is neither a new algorithm, a SLAM system, nor a dynamic world model.</p></section>
<section><h2>Inspect and reproduce</h2><p>Recorded configurations, hardware, random seeds, checkpoint/data hashes, per-view metrics and commands are in the report. The implementation remains local pending publication clearance.</p><nav><a href="../EXPERIMENT_REPORT.md">Experiment report</a><a href="../AUDIT.md">Audit notes</a><a href="../PROVENANCE.md">Provenance</a></nav><pre>python -m vision5800.research train --output outputs/formal_nerf
python -m vision5800.research test --run outputs/formal_nerf --output outputs/formal_test
python scripts/compare_stereo.py</pre></section>'''
(page/'index.html').write_text('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Calibrated Stereo & Audited Neural Rendering — Yixiao Liu</title><style>'+styles+'</style></head><body><main>'+body+'</main></body></html>',encoding='utf-8')
print('Wrote report and local project page.')
